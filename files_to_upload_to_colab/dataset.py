"""
Dataset utilities for gallstone binary classification.
Reads images from stage3/ (fan-masked, inpainted PNGs).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T

LABEL_MAP = {"stones": 1, "normal": 0}

# normal frames cost 6.9x more in the loss (inverse class frequency: 83/12 ≈ 6.9)
CLASS_WEIGHTS = {"stones": 1.0, "normal": 6.9}


# ── transforms ───────────────────────────────────────────────────────────────

def _pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def get_transforms(augment: bool = False) -> T.Compose:
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        return T.Compose([
            T.Lambda(_pad_to_square),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3),   # brightness/contrast only — images are grayscale
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.Lambda(_pad_to_square),
        T.Resize((224, 224)),
        T.ToTensor(),
        norm,
    ])


# ── dataset ──────────────────────────────────────────────────────────────────

class GallstoneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stage3_root: Path, transform=None):
        self.records   = df.reset_index(drop=True)
        self.root      = Path(stage3_root)
        self.transform = transform or get_transforms(augment=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        # processed_relpath: "processed/stones/file.png" → stage3/stones/file.png
        rel   = Path(row["processed_relpath"]).relative_to("processed")
        img   = Image.open(self.root / rel).convert("RGB")
        label = LABEL_MAP[row["label"]]
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── manifest loading ─────────────────────────────────────────────────────────

def load_binary_frames(manifest_path: str | Path, stage3_root: str | Path) -> pd.DataFrame:
    """Load manifest, filter to binary labels, check stage3/ files exist."""
    df = pd.read_csv(manifest_path)
    binary = df[
        df["label"].isin(["stones", "normal"]) &
        df["processed_relpath"].notna() &
        (df["processed_relpath"] != "")
    ].copy()

    stage3_root = Path(stage3_root)
    exists = binary["processed_relpath"].apply(
        lambda p: (stage3_root / Path(p).relative_to("processed")).exists()
    )
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} files not found in stage3/, skipping")
    return binary[exists].reset_index(drop=True)


# ── splits ───────────────────────────────────────────────────────────────────

def _patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with their label (for stratification)."""
    return (
        df.groupby("patient_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )


def make_patient_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified split. Returns (train_df, val_df, test_df).
    Duplicates are kept in train, dropped from val/test.
    """
    from sklearn.model_selection import train_test_split

    pat = _patient_labels(df)
    n   = len(pat)

    def _split(ids, labels, test_size, rng):
        try:
            return train_test_split(ids, test_size=test_size, stratify=labels, random_state=rng)
        except ValueError:
            # fallback when a class has too few members to stratify
            return train_test_split(ids, test_size=test_size, random_state=rng)

    n_test = max(1, round(n * test_frac))
    p_trainval, p_test = _split(pat["patient_id"].tolist(), pat["label"].tolist(), n_test, seed)

    remaining = pat[pat["patient_id"].isin(p_trainval)]
    n_val = max(1, round(n * val_frac))
    p_train, p_val = _split(
        remaining["patient_id"].tolist(),
        remaining["label"].tolist(),
        min(n_val, len(remaining) - 1),
        seed,
    )

    def _subset(ids, drop_dups=False):
        s = df[df["patient_id"].isin(ids)].copy()
        if drop_dups:
            s = s[s["is_duplicate"].fillna("") != "yes"]
        return s.reset_index(drop=True)

    return _subset(p_train), _subset(p_val, drop_dups=True), _subset(p_test, drop_dups=True)


def make_crosssite_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by hospital source. Returns (chughtai_df, gulab_devi_df)."""
    ch = df[df["source"] == "Chughtai"].reset_index(drop=True)
    gd = df[df["source"] == "GulabDevi"].reset_index(drop=True)
    return ch, gd


# ── sampling & loss weighting ─────────────────────────────────────────────────

def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """WeightedRandomSampler that oversamples the minority class (normal)."""
    counts  = df["label"].value_counts()
    weights = torch.tensor(
        df["label"].map(lambda l: 1.0 / counts[l]).values,
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def loss_weights(labels: torch.Tensor, device) -> torch.Tensor:
    """Per-sample loss weights: 1.0 for stones, 6.9 for normal."""
    w = torch.where(
        labels == 0,
        torch.full_like(labels, CLASS_WEIGHTS["normal"]),
        torch.ones_like(labels),
    )
    return w.to(device)
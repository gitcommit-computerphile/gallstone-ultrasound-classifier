"""
Gallstone binary classifier — training & evaluation.

Two models:
  efficientnet  — EfficientNet-B0 frozen backbone → head, then partial unfreeze
  dinov2        — DINOv2 ViT-S/14 fully frozen → linear head

Usage (Google Colab):
  # 1. Mount Drive:
  #    from google.colab import drive; drive.mount('/content/drive')
  # 2. Set DATA_ROOT to where you uploaded gallstone_dataset/:
  #    DATA_ROOT = "/content/drive/MyDrive/gallstone_dataset"
  # 3. Install deps:
  #    !pip install torch torchvision scikit-learn pandas pillow
  # 4. Upload dataset.py, train.py, evaluate.py alongside this file, then:
  #    !python train.py --model efficientnet --data-root $DATA_ROOT
  #    !python train.py --model dinov2       --data-root $DATA_ROOT

Results saved to: <data-root>/results_<model>.json
Checkpoints:      <data-root>/checkpoints/<model>_phase*.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    GallstoneDataset, load_binary_frames,
    make_patient_splits, make_crosssite_splits,
    make_sampler, get_transforms, loss_weights,
)
from evaluate import compute_metrics, find_optimal_threshold, print_results_table


# ── models ───────────────────────────────────────────────────────────────────

def build_efficientnet() -> nn.Module:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 1),
    )
    return model


def unfreeze_efficientnet_tail(model: nn.Module, n_blocks: int = 2):
    """Unfreeze the last n children of model.features for fine-tuning."""
    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


class DINOv2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(384, 1)

    def train(self, mode: bool = True):
        # keep backbone in eval mode regardless — use pretrained BN statistics
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)   # (B, 384) CLS token
        return self.head(feats)        # (B, 1)


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet,
    "dinov2":       DINOv2Classifier,
}


# ── training loop ─────────────────────────────────────────────────────────────

def _make_loader(
    df: pd.DataFrame, stage3_root: Path, augment: bool, batch_size: int, num_workers: int
) -> DataLoader:
    ds      = GallstoneDataset(df, stage3_root, get_transforms(augment))
    sampler = make_sampler(df) if augment else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        w      = loss_weights(labels, device)
        loss   = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * w).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(imgs)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_probs)


def fit(
    model: nn.Module,
    train_df,
    val_df,
    test_df,
    stage3_root: Path,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
) -> tuple[nn.Module, dict]:
    """Train model, return best-checkpoint model and test metrics."""
    train_ld = _make_loader(train_df, stage3_root, augment=True,  batch_size=batch_size, num_workers=num_workers)
    val_ld   = _make_loader(val_df,   stage3_root, augment=False, batch_size=batch_size, num_workers=num_workers)
    test_ld  = _make_loader(test_df,  stage3_root, augment=False, batch_size=batch_size, num_workers=num_workers)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc, best_state = 0.0, None

    print(f"  {'ep':>4}  {'loss':>8}  {'val_auc':>8}")
    for epoch in range(1, epochs + 1):
        tr_loss        = train_epoch(model, train_ld, optimizer, device)
        vl, vp         = predict(model, val_ld, device)
        vm             = compute_metrics(vl, vp)
        val_auc        = vm.get("auc", float("nan"))
        scheduler.step()
        print(f"  {epoch:>4}  {tr_loss:>8.4f}  {val_auc:>8.4f}")
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # find optimal threshold from val set (sensitivity >= 0.90)
    vl, vp = predict(model, val_ld, device)
    opt_thresh = find_optimal_threshold(vl, vp, sens_target=0.90)

    tl, tp = predict(model, test_ld, device)
    metrics = compute_metrics(tl, tp)
    metrics_opt = compute_metrics(tl, tp, threshold=opt_thresh)
    metrics["opt_threshold"] = round(opt_thresh, 4)
    metrics["opt_sens"]      = metrics_opt["sens"]
    metrics["opt_spec"]      = metrics_opt["spec"]
    metrics["opt_f1"]        = metrics_opt["f1"]
    metrics["opt_acc"]       = metrics_opt["acc"]
    return model, metrics


# ── cross-site helpers ────────────────────────────────────────────────────────

def _crosssite_fit(
    model_builder,
    train_site_df,
    test_site_df,
    stage3_root: Path,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
) -> dict:
    """
    Split training site 80/20 for internal val, then evaluate on the other site.
    Drops duplicates from val and test.
    """
    from sklearn.model_selection import train_test_split

    pat    = train_site_df.groupby("patient_id")["label"].agg(lambda x: x.mode()[0]).reset_index()
    n_val  = max(1, round(len(pat) * 0.2))
    try:
        p_tr, p_va = train_test_split(
            pat["patient_id"].tolist(),
            test_size=n_val,
            stratify=pat["label"].tolist(),
            random_state=42,
        )
    except ValueError:
        p_tr, p_va = train_test_split(pat["patient_id"].tolist(), test_size=n_val, random_state=42)

    def _nodups(df):
        return df[df["is_duplicate"].fillna("") != "yes"].reset_index(drop=True)

    tr_df = train_site_df[train_site_df["patient_id"].isin(p_tr)].reset_index(drop=True)
    va_df = _nodups(train_site_df[train_site_df["patient_id"].isin(p_va)])
    te_df = _nodups(test_site_df)

    print(f"  frames — train={len(tr_df)}  val={len(va_df)}  test={len(te_df)}")
    print(f"  train: {dict(tr_df['label'].value_counts())}")

    model = model_builder().to(device)
    _, metrics = fit(model, tr_df, va_df, te_df, stage3_root, device, epochs, lr, batch_size, num_workers)
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train gallstone classifier")
    parser.add_argument("--model",       choices=["efficientnet", "dinov2"], default="efficientnet")
    parser.add_argument("--data-root",   default=".", help="Path to gallstone_dataset/")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--num-workers", type=int,   default=2,
                        help="DataLoader workers (0 on Windows if issues)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root        = Path(args.data_root)
    stage3_root = root / "stage3"
    ckpt_dir    = root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    # ── data ────────────────────────────────────────────────────────────────
    binary = load_binary_frames(root / "manifest.csv", stage3_root)
    print(f"\nFrames: {len(binary)}  |  "
          f"stones={(binary['label']=='stones').sum()}  "
          f"normal={(binary['label']=='normal').sum()}")

    train_df, val_df, test_df = make_patient_splits(binary, seed=args.seed)
    print(f"Mixed split — train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name}: {dict(df['label'].value_counts())}")

    ch_df, gd_df = make_crosssite_splits(binary)
    print(f"\nChughtai frames: {len(ch_df)}  |  GulabDevi frames: {len(gd_df)}")

    model_fn   = MODEL_BUILDERS[args.model]
    fit_kwargs = dict(
        stage3_root=stage3_root,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Phase 1: frozen backbone ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Mixed split — Phase 1: frozen backbone  ({args.epochs} epochs)")
    print(f"{'='*60}")
    model = model_fn().to(device)
    model, mixed_metrics = fit(model, train_df, val_df, test_df, **fit_kwargs)
    torch.save(model.state_dict(), ckpt_dir / f"{args.model}_phase1.pt")
    print(f"  → Test AUC: {mixed_metrics['auc']:.4f}")

    # ── Phase 2: EfficientNet partial unfreeze ────────────────────────────────
    if args.model == "efficientnet":
        ft_epochs = max(10, args.epochs // 2)
        print(f"\n{'='*60}")
        print(f"Mixed split — Phase 2: unfreeze last 2 blocks  ({ft_epochs} epochs, lr={args.lr/10})")
        print(f"{'='*60}")
        unfreeze_efficientnet_tail(model, n_blocks=2)
        model, mixed_metrics = fit(
            model, train_df, val_df, test_df,
            stage3_root=stage3_root, device=device,
            epochs=ft_epochs, lr=args.lr / 10,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        torch.save(model.state_dict(), ckpt_dir / f"{args.model}_phase2.pt")
        print(f"  → Test AUC: {mixed_metrics['auc']:.4f}")

    # ── Cross-site: GulabDevi → Chughtai ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-site — train: GulabDevi  |  test: Chughtai")
    print(f"{'='*60}")
    gd_to_ch = _crosssite_fit(model_fn, gd_df, ch_df, **fit_kwargs)
    print(f"  → Test AUC: {gd_to_ch['auc']:.4f}")

    # ── Cross-site: Chughtai → GulabDevi ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-site — train: Chughtai  |  test: GulabDevi")
    print(f"{'='*60}")
    ch_to_gd = _crosssite_fit(model_fn, ch_df, gd_df, **fit_kwargs)
    print(f"  → Test AUC: {ch_to_gd['auc']:.4f}")

    # ── save & display ────────────────────────────────────────────────────────
    results = {
        "model":       args.model,
        "mixed_split": mixed_metrics,
        "gd_to_ch":    gd_to_ch,
        "ch_to_gd":    ch_to_gd,
    }
    out_path = root / f"results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")
    print_results_table(results)


if __name__ == "__main__":
    main()
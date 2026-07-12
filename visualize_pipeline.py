"""
Generate a LinkedIn-ready before/after preprocessing image.
Saves: pipeline_preview.png in the current directory.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).parent


def load(stage_root, relpath):
    rel = Path(relpath.replace("\\", "/")).relative_to("processed")
    p = stage_root / rel
    return np.array(Image.open(p).convert("RGB")) if p.exists() else None


def pick(df, source, label):
    rows = df[
        (df["source"] == source) &
        (df["label"] == label) &
        df["processed_relpath"].notna() &
        (df["processed_relpath"] != "")
    ]
    return rows.iloc[0] if len(rows) else None


def main():
    df = pd.read_csv(ROOT / "manifest.csv")

    gd_stone  = pick(df, "GulabDevi", "stones")
    ch_normal = pick(df, "Chughtai",  "normal")

    samples = [r for r in [gd_stone, ch_normal] if r is not None]
    if not samples:
        print("No candidates found.")
        return

    changes = {
        "GulabDevi":  "Removed: measurement calipers (+markers, dashed lines)\nApplied: fan mask — blacked out everything outside the ultrasound dome",
        "Chughtai":   "Removed: burned-in text strips (patient info, machine params)\nRemoved: GB label, depth scale numbers\nApplied: fan mask — blacked out everything outside the ultrasound dome",
    }

    n = len(samples)
    fig = plt.figure(figsize=(13, 5.5 * n), facecolor="#0f0f0f")

    TITLE_COLOR   = "#FFFFFF"
    LABEL_COLOR   = "#FFFFFF"
    ACCENT_BEFORE = "#FF6B6B"
    ACCENT_AFTER  = "#6BCB77"
    CAPTION_COLOR = "#AAAAAA"
    BOX_BEFORE    = dict(boxstyle="round,pad=0.4", facecolor="#FF6B6B22", edgecolor=ACCENT_BEFORE, linewidth=1.5)
    BOX_AFTER     = dict(boxstyle="round,pad=0.4", facecolor="#6BCB7722", edgecolor=ACCENT_AFTER,  linewidth=1.5)

    fig.text(
        0.5, 0.98,
        "Gallstone Ultrasound — Preprocessing Pipeline",
        ha="center", va="top", fontsize=17, fontweight="bold",
        color=TITLE_COLOR, fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5, 0.955 if n == 1 else 0.962,
        "Raw ultrasound frames contain burned-in text, measurement calipers, and machine overlays.\n"
        "Every artefact is a shortcut the model could memorise instead of learning anatomy.",
        ha="center", va="top", fontsize=10, color=CAPTION_COLOR,
        linespacing=1.6,
    )

    top_offset = 0.88 if n == 1 else 0.91
    row_height = 0.78 / n

    for i, row in enumerate(samples):
        before = load(ROOT / "processed", row["processed_relpath"])
        after  = load(ROOT / "stage3",    row["processed_relpath"])
        if before is None or after is None:
            continue

        source_label = "Gulab Devi Teaching Hospital — stone frame" \
            if row["source"] == "GulabDevi" else \
            "Chughtai Lab — normal frame"

        top  = top_offset - i * row_height
        left_b  = 0.06
        left_a  = 0.54
        w, h    = 0.38, row_height * 0.72

        ax_b = fig.add_axes([left_b, top - h, w, h])
        ax_a = fig.add_axes([left_a, top - h, w, h])

        ax_b.imshow(before)
        ax_a.imshow(after)

        for ax in (ax_b, ax_a):
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(False)

        # BEFORE / AFTER labels
        ax_b.set_title("BEFORE", fontsize=13, fontweight="bold",
                        color=ACCENT_BEFORE, pad=8)
        ax_a.set_title("AFTER",  fontsize=13, fontweight="bold",
                        color=ACCENT_AFTER,  pad=8)

        # source label centred between the two images
        fig.text(0.5, top - h * 0.05, source_label,
                 ha="center", va="bottom", fontsize=10,
                 color="#DDDDDD", fontstyle="italic")

        # arrow between panels
        arrow = FancyArrowPatch(
            posA=(left_b + w + 0.01, top - h * 0.5),
            posB=(left_a  - 0.01,    top - h * 0.5),
            transform=fig.transFigure,
            arrowstyle="-|>",
            color="#FFFFFF",
            linewidth=1.5,
            mutation_scale=14,
        )
        fig.add_artist(arrow)

        # caption box — what changed
        fig.text(
            0.5, top - h - 0.025,
            changes[row["source"]],
            ha="center", va="top", fontsize=8.5,
            color=CAPTION_COLOR, linespacing=1.7,
        )

    # footer
    fig.text(
        0.5, 0.01,
        "Dataset: 95 frames · 59 patients · Two Pakistani hospital sources (Chughtai Lab + Gulab Devi Teaching Hospital)",
        ha="center", va="bottom", fontsize=8, color="#666666",
    )

    out = ROOT / "pipeline_preview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
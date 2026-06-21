"""
Evaluation metrics for gallstone binary classification.

Label convention: stones=1 (positive/disease), normal=0 (negative).
"""

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sens_target: float = 0.90,
) -> dict:
    """
    Compute AUC-ROC plus sensitivity/specificity/F1/accuracy at threshold 0.5,
    and specificity at the threshold where sensitivity >= sens_target.
    """
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_stones = int((y_true == 1).sum())
    n_normal = int((y_true == 0).sum())

    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "sens": float("nan"), "spec": float("nan"),
            "f1": float("nan"), "acc": float("nan"),
            f"spec_at_sens{int(sens_target*100)}": float("nan"),
            "n_stones": n_stones, "n_normal": n_normal,
            "note": "single class — AUC undefined",
        }

    auc = float(roc_auc_score(y_true, y_prob))

    def _metrics_at(thresh: float) -> dict:
        pred = (y_prob >= thresh).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return {
            "sens": round(sens, 4),
            "spec": round(spec, 4),
            "f1":   round(float(f1_score(y_true, pred, zero_division=0)), 4),
            "acc":  round(float(accuracy_score(y_true, pred)), 4),
        }

    m_05 = _metrics_at(0.5)

    # find threshold where sensitivity >= sens_target
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    idx = int(np.searchsorted(tpr, sens_target))
    spec_at_target = float("nan")
    if idx < len(thresholds):
        spec_at_target = _metrics_at(float(thresholds[idx]))["spec"]

    return {
        "auc":   round(auc, 4),
        **m_05,
        f"spec_at_sens{int(sens_target*100)}": spec_at_target,
        "n_stones": n_stones,
        "n_normal": n_normal,
    }


def print_results_table(results: dict):
    """Print the CLAUDE.md reporting table across all three splits."""
    splits = ["mixed_split", "gd_to_ch", "ch_to_gd"]
    labels = ["Mixed split", "GD → CH", "CH → GD"]
    metrics = [
        ("auc",          "AUC-ROC"),
        ("sens",         "Sensitivity"),
        ("spec",         "Specificity"),
        ("spec_at_sens90", "Spec@Sens≥0.90"),
        ("f1",           "F1"),
        ("acc",          "Accuracy ⚠️"),
    ]

    col_w = 14
    print(f"\n{'Metric':<20}", end="")
    for lbl in labels:
        print(f"  {lbl:>{col_w}}", end="")
    print()
    print("-" * (20 + (col_w + 2) * len(splits)))

    for key, display in metrics:
        print(f"{display:<20}", end="")
        for split in splits:
            m = results.get(split, {})
            v = m.get(key, float("nan"))
            if isinstance(v, float) and not np.isnan(v):
                print(f"  {v:>{col_w}.4f}", end="")
            else:
                print(f"  {'—':>{col_w}}", end="")
        print()

    print()
    for split, lbl in zip(splits, labels):
        m = results.get(split, {})
        ns = m.get("n_stones", "?")
        nn = m.get("n_normal", "?")
        print(f"  {lbl}: stones={ns}  normal={nn}")
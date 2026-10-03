import json
import re

from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .config import RESULTS


def evaluate(name, y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    print(f"{name:22s} P={row['precision']:.3f} R={row['recall']:.3f} "
          f"F1={row['f1']:.3f} PR-AUC={row['pr_auc']:.3f}  (TP={tp} FP={fp} FN={fn})")
    return row


def save_result(row):
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", row["model"].lower()).strip("_")
    (RESULTS / f"{slug}.json").write_text(json.dumps(row, indent=2))

"""Collect results/*.json into one comparison table."""
import json

import pandas as pd

from .config import RESULTS


def main():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    if not rows:
        print("No results yet. Run train_classical and train_nn first.")
        return
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(RESULTS / "comparison.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved results/comparison.csv  (judge on fraud-class precision / recall / F1, not accuracy)")


if __name__ == "__main__":
    main()

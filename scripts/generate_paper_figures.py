"""Generate a bar chart (mean +/- 95% CI, per method, per metric) from
artifacts/final_results_table.csv for direct use in a paper.

Usage:
    python scripts/generate_paper_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
FIGURES_DIR = ROOT / "docs" / "figures"
INPUT_CSV = ARTIFACTS / "final_results_table.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run scripts/analyze_results.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    mean_ci_rows = df[df["row_type"] == "mean_ci"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for metric in sorted(mean_ci_rows["metric"].unique()):
        sub = mean_ci_rows[mean_ci_rows["metric"] == metric].sort_values("method")
        methods = sub["method"].tolist()
        means = sub["mean_across_seeds"].tolist()
        lower_err = [m - lo for m, lo in zip(means, sub["ci_95_low"])]
        upper_err = [hi - m for m, hi in zip(means, sub["ci_95_high"])]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(methods, means, yerr=[lower_err, upper_err], capsize=5)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — mean across seeds, 95% bootstrap CI")
        fig.tight_layout()

        out_path = FIGURES_DIR / f"{metric}_bar_chart.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

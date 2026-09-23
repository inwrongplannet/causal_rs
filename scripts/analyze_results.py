"""Analyze multi-seed experiment results: bootstrap CIs, corrected
significance tests, and effect-size labeling.

Reads artifacts/multi_seed_results.json (produced by
scripts/multi_seed_experiment.py) and produces:
  - artifacts/final_results_table.csv
  - a printed summary to stdout

Usage:
    python scripts/analyze_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import bootstrap_ci, cohens_d_label

ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "multi_seed_results.json"
OUTPUT_CSV = ARTIFACTS / "final_results_table.csv"

METRICS = ["ndcg_mean", "precision_mean", "ild_mean"]
ALPHA = 0.05


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run scripts/multi_seed_experiment.py first."
        )
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def per_seed_matrix(df, method, metric):
    """Return an array of one value per seed for (method, metric), sorted by seed."""
    sub = df[df["method"] == method].sort_values("seed")
    return sub[metric].to_numpy()


def main():
    df = load_results()
    methods_present = sorted(df["method"].unique())
    seeds = sorted(df["seed"].unique())
    print(f"Loaded results for {len(seeds)} seeds: {seeds}")
    print(f"Methods present: {methods_present}")

    summary_rows = []
    raw_pvalues = []
    comparison_labels = []

    baseline_methods = [m for m in methods_present if m != "PPO"]

    for metric in METRICS:
        ppo_vals = per_seed_matrix(df, "PPO", metric)

        for method_name in methods_present:
            vals = per_seed_matrix(df, method_name, metric)
            ci = bootstrap_ci(vals.tolist(), n_boot=10000, seed=42)
            summary_rows.append({
                "metric": metric,
                "row_type": "mean_ci",
                "method": method_name,
                "mean_across_seeds": ci["mean"],
                "ci_95_low": ci["ci_low"],
                "ci_95_high": ci["ci_high"],
                "n_seeds": len(vals),
            })

        for baseline_name in baseline_methods:
            baseline_vals = per_seed_matrix(df, baseline_name, metric)
            diffs = ppo_vals - baseline_vals
            t_stat, p_val = stats.ttest_rel(ppo_vals, baseline_vals)
            d = float(np.mean(diffs) / (np.std(ppo_vals, ddof=1) + 1e-10))
            raw_pvalues.append(p_val)
            comparison_labels.append(f"PPO vs {baseline_name} ({metric})")
            summary_rows.append({
                "metric": metric,
                "row_type": "comparison",
                "method": f"PPO vs {baseline_name}",
                "t_stat": float(t_stat),
                "p_raw": float(p_val),
                "cohens_d": d,
                "effect_label": cohens_d_label(d),
            })

    reject, p_corrected, _, _ = multipletests(raw_pvalues, alpha=ALPHA, method="holm")

    print("\n" + "=" * 70)
    print(f"CORRECTED SIGNIFICANCE RESULTS (Holm-Bonferroni, alpha={ALPHA:.2f})")
    print("=" * 70)
    idx = 0
    for row in summary_rows:
        if row.get("row_type") == "comparison":
            row["p_adj"] = float(p_corrected[idx])
            row["significant_after_correction"] = bool(reject[idx])
            print(f"{comparison_labels[idx]:40s} p_raw={raw_pvalues[idx]:.4f}  "
                  f"p_adj={p_corrected[idx]:.4f}  d={row['cohens_d']:.3f} "
                  f"({row['effect_label']})  "
                  f"{'SIGNIFICANT' if reject[idx] else 'not significant'} (corrected)")
            idx += 1

    result_df = pd.DataFrame(summary_rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull table written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

# Pre-Registration: Confirmatory Evaluation Configuration

This document fixes the exact configuration used for the **one** confirmatory
result reported in the paper/README, decided **before** looking at its
results. Everything in `docs/RESEARCH_LOG.md` before this document's
creation date is **exploratory** work (tuning, debugging, ablations) and
must not be cited as the paper's evidence — only the run described here can
be.

## Fixed Configuration

| Parameter | Value |
|---|---|
| Click weight `w` | 0.3 |
| Candidates per step `K` | 10 |
| Episode length `T` | 1 |
| Total training timesteps | 200,000 |
| Number of independent training seeds | 5 (`seeds = [1, 2, 3, 4, 5]`) |
| GCM mechanism | `fit_gcm_item_sensitive()` (HistGradientBoostingRegressor), per `docs/RESEARCH_LOG.md` 2026-06-17 entry |
| CDI normalization | Min-max normalized within each step's candidate pool (`NewsRecommendEnv._min_max_cdi()`) |
| Reported metrics | NDCG@10, Precision@10, ILD |
| Baselines compared | Random, Popularity, Logistic-CF (Phase 5) |
| Significance threshold (α) | 0.05 |
| Multiple-comparison correction | Holm-Bonferroni, applied across all comparisons in the same confirmatory table |
| Effect size | Cohen's d, computed across seed-level means (not across test-session-level scores) |
| Confidence intervals | Bootstrap, 95%, 10,000 resamples, computed across seed-level means |

## What Counts as Confirmatory

Only the output of `scripts/multi_seed_experiment.py` (Phase 4 of the
research-ready plan) run with the exact configuration above, analyzed by
`scripts/analyze_results.py`, may be reported as the paper's result. Any
number from `docs/RESEARCH_LOG.md`, `scripts/eval_only.py`, or any other ad
hoc run is exploratory and must be labeled as such if shown in the paper
(e.g. in an ablation/appendix section).

## Decision Rule (fixed in advance, not to be changed after seeing results)

The PPO-vs-Random comparison on NDCG is treated as a genuine positive result
**if and only if**:
1. `p_adj < 0.05` (Holm-Bonferroni corrected) for the PPO-vs-Random NDCG comparison, **and**
2. The corresponding effect size label is not `"negligible"` (i.e. `|d| >= 0.2`).

If both conditions hold: report a positive result (README/paper "Path A").
If either condition fails: report a diagnostic/negative result explaining
why the causal reward signal did not translate into a measurable
improvement (README/paper "Path B"). See Phase 6 of the implementation
plan for the exact template text for each path.

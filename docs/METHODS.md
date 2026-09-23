# Methods

This document mirrors `docs/PREREGISTRATION.md` in prose form, suitable for
direct inclusion in a paper's Methods section.

## Data
Microsoft News (MIND) dataset (MIND-small and/or MIND-large variants).
Impressions are split 70/15/15 into train/val/test by impression ID, seeded
deterministically (see `src/data_pipeline/scm_builder.py`).

## Causal Model
A structural causal model is fit over user PCA features (32 dims), item
entity-embedding PCA features (32 dims), item title-embedding PCA features
(32 dims), category, sentiment, and dwell time, with treatment `A`
(exposure) and outcome `Y_diversity`. The average treatment effect (ATE) is
estimated via inverse propensity weighting and linear regression
(`src/causal_model/model.py`), and validated with placebo, data-subset, and
random-common-cause refutation tests (`src/causal_model/refutation.py`).

## Counterfactual Diversity Impact (CDI)
A DoWhy `StructuralCausalModel` (GCM) is fit over the same feature set,
using `HistGradientBoostingRegressor` mechanisms for `Y_diversity` and
`Y_click` (`fit_gcm_item_sensitive()`), chosen because linear mechanisms
were dominated by the 32 user-PCA features and produced near-constant CDI
scores (see `docs/RESEARCH_LOG.md`, 2026-06-17 entry). CDI is computed as
`E[Y_diversity | do(A=1, item features)]` and normalized via min-max
scaling within each step's candidate pool (`NewsRecommendEnv._min_max_cdi`).

## RL Agent
A Gymnasium environment (`NewsRecommendEnv`) with observation = user history
embedding (EMA-updated on click) + current diversity score, action = choice
among `K` candidate articles, reward = `w * click + (1 - w) * CDI`. Trained
with PPO (stable-baselines3) using the hyperparameters fixed in
`docs/PREREGISTRATION.md`.

## Evaluation Protocol
Offline replay evaluation on held-out test sessions
(`src/evaluation/metrics.py::replay_evaluate`), computing NDCG@10,
Precision@10, and Intra-List Diversity (ILD) against ground-truth clicks.
Baselines: Random (seeded), Popularity (train-set frequency), and
Logistic-CF (`src/baselines/logistic_cf.py`, a click-probability model
trained on the same feature set).

## Statistical Procedure
PPO and the Logistic-CF baseline are each evaluated across 5 independent
seeds (Random and Popularity are deterministic given a seeded RNG). For
each metric, a 95% bootstrap confidence interval (10,000 resamples) is
computed across the 5 per-seed means (`src/evaluation/metrics.py::bootstrap_ci`).
Paired t-tests compare PPO against each baseline at the seed level, with
Holm-Bonferroni correction applied across all comparisons reported in the
same table (`scripts/analyze_results.py`). Effect sizes are reported as
Cohen's d with standard-convention labels (negligible/small/medium/large;
`src/evaluation/metrics.py::cohens_d_label`).

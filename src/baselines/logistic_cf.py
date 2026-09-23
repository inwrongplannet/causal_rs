"""Logistic regression click-prediction baseline (a minimal, correlational
'CF-style' baseline) for comparison against the causal-RL PPO agent.

This is the baseline the original project plan called for ("<5% degradation
vs a CF baseline") but was never implemented before this module — only
Random and Popularity baselines existed previously.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_PREFIXES = ("U_pca_", "I_entity_pca_", "I_title_pca_")
EXTRA_NUMERIC_FEATURES = ("U_dwell_mean", "I_sentiment")
CATEGORICAL_FEATURES = ("I_category",)


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Find all PCA + numeric feature columns present in the dataframe."""
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    cols += [c for c in EXTRA_NUMERIC_FEATURES if c in df.columns]
    return cols


def build_feature_matrix(df: pd.DataFrame, fit_columns: list[str] | None = None):
    """Build a numeric feature matrix (PCA dims + numeric + one-hot category).

    Args:
        df: SCM dataframe with PCA/numeric/categorical columns.
        fit_columns: If provided (from a previously-built training matrix),
            the returned matrix's columns are reindexed to match exactly,
            filling any missing one-hot category with 0. Pass this when
            transforming test data so train/test have identical columns.

    Returns:
        Tuple of (numpy feature matrix, list of column names used).
    """
    numeric_cols = _select_feature_columns(df)
    numeric_part = df[numeric_cols].fillna(0.0)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if cat_cols:
        cat_part = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
    else:
        cat_part = pd.DataFrame(index=df.index)

    full = pd.concat([numeric_part, cat_part], axis=1)

    if fit_columns is not None:
        full = full.reindex(columns=fit_columns, fill_value=0.0)
        columns = fit_columns
    else:
        columns = list(full.columns)

    return full.to_numpy(dtype=np.float64), columns


def train_logistic_cf(train_df: pd.DataFrame, seed: int = 42) -> dict:
    """Train a logistic regression click-prediction model.

    Args:
        train_df: SCM training dataframe with a "Y_click" column.
        seed: Random seed for the LogisticRegression solver.

    Returns:
        Dict with keys "model" (fitted LogisticRegression) and "columns"
        (the exact feature column order used — required for scoring test
        data with a matching feature matrix shape).
    """
    X, columns = build_feature_matrix(train_df)
    y = train_df["Y_click"].to_numpy()
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X, y)
    return {"model": model, "columns": columns}


def score_candidates(fitted: dict, candidates_df: pd.DataFrame) -> np.ndarray:
    """Score a set of candidate rows with the fitted model.

    Args:
        fitted: Output of train_logistic_cf().
        candidates_df: Dataframe of candidate rows with the same feature
            columns as the training data (an extra Y_click column, if
            present, is ignored).

    Returns:
        1-D array of predicted click probabilities, one per row, in the
        same row order as candidates_df.
    """
    X, _ = build_feature_matrix(candidates_df, fit_columns=fitted["columns"])
    return fitted["model"].predict_proba(X)[:, 1]

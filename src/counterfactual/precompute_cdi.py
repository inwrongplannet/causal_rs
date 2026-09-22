import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from dowhy import gcm
from dowhy.gcm.fitting_sampling import PARENTS_DURING_FIT

from src.counterfactual.queries import category_to_int

log = logging.getLogger(__name__)


def precompute_cdi_cache(
    scm: gcm.StructuralCausalModel,
    sessions: list,
    news_df: pd.DataFrame,
    df_train: pd.DataFrame,
    categories: list = None,
    cache_path: Path = None,
    n_draws: int = 200,
) -> dict:
    """Precompute and cache CDI scores for all (user, candidate) pairs.

    Iterates over training sessions and candidate pools, and computes CDI
    for all items in a session in a single batched GCM evaluation call.
    Results are stored in a dict keyed by (user_id, item_id). Optionally
    persists to pickle.

    Args:
        scm: Fitted StructuralCausalModel.
        sessions: List of session objects with .user_id and .candidate_pool.
        news_df: News DataFrame indexed by item_id with I_category,
                 I_sentiment columns.
        df_train: Training DataFrame indexed by user_id.
        categories: Optional list of known categories for encoding.
        cache_path: Optional Path to save the cache as pickle.
        n_draws: Number of noise draws for each counterfactual query.
                 Default 200.

    Returns:
        Dict mapping (user_id, item_id) -> CDI score.
    """
    parent_order = scm.graph.nodes["Y_diversity"].get(
        PARENTS_DURING_FIT,
        sorted(scm.graph.predecessors("Y_diversity")),
    )
    mech = scm.causal_mechanism("Y_diversity")

    cdi_cache = {}
    for session in tqdm(sessions, desc="Precomputing CDI"):
        try:
            user_row = df_train[df_train["user_id"] == session.user_id].iloc[0]
        except IndexError:
            log.warning("User %s not found in training data, skipping", session.user_id)
            continue

        unique_items = list(set(session.candidate_pool))
        parent_rows = []
        valid_item_ids = []

        for item_id in unique_items:
            try:
                item = news_df.loc[item_id]
            except KeyError:
                continue
            parent_row = pd.DataFrame([user_row], columns=parent_order)
            parent_row["A"] = 1
            parent_row["I_category"] = item["I_category"]
            parent_row["I_sentiment"] = item["I_sentiment"]
            for col in parent_row.columns:
                if col.startswith("I_entity_pca_") or col.startswith("I_title_pca_"):
                    if col in item.index:
                        parent_row[col] = item[col]
            parent_rows.append(parent_row.values[0])
            valid_item_ids.append(item_id)

        if not valid_item_ids:
            continue

        parent_values = np.stack(parent_rows, axis=0)
        n_items = len(valid_item_ids)
        tiled = np.repeat(parent_values, n_draws, axis=0)
        noise = mech.draw_noise_samples(num_samples=n_items * n_draws)
        evals = mech.evaluate(tiled, noise)
        cdi_values = evals.reshape(n_items, n_draws).mean(axis=1)

        for item_id, cdi in zip(valid_item_ids, cdi_values):
            cdi_cache[(session.user_id, item_id)] = float(cdi)

    log.info("Cached %d CDI values", len(cdi_cache))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cdi_cache, f)
        log.info("CDI cache saved to %s", cache_path)

    return cdi_cache


def load_cdi_cache(cache_path: Path) -> dict:
    """Load a previously saved CDI cache from pickle.

    Args:
        cache_path: Path to the pickle file.

    Returns:
        Dict mapping (user_id, item_id) -> CDI score.
    """
    with open(cache_path, "rb") as f:
        return pickle.load(f)

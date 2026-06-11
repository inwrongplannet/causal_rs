import pickle
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dowhy import gcm

from src.counterfactual.queries import predict_diversity_counterfactual, category_to_int

log = logging.getLogger(__name__)


def precompute_cdi_cache(
    scm: gcm.StructuralCausalModel,
    sessions: list,
    news_df: pd.DataFrame,
    df_train: pd.DataFrame,
    categories: list = None,
    cache_path: Path = None,
) -> dict:
    """Precompute and cache CDI scores for all (user, candidate) pairs.

    Iterates over training sessions and candidate pools, queries the GCM
    for each pair's counterfactual diversity score, and stores results in
    a dict keyed by (user_id, item_id). Optionally persists to pickle.

    Args:
        scm: Fitted StructuralCausalModel.
        sessions: List of session objects with .user_id and .candidate_pool.
        news_df: News DataFrame indexed by item_id with I_category,
                 I_sentiment columns.
        df_train: Training DataFrame indexed by user_id.
        categories: Optional list of known categories for encoding.
        cache_path: Optional Path to save the cache as pickle.

    Returns:
        Dict mapping (user_id, item_id) -> CDI score.
    """
    cdi_cache = {}
    for session in tqdm(sessions, desc="Precomputing CDI"):
        try:
            user_row = df_train[df_train["user_id"] == session.user_id].iloc[0]
        except IndexError:
            log.warning("User %s not found in training data, skipping", session.user_id)
            continue
        # Discover item PCA columns in the news lookup
        entity_pca_cols = [c for c in news_df.columns if c.startswith("I_entity_pca_")]
        title_pca_cols = [c for c in news_df.columns if c.startswith("I_title_pca_")]

        for item_id in session.candidate_pool:
            try:
                item = news_df.loc[item_id]
                item_entity_pca = None
                item_title_pca = None
                if entity_pca_cols:
                    item_entity_pca = {c: item[c] for c in entity_pca_cols}
                if title_pca_cols:
                    item_title_pca = {c: item[c] for c in title_pca_cols}
                cdi = predict_diversity_counterfactual(
                    scm, user_row, item["I_category"], item["I_sentiment"],
                    new_item_entity_pca=item_entity_pca,
                    new_item_title_pca=item_title_pca,
                    categories=categories,
                )
                cdi_cache[(session.user_id, item_id)] = cdi
            except Exception as e:
                log.warning("CDI failed for (%s, %s): %s", session.user_id, item_id, e)
                continue

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

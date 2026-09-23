from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.data_pipeline.nlp_utils import (
    mean_embeddings,
    score_sentiment,
    stable_hash_vector,
)
from src.data_pipeline.parsers import parse_entities


def compute_news_features(
    news_df: pd.DataFrame,
    title_encoder: Callable[[list[str]], np.ndarray],
    sentiment_analyzer,
    entity_dim: int,
    pretrained_entity_lookup: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    titles = news_df["Title"].astype(str).tolist()
    title_embeddings = title_encoder(titles)
    if title_embeddings.shape[0] != news_df.shape[0]:
        raise ValueError("Title embedding count mismatch with news rows.")

    entity_embeddings: list[list[float]] = []
    for entity_blob in news_df["TitleEntities"].tolist():
        entity_ids = parse_entities(entity_blob)
        if entity_ids:
            if pretrained_entity_lookup:
                vectors = []
                for eid in entity_ids:
                    vec = pretrained_entity_lookup.get(eid)
                    if vec is not None:
                        vectors.append(vec)
                    else:
                        vectors.append(stable_hash_vector(eid, entity_dim))
            else:
                vectors = [stable_hash_vector(entity_id, entity_dim) for entity_id in entity_ids]
            entity_vector = mean_embeddings(vectors, entity_dim)
        else:
            entity_vector = np.zeros(entity_dim, dtype=np.float32)
        entity_embeddings.append(entity_vector.tolist())

    sentiments = [score_sentiment(title, sentiment_analyzer) for title in titles]

    features_df = pd.DataFrame({
        "item_id": news_df["NewsID"].astype(str),
        "I_category": news_df["Category"].replace("", "unknown").astype(str),
        "I_subcategory": news_df["SubCategory"].replace("", "unknown").astype(str),
        "I_sentiment": sentiments,
    })
    features_df["I_title_emb_full"] = [row.astype(np.float32).tolist() for row in title_embeddings]
    features_df["I_entity_emb_full"] = entity_embeddings
    features_df = features_df.drop_duplicates(subset=["item_id"], keep="first")
    return features_df.set_index("item_id", drop=True)


def prepare_behaviors(behaviors_df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    from src.data_pipeline.parsers import parse_history, parse_impressions

    prepared = behaviors_df.copy()
    if max_rows is not None:
        prepared = prepared.head(int(max_rows)).copy()
    prepared["history_ids"] = prepared["History"].apply(parse_history)
    prepared["parsed_impressions"] = prepared["Impressions"].apply(parse_impressions)
    prepared["impression_count"] = prepared["parsed_impressions"].apply(len)
    prepared = prepared[prepared["impression_count"] > 0].copy()
    return prepared.reset_index(drop=True)


def build_session_user_features(
    behaviors_df: pd.DataFrame,
    news_features_df: pd.DataFrame,
    title_dim: int,
) -> pd.DataFrame:
    title_lookup = news_features_df["I_title_emb_full"].to_dict()
    records = []
    for row in behaviors_df.itertuples(index=False):
        history_vectors = [
            np.asarray(title_lookup[item_id], dtype=np.float32)
            for item_id in row.history_ids
            if item_id in title_lookup
        ]
        u_history_emb = mean_embeddings(history_vectors, title_dim)
        records.append({
            "impression_id": str(row.ImpressionID),
            "user_id": str(row.UserID),
            "U_history_emb_full": u_history_emb.tolist(),
            "U_dwell_mean": float(row.impression_count),
            "U_click_count": float(len(row.history_ids)),
        })
    return pd.DataFrame(records)


def sample_negative_items(
    item_pool: np.ndarray, shown_item_ids: Sequence[str], count: int, rng
) -> list[str]:
    if count <= 0:
        return []
    shown_set = set(shown_item_ids)
    candidate_pool = [item_id for item_id in item_pool if item_id not in shown_set]
    if not candidate_pool:
        return []
    replace = count > len(candidate_pool)
    sampled = rng.choice(candidate_pool, size=count, replace=replace)
    return [str(item_id) for item_id in sampled.tolist()]

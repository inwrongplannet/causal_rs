import logging
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    DATASET,
    NEG_RATIO,
    PCA_COMPONENTS,
    SEED,
    SPLIT_RATIOS,
    TITLE_EMBED_DIM,
)

logger = logging.getLogger(__name__)


def _try_import_pca():
    """Return a GPU-accelerated PCA if cuML is available, else sklearn."""
    try:
        from cuml import PCA as GpuPCA
        logger.info("Using cuML PCA (GPU-accelerated)")
        return GpuPCA, None
    except ImportError:
        from sklearn.decomposition import PCA, IncrementalPCA
        logger.info("cuML not available, using sklearn PCA (CPU)")
        return PCA, IncrementalPCA


PCA, IncrementalPCA = _try_import_pca()


def build_scm_dataframe(
    behaviors_df: pd.DataFrame,
    news_features_df: pd.DataFrame,
    session_user_features_df: pd.DataFrame,
    neg_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    news_lookup = news_features_df.to_dict("index")
    session_lookup = session_user_features_df.set_index("impression_id").to_dict("index")
    item_pool = np.array(news_features_df.index.tolist())

    from src.data_pipeline.features import sample_negative_items
    from src.gpu_utils import batch_cosine_diversity

    # Two-pass approach:
    #   1) Build records + collect embedding pairs for batched GPU cosine diversity.
    #   2) Compute all diversity scores in one shot, then assign back.
    records: list[dict] = []
    user_embs: list[np.ndarray] = []
    item_embs: list[np.ndarray] = []

    for row in behaviors_df.itertuples(index=False):
        impression_id = str(row.ImpressionID)
        if impression_id not in session_lookup:
            continue
        session_features = session_lookup[impression_id]
        u_history_emb = np.asarray(session_features["U_history_emb_full"], dtype=np.float32)

        shown_items: list[str] = []
        for item_id, clicked in row.parsed_impressions:
            item_features = news_lookup.get(item_id)
            if item_features is None:
                continue
            shown_items.append(item_id)
            i_title_emb = np.asarray(item_features["I_title_emb_full"], dtype=np.float32)
            user_embs.append(u_history_emb)
            item_embs.append(i_title_emb)
            records.append({
                "user_id": str(row.UserID),
                "impression_id": impression_id,
                "time": str(row.Time),
                "item_id": str(item_id),
                "A": 1,
                "Y_click": int(clicked),
                "U_dwell_mean": float(session_features["U_dwell_mean"]),
                "U_click_count": float(session_features["U_click_count"]),
                "I_category": str(item_features["I_category"]),
                "I_subcategory": str(item_features["I_subcategory"]),
                "I_sentiment": float(item_features["I_sentiment"]),
                "U_history_emb_full": u_history_emb.tolist(),
                "I_title_emb_full": i_title_emb.tolist(),
                "I_entity_emb_full": list(item_features["I_entity_emb_full"]),
            })

        negative_count = len(shown_items) * int(neg_ratio)
        sampled_negatives = sample_negative_items(item_pool, shown_items, negative_count, rng)
        for item_id in sampled_negatives:
            item_features = news_lookup.get(item_id)
            if item_features is None:
                continue
            i_title_emb = np.asarray(item_features["I_title_emb_full"], dtype=np.float32)
            user_embs.append(u_history_emb)
            item_embs.append(i_title_emb)
            records.append({
                "user_id": str(row.UserID),
                "impression_id": impression_id,
                "time": str(row.Time),
                "item_id": str(item_id),
                "A": 0,
                "Y_click": 0,
                "U_dwell_mean": float(session_features["U_dwell_mean"]),
                "U_click_count": float(session_features["U_click_count"]),
                "I_category": str(item_features["I_category"]),
                "I_subcategory": str(item_features["I_subcategory"]),
                "I_sentiment": float(item_features["I_sentiment"]),
                "U_history_emb_full": u_history_emb.tolist(),
                "I_title_emb_full": i_title_emb.tolist(),
                "I_entity_emb_full": list(item_features["I_entity_emb_full"]),
            })

    if not records:
        raise ValueError("SCM dataframe is empty. Check MIND-small extraction and parsing logic.")

    # Batched diversity computation (GPU via cupy if available, else CPU).
    if user_embs and item_embs:
        user_matrix = np.vstack(user_embs).astype(np.float32)
        item_matrix = np.vstack(item_embs).astype(np.float32)
        diversities = batch_cosine_diversity(user_matrix, item_matrix)
        for rec, div in zip(records, diversities):
            rec["Y_diversity"] = float(div)
    else:
        for rec in records:
            rec["Y_diversity"] = 0.0

    scm_df = pd.DataFrame.from_records(records)
    scm_df.insert(0, "row_id", np.arange(scm_df.shape[0], dtype=np.int64))
    scm_df["A"] = scm_df["A"].astype(np.int8)
    scm_df["Y_click"] = scm_df["Y_click"].astype(np.int8)
    scm_df["Y_diversity"] = scm_df["Y_diversity"].astype(np.float32)
    scm_df["U_dwell_mean"] = scm_df["U_dwell_mean"].astype(np.float32)
    scm_df["U_click_count"] = scm_df["U_click_count"].astype(np.float32)
    scm_df["I_sentiment"] = scm_df["I_sentiment"].astype(np.float32)
    return scm_df


def reduce_embedding_columns(
    scm_df: pd.DataFrame, n_components: int, seed: int, chunk_size: int = 100000
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    reduced_df = scm_df.copy()
    pca_report: dict[str, dict[str, float]] = {}
    specs = [
        ("U_history_emb_full", "U_pca"),
        ("I_entity_emb_full", "I_entity_pca"),
        ("I_title_emb_full", "I_title_pca"),
    ]

    for source_col, prefix in specs:
        total = len(reduced_df)
        dim = int(max(np.asarray(reduced_df[source_col].iloc[0], dtype=np.float32).size, 1))
        components = int(min(n_components, total, dim))

        if components <= 0:
            raise ValueError(f"Cannot run PCA on column {source_col}; rows={total}, dim={dim}.")

        if total * dim * 4 < 512 * 1024 * 1024 and IncrementalPCA is not None:
            # Small enough for in-memory PCA
            matrix_list = [
                np.asarray(row, dtype=np.float32)
                for row in reduced_df[source_col].to_numpy()
            ]
            matrix = np.vstack(matrix_list)
            pca = PCA(n_components=components, random_state=seed)
            transformed = pca.fit_transform(matrix).astype(np.float32)
        elif IncrementalPCA is not None:
            # Large dataset — use IncrementalPCA in chunks
            logger.info("Using IncrementalPCA for %s (%d rows x %d dims)", source_col, total, dim)
            ipca = IncrementalPCA(n_components=components, batch_size=chunk_size)
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_rows = [
                    np.asarray(row, dtype=np.float32)
                    for row in reduced_df[source_col].iloc[start:end].to_numpy()
                ]
                chunk_matrix = np.vstack(chunk_rows)
                ipca.partial_fit(chunk_matrix)
                logger.info("  IncrementalPCA partial_fit %d / %d", end, total)
            # Transform in chunks
            transformed_list = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_rows = [
                    np.asarray(row, dtype=np.float32)
                    for row in reduced_df[source_col].iloc[start:end].to_numpy()
                ]
                chunk_matrix = np.vstack(chunk_rows)
                transformed_list.append(ipca.transform(chunk_matrix).astype(np.float32))
            transformed = np.vstack(transformed_list)
            pca = ipca
        else:
            # No IncrementalPCA available — try in-memory with warning
            logger.warning("IncrementalPCA unavailable, attempting in-memory PCA for %s", source_col)
            matrix_list = [
                np.asarray(row, dtype=np.float32)
                for row in reduced_df[source_col].to_numpy()
            ]
            matrix = np.vstack(matrix_list)
            pca = PCA(n_components=components, random_state=seed)
            transformed = pca.fit_transform(matrix).astype(np.float32)

        for idx in range(components):
            reduced_df[f"{prefix}_{idx}"] = transformed[:, idx]
        for idx in range(components, n_components):
            reduced_df[f"{prefix}_{idx}"] = 0.0

        explained_var = getattr(pca, "explained_variance_ratio_", None)
        if explained_var is not None:
            pca_report[prefix] = {
                "n_components": components,
                "explained_variance_ratio_sum": float(np.sum(explained_var)),
            }
        else:
            pca_report[prefix] = {
                "n_components": components,
                "explained_variance_ratio_sum": None,
            }
    return reduced_df, pca_report


def split_by_impression_id(
    scm_df: pd.DataFrame,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    keys = scm_df["impression_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_total = len(keys)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    train_df = scm_df[scm_df["impression_id"].isin(train_keys)].copy()
    val_df = scm_df[scm_df["impression_id"].isin(val_keys)].copy()
    test_df = scm_df[scm_df["impression_id"].isin(test_keys)].copy()

    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("Leakage detected: overlapping ImpressionID keys across splits.")

    split_report = {
        "total_impressions": n_total,
        "train_impressions": len(train_keys),
        "val_impressions": len(val_keys),
        "test_impressions": len(test_keys),
    }
    return train_df, val_df, test_df, split_report


def run_quality_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    required_cols: Sequence[str],
    neg_ratio: int,
) -> dict[str, object]:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    missing_columns = [col for col in required_cols if col not in combined.columns]
    critical_cols = [
        "user_id", "item_id", "impression_id", "A", "Y_click",
        "Y_diversity", "U_dwell_mean", "I_category", "I_sentiment",
    ]
    null_summary = {col: int(combined[col].isna().sum()) for col in critical_cols if col in combined.columns}
    y_click_values = sorted(set(combined["Y_click"].dropna().astype(int).tolist())) if "Y_click" in combined.columns else []
    y_click_binary = set(y_click_values).issubset({0, 1})
    y_div_min = float(combined["Y_diversity"].min()) if "Y_diversity" in combined.columns else None
    y_div_max = float(combined["Y_diversity"].max()) if "Y_diversity" in combined.columns else None
    y_div_complete = bool(combined["Y_diversity"].notna().all()) if "Y_diversity" in combined.columns else False
    treatment_count = int((combined["A"] == 1).sum()) if "A" in combined.columns else 0
    control_count = int((combined["A"] == 0).sum()) if "A" in combined.columns else 0
    observed_ratio = float(control_count / treatment_count) if treatment_count > 0 else None
    ratio_ok = (
        observed_ratio is not None and abs(observed_ratio - neg_ratio) <= (0.35 * neg_ratio)
    )
    train_keys = set(train_df["impression_id"].astype(str).unique())
    val_keys = set(val_df["impression_id"].astype(str).unique())
    test_keys = set(test_df["impression_id"].astype(str).unique())
    leakage_ok = not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)

    report = {
        "missing_columns": missing_columns,
        "null_summary": null_summary,
        "y_click_values": y_click_values,
        "y_click_binary": y_click_binary,
        "y_diversity_min": y_div_min,
        "y_diversity_max": y_div_max,
        "y_diversity_complete": y_div_complete,
        "treatment_count": treatment_count,
        "control_count": control_count,
        "observed_control_per_treatment": observed_ratio,
        "target_control_per_treatment": float(neg_ratio),
        "ratio_check_pass": ratio_ok,
        "split_leakage_check_pass": leakage_ok,
    }

    if missing_columns:
        raise AssertionError(f"Missing required columns: {missing_columns}")
    if sum(null_summary.values()) > 0:
        raise AssertionError(f"Critical nulls found: {null_summary}")
    if not y_click_binary:
        raise AssertionError(f"Y_click is not binary. Values={y_click_values}")
    if not y_div_complete:
        raise AssertionError("Y_diversity contains null values.")
    if y_div_min is None or y_div_max is None or y_div_min < 0.0 or y_div_max > 1.0:
        raise AssertionError(f"Y_diversity out of expected range [0, 1]: min={y_div_min}, max={y_div_max}")
    if not leakage_ok:
        raise AssertionError("Split leakage detected by ImpressionID.")

    return report


def build_phase1_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_report: dict[str, int],
    quality_report: dict[str, object],
    pca_report: dict[str, dict[str, float]],
    encoder_meta: dict[str, str],
    output_paths: dict[str, str],
) -> dict[str, object]:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": DATASET,
            "seed": SEED,
            "neg_ratio": NEG_RATIO,
            "pca_components": PCA_COMPONENTS,
            "title_embedding_dim": TITLE_EMBED_DIM,
            "split_ratios": list(SPLIT_RATIOS),
            "split_key": "impression_id",
        },
        "encoder": encoder_meta,
        "rows": {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
            "total": int(all_df.shape[0]),
        },
        "class_balance": {
            "A_counts": {str(k): int(v) for k, v in all_df["A"].value_counts(dropna=False).to_dict().items()},
            "Y_click_counts": {str(k): int(v) for k, v in all_df["Y_click"].value_counts(dropna=False).to_dict().items()},
        },
        "null_summary": {col: int(all_df[col].isna().sum()) for col in ["A", "Y_click", "Y_diversity", "U_dwell_mean", "I_sentiment"]},
        "split_summary": split_report,
        "pca": pca_report,
        "quality_checks": quality_report,
        "outputs": output_paths,
    }

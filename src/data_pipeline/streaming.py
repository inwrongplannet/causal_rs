import gc
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from src.config import MAX_BEHAVIOR_ROWS, SPLIT_RATIOS, TITLE_EMBED_DIM
from src.data_pipeline.features import build_session_user_features, prepare_behaviors
from src.data_pipeline.scm_builder import build_scm_dataframe


def _infer_split_source(path: Path) -> str:
    marker = path.parent.as_posix().lower()
    if "train" in marker:
        return "train"
    if "dev" in marker or "valid" in marker or "val" in marker:
        return "dev"
    if "test" in marker:
        return "test"
    return "unknown"


def hash_split(session_key: str, ratios: Tuple[float, float, float]) -> str:
    import hashlib
    val = int(hashlib.md5(session_key.encode("utf-8")).hexdigest(), 16) % 100
    train_thresh = int(ratios[0] * 100)
    val_thresh = train_thresh + int(ratios[1] * 100)
    if val < train_thresh:
        return "train"
    elif val < val_thresh:
        return "val"
    return "test"


def process_behavior_chunk(
    behaviors_df: pd.DataFrame,
    news_features_df: pd.DataFrame,
    neg_ratio: int,
    seed: int,
    split_source: str,
    global_row_limit: int = MAX_BEHAVIOR_ROWS,
    rows_consumed_so_far: int = 0,
) -> pd.DataFrame:
    rows_remaining = global_row_limit - rows_consumed_so_far
    if rows_remaining <= 0:
        return pd.DataFrame()
    chunk_max = min(len(behaviors_df), rows_remaining)
    behaviors_df = behaviors_df.head(chunk_max)
    behaviors_df = behaviors_df.copy()
    behaviors_df["SplitSource"] = split_source
    behaviors_df["SessionKey"] = behaviors_df["SplitSource"] + ":" + behaviors_df["ImpressionID"]
    behaviors_df = prepare_behaviors(behaviors_df, max_rows=None)
    if behaviors_df.empty:
        return pd.DataFrame()

    session_user_features_df = build_session_user_features(
        behaviors_df=behaviors_df,
        news_features_df=news_features_df,
        title_dim=TITLE_EMBED_DIM,
    )
    scm_df = build_scm_dataframe(
        behaviors_df=behaviors_df,
        news_features_df=news_features_df,
        session_user_features_df=session_user_features_df,
        neg_ratio=neg_ratio,
        seed=seed,
    )
    return scm_df


def stream_and_build(
    behavior_paths: Sequence[Path],
    news_features_df: pd.DataFrame,
    output_dir: Path,
    chunksize: int = 2000,
    neg_ratio: int = 4,
    seed: int = 42,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]

    from src.data_pipeline.io_utils import save_parquet

    part_idx = 0
    stats: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "total_rows": 0}

    rows_consumed = 0
    for path in behavior_paths:
        split_source = _infer_split_source(Path(path))
        reader = pd.read_csv(
            path, sep="\t", names=columns, header=None,
            dtype=str, keep_default_na=False, na_filter=False,
            encoding="utf-8", chunksize=chunksize,
        )
        for beh_chunk in reader:
            if rows_consumed >= MAX_BEHAVIOR_ROWS:
                break
            scm_chunk = process_behavior_chunk(
                beh_chunk, news_features_df, neg_ratio, seed + part_idx, split_source,
                global_row_limit=MAX_BEHAVIOR_ROWS,
                rows_consumed_so_far=rows_consumed,
            )
            rows_consumed += len(beh_chunk)
            if scm_chunk.empty:
                continue
            scm_chunk["split_bucket"] = scm_chunk["impression_id"].apply(
                lambda x: hash_split(x, SPLIT_RATIOS)
            )
            for bucket in ["train", "val", "test"]:
                bucket_df = scm_chunk[scm_chunk["split_bucket"] == bucket].copy()
                if not bucket_df.empty:
                    bucket_df = bucket_df.drop(columns=["split_bucket"])
                    out_path = output_dir / f"scm_{bucket}" / f"part_{part_idx:05d}.parquet"
                    save_parquet(bucket_df, out_path)
                    stats[bucket] += len(bucket_df)
                    stats["total_rows"] += len(bucket_df)
            part_idx += 1
            del beh_chunk, scm_chunk
            gc.collect()

    return stats

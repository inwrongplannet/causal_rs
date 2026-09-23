import os
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def canonicalize_split_files(split_dir: Path) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    news_candidates = sorted(split_dir.rglob("news.tsv"))
    behavior_candidates = sorted(split_dir.rglob("behaviors.tsv"))
    if not news_candidates or not behavior_candidates:
        return {}
    news_src = news_candidates[0]
    behavior_src = behavior_candidates[0]
    news_dst = split_dir / "news.tsv"
    behavior_dst = split_dir / "behaviors.tsv"
    if news_src.resolve() != news_dst.resolve():
        shutil.copy2(news_src, news_dst)
    if behavior_src.resolve() != behavior_dst.resolve():
        shutil.copy2(behavior_src, behavior_dst)
    return {"news": news_dst, "behaviors": behavior_dst}


def discover_split_pairs(root_dir: Path) -> dict[str, dict[str, Path]]:
    discovered: dict[str, dict[str, Path]] = {}
    if not root_dir.exists():
        return discovered
    for news_path in sorted(root_dir.rglob("news.tsv")):
        marker = news_path.parent.as_posix().lower()
        if "train" in marker:
            split = "train"
        elif "dev" in marker or "valid" in marker or "val" in marker:
            split = "dev"
        elif "test" in marker:
            split = "test"
        else:
            continue
        if split in discovered:
            continue
        behavior_path = news_path.parent / "behaviors.tsv"
        if not behavior_path.exists():
            alt_candidates = sorted(news_path.parent.rglob("behaviors.tsv"))
            if not alt_candidates:
                continue
            behavior_path = alt_candidates[0]
        discovered[split] = {"news": news_path, "behaviors": behavior_path}
    return discovered


def prepare_mind_large_dataset(raw_large_dir: Path, external_candidates: Sequence[Path]) -> dict[str, Path]:
    split_dirs = {
        "train": raw_large_dir / "train",
        "dev": raw_large_dir / "dev",
        "test": raw_large_dir / "test",
    }
    resolved: dict[str, dict[str, Path]] = {}
    for split, split_dir in split_dirs.items():
        canonical = canonicalize_split_files(split_dir)
        if canonical:
            resolved[split] = canonical

    for candidate_root in external_candidates:
        if not candidate_root.exists():
            continue
        for zip_path in sorted(candidate_root.rglob("*.zip")):
            name = zip_path.name.lower()
            if "mindlarge" not in name and "mind_large" not in name:
                continue
            if "train" in name:
                split = "train"
            elif "dev" in name or "valid" in name or "val" in name:
                split = "dev"
            elif "test" in name:
                split = "test"
            else:
                continue
            extract_zip(zip_path, split_dirs[split])

    env_urls = {
        "train": os.getenv("MIND_LARGE_TRAIN_URL", "").strip(),
        "dev": os.getenv("MIND_LARGE_DEV_URL", "").strip(),
        "test": os.getenv("MIND_LARGE_TEST_URL", "").strip(),
    }
    for split, url in env_urls.items():
        if not url:
            continue
        zip_destination = raw_large_dir / f"MINDlarge_{split}.zip"
        if not zip_destination.exists():
            download_file(url, zip_destination)
        extract_zip(zip_destination, split_dirs[split])

    for candidate_root in external_candidates:
        discovered = discover_split_pairs(candidate_root)
        for split, file_pair in discovered.items():
            split_dirs[split].mkdir(parents=True, exist_ok=True)
            news_dst = split_dirs[split] / "news.tsv"
            behavior_dst = split_dirs[split] / "behaviors.tsv"
            if file_pair["news"].resolve() != news_dst.resolve():
                shutil.copy2(file_pair["news"], news_dst)
            if file_pair["behaviors"].resolve() != behavior_dst.resolve():
                shutil.copy2(file_pair["behaviors"], behavior_dst)

    for split, split_dir in split_dirs.items():
        canonical = canonicalize_split_files(split_dir)
        if canonical:
            resolved[split] = canonical

    missing_required = [split for split in ("train", "dev") if split not in resolved]
    if missing_required:
        raise FileNotFoundError(
            "Missing required MIND-large splits. Expected train/dev news.tsv and behaviors.tsv. "
            "Place local files under data/raw/MIND-large/{train,dev}/ or provide "
            "MIND_LARGE_TRAIN_URL and MIND_LARGE_DEV_URL environment variables."
        )

    result = {
        "train_news": resolved["train"]["news"],
        "train_behaviors": resolved["train"]["behaviors"],
        "dev_news": resolved["dev"]["news"],
        "dev_behaviors": resolved["dev"]["behaviors"],
    }
    if "test" in resolved:
        result["test_news"] = resolved["test"]["news"]
        result["test_behaviors"] = resolved["test"]["behaviors"]
    return result


def prepare_mind_small_dataset(raw_small_dir: Path, external_candidates: Sequence[Path]) -> dict[str, Path]:
    split_dirs = {
        "train": raw_small_dir / "train",
        "dev": raw_small_dir / "dev",
        "test": raw_small_dir / "test",
    }
    resolved: dict[str, dict[str, Path]] = {}
    for split, split_dir in split_dirs.items():
        canonical = canonicalize_split_files(split_dir)
        if canonical:
            resolved[split] = canonical

    for candidate_root in external_candidates:
        if not candidate_root.exists():
            continue
        for zip_path in sorted(candidate_root.rglob("*.zip")):
            name = zip_path.name.lower()
            if "mindsmall" not in name and "mind_small" not in name:
                continue
            if "train" in name:
                split = "train"
            elif "dev" in name or "valid" in name or "val" in name:
                split = "dev"
            elif "test" in name:
                split = "test"
            else:
                continue
            extract_zip(zip_path, split_dirs[split])

    env_urls = {
        "train": os.getenv("MIND_SMALL_TRAIN_URL", "").strip(),
        "dev": os.getenv("MIND_SMALL_DEV_URL", "").strip(),
        "test": os.getenv("MIND_SMALL_TEST_URL", "").strip(),
    }
    for split, url in env_urls.items():
        if not url:
            continue
        zip_destination = raw_small_dir / f"MINDsmall_{split}.zip"
        if not zip_destination.exists():
            download_file(url, zip_destination)
        extract_zip(zip_destination, split_dirs[split])

    for candidate_root in external_candidates:
        discovered = discover_split_pairs(candidate_root)
        for split, file_pair in discovered.items():
            split_dirs[split].mkdir(parents=True, exist_ok=True)
            news_dst = split_dirs[split] / "news.tsv"
            behavior_dst = split_dirs[split] / "behaviors.tsv"
            if file_pair["news"].resolve() != news_dst.resolve():
                shutil.copy2(file_pair["news"], news_dst)
            if file_pair["behaviors"].resolve() != behavior_dst.resolve():
                shutil.copy2(file_pair["behaviors"], behavior_dst)

    for split, split_dir in split_dirs.items():
        canonical = canonicalize_split_files(split_dir)
        if canonical:
            resolved[split] = canonical

    missing_required = [split for split in ("train", "dev") if split not in resolved]
    if missing_required:
        raise FileNotFoundError(
            "Missing required MIND-small splits. Expected train/dev news.tsv and behaviors.tsv. "
            "Place local files under data/raw/MIND-small/{train,dev}/ or provide "
            "MIND_SMALL_TRAIN_URL and MIND_SMALL_DEV_URL environment variables."
        )

    result = {
        "train_news": resolved["train"]["news"],
        "train_behaviors": resolved["train"]["behaviors"],
        "dev_news": resolved["dev"]["news"],
        "dev_behaviors": resolved["dev"]["behaviors"],
    }
    if "test" in resolved:
        result["test_news"] = resolved["test"]["news"]
        result["test_behaviors"] = resolved["test"]["behaviors"]
    return result


def load_news_frames(news_paths: Sequence[Path]) -> pd.DataFrame:
    columns = [
        "NewsID", "Category", "SubCategory", "Title", "Abstract",
        "URL", "TitleEntities", "AbstractEntities",
    ]
    frames = []
    for path in news_paths:
        frame = pd.read_csv(
            path, sep="\t", names=columns, header=None, dtype=str,
            keep_default_na=False, na_filter=False, encoding="utf-8",
        )
        frames.append(frame)
    news_df = pd.concat(frames, ignore_index=True)
    news_df["NewsID"] = news_df["NewsID"].astype(str)
    news_df = news_df.drop_duplicates(subset=["NewsID"], keep="first")
    return news_df.reset_index(drop=True)


def load_behavior_frames(behavior_paths: Sequence[Path]) -> pd.DataFrame:
    columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
    frames = []
    for path in behavior_paths:
        frame = pd.read_csv(
            path, sep="\t", names=columns, header=None, dtype=str,
            keep_default_na=False, na_filter=False, encoding="utf-8",
        )
        frames.append(frame)
    behaviors_df = pd.concat(frames, ignore_index=True)
    behaviors_df["ImpressionID"] = behaviors_df["ImpressionID"].astype(str)
    behaviors_df["UserID"] = behaviors_df["UserID"].astype(str)
    return behaviors_df.reset_index(drop=True)


def load_entity_embeddings(vec_path: Path) -> dict[str, np.ndarray]:
    """Load pre-trained TransE entity embeddings from a MIND .vec file.

    Format: one entity per line: ``entity_id dim1 dim2 ... dimN`` (space-separated).

    Args:
        vec_path: Path to the ``entity_embedding.vec`` file.

    Returns:
        Dict mapping entity ID (str) -> embedding vector (np.ndarray float32).
    """
    lookup: dict[str, np.ndarray] = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            entity_id = parts[0]
            vec = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            lookup[entity_id] = vec
    return lookup


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save parquet at {output_path}. Install pyarrow or fastparquet."
        ) from exc

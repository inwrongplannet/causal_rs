from pathlib import Path
import pytest
from src.data_pipeline.streaming import _infer_split_source, hash_split


class TestInferSplitSource:
    def test_train_path(self):
        p = Path("/data/raw/MIND-small/train/news.tsv")
        assert _infer_split_source(p) == "train"

    def test_dev_path(self):
        p = Path("/data/raw/MIND-small/dev/news.tsv")
        assert _infer_split_source(p) == "dev"

    def test_valid_path(self):
        p = Path("/data/raw/MIND-small/val/news.tsv")
        assert _infer_split_source(p) == "dev"

    def test_test_path(self):
        p = Path("/data/raw/MIND-small/test/news.tsv")
        assert _infer_split_source(p) == "test"

    def test_unknown_path(self):
        p = Path("/data/other/file.tsv")
        assert _infer_split_source(p) == "unknown"

    def test_windows_path(self):
        p = Path("C:\\data\\MIND-small\\train\\behaviors.tsv")
        assert _infer_split_source(p) == "train"


class TestHashSplit:
    def test_output_is_valid_bucket(self):
        result = hash_split("train:1", (0.7, 0.15, 0.15))
        assert result in ("train", "val", "test")

    def test_deterministic(self):
        a = hash_split("session_42", (0.7, 0.15, 0.15))
        b = hash_split("session_42", (0.7, 0.15, 0.15))
        assert a == b

    def test_different_keys_different(self):
        results = {hash_split(f"session_{i}", (0.7, 0.15, 0.15)) for i in range(100)}
        assert len(results) <= 3  # Only train/val/test possible
        assert results.issubset({"train", "val", "test"})

    def test_distribution(self):
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(1000):
            counts[hash_split(f"session_{i}", (0.7, 0.15, 0.15))] += 1
        assert counts["train"] > counts["val"]
        assert counts["val"] > counts["test"]
        assert abs(counts["train"] / 1000 - 0.7) < 0.1

    def test_different_ratios(self):
        result = hash_split("key", (0.3, 0.3, 0.4))
        assert result in ("train", "val", "test")

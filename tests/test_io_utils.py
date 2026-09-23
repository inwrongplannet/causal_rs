from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.io_utils import save_parquet
from src.data_pipeline.parsers import parse_history


class TestSaveParquet:
    def test_saves_and_loads(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = tmp_path / "test.parquet"
        save_parquet(df, out)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert loaded.equals(df)

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        out = tmp_path / "sub" / "nested" / "test.parquet"
        save_parquet(df, out)
        assert out.exists()

    def test_raises_on_bad_path(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(RuntimeError, match="Failed to save parquet"):
            save_parquet(df, Path(""))  # nosec


class TestParseHistory:
    def test_multi_item(self):
        assert parse_history("N1 N2 N3") == ["N1", "N2", "N3"]

    def test_empty(self):
        assert parse_history(None) == []

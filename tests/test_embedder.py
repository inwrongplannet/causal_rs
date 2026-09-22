import numpy as np
import pytest
from src.data_pipeline.embedder import build_title_encoder


class TestBuildTitleEncoder:
    def test_fallback_to_hashing_when_no_sentence_transformers(self, monkeypatch):
        monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError("no module")))
        encoder, meta = build_title_encoder("fake-model", 100)
        texts = ["hello world", "test article"]
        embeddings = encoder(texts)
        assert embeddings.shape == (2, 100)
        assert embeddings.dtype == np.float32

    def test_fallback_metadata(self, monkeypatch):
        monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError("no module")))
        encoder, meta = build_title_encoder("fake-model", 768)
        assert meta["encoder_type"] == "hashing_vectorizer_fallback"
        assert meta["model"] == "hashing"
        assert meta["embedding_dim"] == "768"

    def test_fallback_normalized(self, monkeypatch):
        monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError("no module")))
        encoder, meta = build_title_encoder("fake-model", 100)
        texts = ["hello world"]
        embeddings = encoder(texts)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_fallback_handles_empty_string(self, monkeypatch):
        monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError("no module")))
        encoder, meta = build_title_encoder("fake-model", 100)
        embeddings = encoder([""])
        assert embeddings.shape == (1, 100)

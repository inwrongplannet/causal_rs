# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| MIND dataset (local files) | File system (TSV + Parquet) | Raw news recommendation dataset (Microsoft News) | None (local files) | High — pipeline cannot run without this data | `src/data_pipeline/io_utils.py`, `data/raw/MIND-small/` |
| HuggingFace Hub (SBERT) | Remote model hub (via `sentence-transformers`) | Pre-trained `all-mpnet-base-v2` for title embedding | None (public model) | High — title embedding requires this download | `src/config.py` (`TITLE_EMBED_MODEL`), `src/data_pipeline/embedder.py` |

The project has **no** external API calls, database connections, message queues, or authentication services. The only network-dependent operation is the one-time download of the sentence-transformers model from HuggingFace Hub.

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| MIND-small TSV files (`data/raw/MIND-small/`) | Raw input (behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec) | `pandas.read_csv` with custom parsers | File not found after interrupted extraction | `src/data_pipeline/io_utils.py`, `src/data_pipeline/parsers.py` |
| Parquet artifacts (`data/scm_parts/`, `data/*.parquet`) | Intermediate + final SCM DataFrames | `pandas.DataFrame.to_parquet()`, `pyarrow.parquet.read_table()` | ArrowMemoryError with large contiguous tables (was fixed with fragment-by-fragment loading) | `src/data_pipeline/io_utils.py`, `src/data_pipeline/scm_builder.py` |
| Pickle artifacts (`artifacts/*.pkl`, `artifacts/checkpoints/*.zip`) | Trained GCM model, CDI cache, PPO policy | `pickle.dump/load`, SB3 `model.save/load` | Pickle format brittleness across Python/package versions | `src/counterfactual/precompute_cdi.py`, `src/rl_agent/train_ppo.py` |

### 3) Secrets and Credentials Handling

- **Credential sources**: None. The project uses only public datasets and public HuggingFace models.
- **Hardcoding checks**: No hardcoded credentials, tokens, or API keys found anywhere in the codebase.
- **Rotation or lifecycle notes**: N/A — no secrets to rotate.

### 4) Reliability and Failure Behavior

- **Retry/backoff behavior**: None. No remote API calls that would require retry logic.
- **Timeout policy**: None configured. Jupyter nbconvert `--execute` uses a 3600s timeout per notebook.
- **Circuit-breaker or fallback behavior**: Present only for GPU operations — if cupy fails, the chunk falls back to CPU (`src/gpu_utils.py`). If sentence-transformers fails to load, falls back to `HashingVectorizer` (`src/data_pipeline/embedder.py`).

### 5) Observability for Integrations

- **Logging around external calls**: No external calls to log (aside from module/hub loading which produces its own HuggingFace logs).
- **Metrics/tracing coverage**: TensorBoard logs for PPO training only (`artifacts/tb_logs/`). No other metrics collection.
- **Missing visibility gaps**: No logging of data loading duration or disk I/O bottlenecks. The only timing information comes from manual notebook cell execution times.

### 6) Evidence

- `src/data_pipeline/io_utils.py` — data loading with multiple fallback strategies
- `src/data_pipeline/embedder.py` — SBERT model loading with HashingVectorizer fallback
- `src/gpu_utils.py` — chunked GPU ops with per-chunk CPU fallback
- `artifacts/tb_logs/` — TensorBoard training logs

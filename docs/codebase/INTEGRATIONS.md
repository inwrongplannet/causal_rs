# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| MIND dataset | Local TSV files | News recommendation dataset (download, parse) | None (download URL) | High — pipeline cannot run without it | `src/data_pipeline/io_utils.py:14-22,68-143` |
| HuggingFace Hub (sentence-transformers) | External model download | SBERT model "all-mpnet-base-v2" for title embeddings | None | High — title embedding generation | `src/config.py:37`, `src/data_pipeline/embedder.py:24-27` |
| nltk VADER | Local nltk data | Sentiment analysis of article titles | None | Low — defaults to 0.0 | `src/data_pipeline/nlp_utils.py:48-59` |

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| Local parquet files (`data/`) | SCM training/val/test DataFrames | pandas `read_parquet` / `to_parquet` | Arrow MemoryError with large datasets; fixed with fragment loading | `src/data_pipeline/scm_builder.py:280-287`, `docs/RESEARCH_LOG.md` (Arrow MemoryError entry) |
| Local pickle files (`artifacts/`) | GCM model, CDI cache, PPO checkpoints | pickle `load`/`dump`; stable-baselines3 `PPO.load`/`save` | Version drift between pickle and code | `src/counterfactual/precompute_cdi.py:73-77,82-92` |
| TensorBoard logs (`artifacts/tb_logs/`) | Training metrics | stable-baselines3 `tensorboard_log` | None (local only) | `src/rl_agent/train_ppo.py:53,117` |

### 3) Secrets and Credentials Handling

- Credential sources: No secrets management. MIND dataset download URLs are optional env vars (`MIND_*_URL`).
- Hardcoding checks: No hardcoded credentials found.
- Rotation or lifecycle notes: N/A — no production credentials.

### 4) Reliability and Failure Behavior

- Retry/backoff behavior: None. GPU ops have a single try/except with immediate CPU fallback.
- Timeout policy: None configured. Download uses `urllib.request.urlretrieve` with no timeout.
- Circuit-breaker or fallback behavior:
  - GPU -> CPU fallback per chunk (`src/gpu_utils.py:138-156`)
  - sentence-transformers -> HashingVectorizer fallback (`src/data_pipeline/embedder.py:22-62`)
  - cuML PCA -> sklearn PCA fallback (`src/data_pipeline/scm_builder.py:13-22`)

### 5) Observability for Integrations

- Logging around external calls: Yes. `logger.info`/`warning` before/after GPU ops, PCA fallback, sentence-transformers init.
- Metrics/tracing coverage: None beyond TensorBoard logs for PPO training.
- Missing visibility gaps: No instrumentation for MIND dataset download times, no GCM fit data quality metrics, no CDI compute throughput tracking.

### 6) Evidence

- `src/data_pipeline/io_utils.py` (download, extraction, file I/O)
- `src/data_pipeline/embedder.py` (sentence-transformers / fallback)
- `src/data_pipeline/nlp_utils.py` (nltk VADER sentiment)
- `src/gpu_utils.py` (GPU -> CPU fallback)
- `src/config.py:37` (`TITLE_EMBED_MODEL` config)

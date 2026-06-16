# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `src/` | Python package (5 subpackages, 25 files) | `README.md:65-72`, scan output lines 155-182 |
| `src/data_pipeline/` | MIND TSV I/O, parsing, feature engineering, SCM builder, streaming | `src/data_pipeline/__init__.py` (lines 169-178) |
| `src/causal_model/` | DoWhy causal inference: model creation, graph GML, CDI batch, refutation | `src/causal_model/__init__.py` (lines 157-161) |
| `src/counterfactual/` | GCM graph building, fitting, counterfactual queries, CDI precomputation | `src/counterfactual/__init__.py` (lines 164-167) |
| `src/rl_agent/` | Gymnasium environment, PPO training | `src/rl_agent/__init__.py` (lines 179-180) |
| `src/evaluation/` | Metrics (NDCG, Precision, ILD), significance tests | `src/evaluation/__init__.py` (lines 178-179) |
| `notebooks/` | 10 Jupyter notebooks across 5 pipeline phases (MIND-small + MIND-large) | Scan output lines 134-147 |
| `tests/` | pytest test suite (12 test files) | Scan output lines 186-198 |
| `data/` | Raw MIND-small/MIND-large, interim embeddings, processed reports, SCM parquet | Scan output lines 39-117 |
| `artifacts/` | Trained GCM pickle, CDI cache pickle, PPO checkpoints, TensorBoard logs | Scan output lines 16-37 |
| `scripts/` | Standalone scripts: training phases, GCM refit, evaluation | Scan output lines 151-153 |
| `config.yaml` | Project configuration (dataset, seed, dims, GPU) | `config.yaml` |
| `docs/` | Research log, codebase documentation, CausalRS reference materials | `docs/` (scan lines 118-133) |

### 2) Entry Points

- Main runtime entry: Jupyter notebooks in `notebooks/` (5 phases, sequentially executed)
- Secondary entry points (scripts): `scripts/run_phase4_5.py`, `scripts/run_phase4_large.py`, `scripts/refit_full_gcm_and_cdi.py`, `scripts/eval_only.py`
- How entry is selected: Phase notebooks are run sequentially via `jupyter nbconvert --execute`. Scripts provide CLI alternatives for headless execution (e.g., `python scripts/refit_full_gcm_and_cdi.py --large`).

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `data_pipeline` | MIND I/O, TSV parsing, feature extraction, SCM DataFrame building, streaming | Causal inference logic, RL training, evaluation metrics |
| `causal_model` | DoWhy CausalModel creation, ATE estimation, refutation tests | Data loading/parsing, GCM fitting, RL env |
| `counterfactual` | NetworkX DAG construction, GCM fitting, counterfactual queries, CDI caching | Data pipeline, PPO training, evaluation |
| `rl_agent` | Gymnasium env, PPO model definition and training | Causal inference, evaluation metrics |
| `evaluation` | NDCG/Precision/ILD, significance tests, replay evaluation | Training logic, data pipeline, causal modeling |

### 4) Naming and Organization Rules

- File naming pattern: snake_case (e.g., `scm_builder.py`, `gcm_fit.py`, `train_ppo.py`)
- Directory organization pattern: By domain/function (causal_model, data_pipeline, rl_agent, evaluation, counterfactual)
- Import aliasing or path conventions: Absolute imports from `src` package (e.g., `from src.config import SEED`, `from src.data_pipeline.nlp_utils import l2_normalize`)
- Test file naming: `test_<module>.py` (e.g., `test_features.py`, `test_scm_builder.py`)

### 5) Evidence

- Scan output (directory tree, lines 2-199)
- `README.md` (structure section, lines 62-78)
- Module `__init__.py` files in each subpackage
- Import patterns across source files

## Extended Sections (Optional)

### Subdirectory deep map

- `src/data_pipeline/`:
  - `io_utils.py` — File download, zip extraction, TSV loading, parquet save, entity vec loading
  - `parsers.py` — History/impressions/entity string parsing
  - `embedder.py` — Title encoder (sentence-transformers with HashingVectorizer fallback)
  - `nlp_utils.py` — L2 normalize, cosine diversity, hash embedding, sentiment, batch GPU wrappers
  - `features.py` — News feature computation, session user features, negative sampling
  - `scm_builder.py` — SCM DataFrame construction, PCA reduction, split, quality checks
  - `streaming.py` — Chunked behavior processing, hash-based split

- `src/counterfactual/`:
  - `gcm_fit.py` — Causal DAG builder, GCM fitting
  - `queries.py` — Counterfactual diversity prediction
  - `precompute_cdi.py` — Batch CDI cache precomputation

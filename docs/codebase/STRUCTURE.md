# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `src/` | Main application source code (25 Python files across 5 packages) | `src/counterfactual/`, `src/data_pipeline/`, `src/rl_agent/`, etc. |
| `notebooks/` | Pipeline execution entry points (5 Jupyter notebooks, phases 1–5) | `phase_1_data_pipeline_mind_small.ipynb` through `phase_5_evaluation.ipynb` |
| `tests/` | pytest test suite (12 test files + conftest) | `tests/test_counterfactual.py`, `tests/test_environment.py`, etc. |
| `data/` | Data artifacts (raw MIND, interim embeddings, processed SCM data, split parquet parts) | `data/raw/MIND-small/`, `data/interim/`, `data/scm_parts/` |
| `artifacts/` | Trained models and cache (GCM model pickle, CDI cache, PPO policy checkpoint) | `artifacts/gcm_model.pkl`, `artifacts/cdi_cache.pkl`, `artifacts/checkpoints/` |
| `docs/` | Documentation | `docs/RESEARCH_LOG.md`, `docs/resources/` (CausalRS reference docs), `docs/codebase/` |
| `scripts/` | Currently empty — standalone runner scripts were moved into notebooks | `scripts/` (empty) |

### 2) Entry Points

- **Main runtime entry**: There is no single CLI entry point. The project is executed via Jupyter notebooks (`notebooks/phase_*.ipynb`), each corresponding to one pipeline phase.
- **Secondary entry points**: `pytest tests/ -v` runs the test suite. Individual Python modules can be imported directly (`from src.counterfactual.gcm_fit import fit_gcm`).
- **How entry is selected**: The user runs notebooks sequentially: Phase 1 (data pipeline) → Phase 2 (causal modeling) → Phase 3 (counterfactual GCM + CDI) → Phase 4 (PPO training) → Phase 5 (evaluation).

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `src/data_pipeline/` | MIND dataset I/O, parsing, feature engineering, embedding, SCM DataFrame builder, streaming | Causal inference, RL training, evaluation logic |
| `src/causal_model/` | DoWhy causal graph GML, model/estimand/estimate wrappers, CDI batch, refutations | Data loading, counterfactual queries, RL environment |
| `src/counterfactual/` | GCM causal graph (NetworkX), SCM fitting, counterfactual query, CDI precomputation | Data pipeline logic, DoWhy refutations, RL agent |
| `src/rl_agent/` | Gymnasium environment (`NewsRecommendEnv`), PPO training via stable-baselines3 | Causal inference, evaluation metrics, data pipeline |
| `src/evaluation/` | NDCG, Precision, ILD, homogeneity, replay evaluation, significance testing | RL training loops, causal graph construction |
| `src/config.py` | Path constants and pipeline configuration (embedding dims, PCA components, GPU settings) | Business logic, data transformations |
| `src/gpu_utils.py` | GPU-accelerated array ops with cupy/numpy fallback | Any domain-specific logic |

### 4) Naming and Organization Rules

- **File naming pattern**: `snake_case.py` — e.g., `gcm_fit.py`, `precompute_cdi.py`, `scm_builder.py`, `nlp_utils.py`
- **Directory organization pattern**: **Layer-based** — split by pipeline stage (data_pipeline → causal_model → counterfactual → rl_agent → evaluation), not by feature
- **Import aliasing or path conventions**: All imports from `src.*` using absolute package imports (e.g., `from src.counterfactual.queries import predict_diversity_counterfactual`). No relative imports.

### 5) Evidence

- `src/` directory listing (25 Python files, 5 subpackages)
- `notebooks/` listing (5 source notebooks)
- `tests/` listing (12 test files)
- `src/config.py` for path constants

# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python 3.13+ | `.venv` created with 3.14.5, `requirements.txt`, `src/config.py` |
| Runtime + version | CPython 3.14.5 | `python --version`, `src/config.py` |
| Package manager | pip (via virtual env `.venv/`) | `requirements.txt` is a full `pip freeze` output |
| Module/build system | Standard Python package (no build backend declared) | `src/__init__.py`, no `pyproject.toml` or `setup.py` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| torch | 2.6.0+cu124 | GPU tensor ops, PCA fallback, PPO policy inference | `.venv pip list`, `src/gpu_utils.py` |
| pyarrow | 19+ | Parquet I/O for large dataset fragments | `src/data_pipeline/io_utils.py`, `requirements.txt` |
| dowhy | 0.14 | Causal inference (identification, estimation, refutation, GCM) | `src/causal_model/model.py`, `src/counterfactual/gcm_fit.py` |
| stable-baselines3 | 2.8.0 | PPO RL agent training | `src/rl_agent/train_ppo.py` |
| gymnasium | 1.2.3 | RL environment interface (`NewsRecommendEnv`) | `src/rl_agent/environment.py` |
| pandas | 3.0.3 | DataFrame-based data pipeline, features, SCM building | `src/data_pipeline/scm_builder.py`, everywhere |
| numpy | 2.4.6 | Numerical ops, embeddings, diversity computation | `src/gpu_utils.py`, `src/data_pipeline/nlp_utils.py` |
| scikit-learn | 1.9.0 | PCA fallback, `HashingVectorizer` fallback, logistic regression | `src/data_pipeline/scm_builder.py`, `src/data_pipeline/embedder.py`, `src/causal_model/model.py` |
| sentence-transformers | 5.5.1 | Title embedding (all-mpnet-base-v2, 768-dim) | `src/config.py` (`TITLE_EMBED_MODEL`), `src/data_pipeline/embedder.py` |
| transformers | 5.10.2 | HuggingFace model loading (backing sentence-transformers) | `.venv pip list` |
| networkx | 3.6.1 | Causal DAG construction for GCM | `src/counterfactual/gcm_fit.py` |
| pyyaml | 6.0.3 | YAML config file loader (config.yaml) | `src/config.py`, `config.yaml` |
| matplotlib | 3.10.9 | Positivity-check plot, metric visualizations | `src/causal_model/model.py` |
| tqdm | 4.68.2 | Progress bars in CDI precomputation | `src/counterfactual/precompute_cdi.py` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| pytest | Test runner (140 tests) | `tests/*.py`, `pytest==9.0.3` |
| pytest-cov | Coverage measurement (installed, not configured) | `pytest-cov==7.1.0` |
| black | Code formatter (declared in requirements.txt, not installed) | `black==26.1.0` in `requirements.txt` |

No linting config files found in project root. No pyproject.toml or setup.cfg with tool configs.

### 4) Key Commands

```bash
# Install (from existing venv)
.venv\Scripts\pip install -r requirements.txt

# Run all tests
.venv\Scripts\python -m pytest tests/ -v

# Run a specific test file
.venv\Scripts\python -m pytest tests/test_counterfactual.py -v

# Execute a pipeline phase (Jupyter notebook)
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/phase_3_counterfactual_gcm.ipynb

# Run a Python module directly
.venv\Scripts\python -c "from src.counterfactual.gcm_fit import fit_gcm; ..."
```

### 5) Environment and Config

- Config sources: `config.yaml` (YAML file) + `src/config.py` (loader with env var and CLI override support). Priority: CLI args > env vars (`CAUSAL_RS_*`) > YAML > defaults.
- Required env vars: None hard-required. `CAUSAL_RS_*` can override any YAML key. `NVIDIA_CUDA_PATH` auto-detected for GPU (see `src/gpu_utils.py`). No `.env` file or `.env.example` found.
- Deployment/runtime constraints: Windows 10+ primary target; Linux/macOS secondary. GPU optional (auto-fallback to CPU in all paths). MIND-small or MIND-large dataset required in `data/raw/MIND-small/` or `data/raw/MIND-large/`. MIND-large (~10x data) requires 32GB+ RAM for full processing.

### 6) Evidence

- `requirements.txt` — full dependency list
- `src/config.py` — runtime constants and paths (YAML loader)
- `config.yaml` — editable YAML config file
- `src/gpu_utils.py` — GPU availability and batch-size config

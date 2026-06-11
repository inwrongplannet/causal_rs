# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python 3.13+ | `.venv` created with 3.13.5, `requirements.txt`, `src/config.py` |
| Runtime + version | CPython 3.13.5 | Tracks in `python-3.13.5-amd64.exe` (installer), `src/config.py` |
| Package manager | pip (via virtual env `.venv/`) | `requirements.txt` is a full `pip freeze` output |
| Module/build system | Standard Python package (no build backend declared) | `src/__init__.py`, no `pyproject.toml` or `setup.py` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| torch | 2.6.0+cu124 | GPU tensor ops, PCA fallback, PPO policy inference | `pip list`, `src/gpu_utils.py` |
| dowhy | 0.14 | Causal inference (identification, estimation, refutation, GCM) | `src/causal_model/model.py`, `src/counterfactual/gcm_fit.py` |
| stable-baselines3 | 2.8.0 | PPO RL agent training | `src/rl_agent/train_ppo.py` |
| gymnasium | 1.2.3 | RL environment interface (`NewsRecommendEnv`) | `src/rl_agent/environment.py` |
| pandas | 2.3.2 | DataFrame-based data pipeline, features, SCM building | `src/data_pipeline/scm_builder.py`, everywhere |
| numpy | 1.24.4 | Numerical ops, embeddings, diversity computation | `src/gpu_utils.py`, `src/data_pipeline/nlp_utils.py` |
| scikit-learn | 1.7.2 | PCA fallback, `HashingVectorizer` fallback, logistic regression | `src/data_pipeline/scm_builder.py`, `src/data_pipeline/embedder.py`, `src/causal_model/model.py` |
| sentence-transformers | 5.5.1 | Title embedding (all-mpnet-base-v2, 768-dim) | `src/config.py` (`TITLE_EMBED_MODEL`), `src/data_pipeline/embedder.py` |
| transformers | 5.10.2 | HuggingFace model loading (backing sentence-transformers) | `pip list` |
| networkx | 3.4.2 | Causal DAG construction for GCM | `src/counterfactual/gcm_fit.py` |
| xgboost | 3.0.5 | Potential alternative estimator (not actively used in main pipeline) | `requirements.txt` |
| shap | 0.48.0 | Model explainability (not actively used) | `requirements.txt` |
| matplotlib | 3.10.6 | Positivity-check plot, metric visualizations | `src/causal_model/model.py` |
| tqdm | (implicit) | Progress bars in CDI precomputation | `src/counterfactual/precompute_cdi.py` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| pytest | Test runner | `tests/*.py`, `pytest==8.1.1` in `requirements.txt` |
| black | Code formatter | `black==26.1.0` in `requirements.txt` |

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

- Config sources: `src/config.py` (single config module)
- Required env vars: `NVIDIA_CUDA_PATH` (auto-detected for GPU acceleration, see `src/gpu_utils.py`). No `.env` file or `.env.example` found.
- Deployment/runtime constraints: Windows 10+ primary target; Linux/macOS secondary. GPU optional (auto-fallback to CPU in all paths). MIND-small dataset required in `data/raw/MIND-small/`

### 6) Evidence

- `requirements.txt` — full dependency list
- `src/config.py` — runtime constants and paths
- `src/gpu_utils.py` — GPU availability and batch-size config

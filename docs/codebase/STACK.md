# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python 3.13+ | `README.md:7`, Python 3.13.13 via `python --version` |
| Runtime + version | CPython 3.13.13 | `python --version` |
| Package manager | pip 25+ | `README.md:84-85`, `requirements.txt` |
| Module/build system | Pure Python (no build step), Jupyter notebooks | No `setup.py`/`pyproject.toml`, notebooks in `notebooks/` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| PyTorch | 2.6.0+cu124 | Deep learning backend, GPU tensor ops for env inference | `README.md:117` |
| DoWhy | 0.14 | Causal inference (CausalModel, GCM, refutation) | `README.md:118`, `src/causal_model/model.py:8-9` |
| stable-baselines3 | 2.8.0 | PPO RL agent training | `README.md:119`, `src/rl_agent/train_ppo.py:5-6` |
| Gymnasium | 1.2.3 | RL environment interface | `README.md:120`, `src/rl_agent/environment.py:4` |
| pandas | 3.0.3 | Data pipeline, parquet I/O, DataFrame ops | `README.md:121` |
| numpy | 2.4.6 | Numerical ops, GPU fallback | `README.md:122` |
| scikit-learn | 1.9.0 | PCA, HashingVectorizer fallback | `README.md:123` |
| sentence-transformers | 5.5.1 | Title embedding (all-mpnet-base-v2) | `README.md:124`, `src/data_pipeline/embedder.py:24-27` |
| NetworkX | 3.6.1 | Causal DAG for GCM | `README.md:125`, `src/counterfactual/gcm_fit.py:2` |
| PyYAML | 6.0.3 | Config file loader | `README.md:126`, `src/config.py:20` |
| matplotlib | 3.10.9 | Propensity score plots | `README.md:127`, `src/causal_model/model.py:77` |
| scipy | — | Significance tests (paired t-test, Cohen's d) | `src/evaluation/metrics.py:3,193` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| pytest | 9.0.3 | Test runner | `README.md:128`, `.pytest_cache/v/cache/` |
| pytest-cov | 7.1.0 | Optional coverage | `README.md:129` |
| black | 26.1.0 | Code formatter (installed but not configured) | `requirements.txt` (scan: line 255) |
| Jupyter | — | Notebook execution environment | `README.md:144-168` |

### 4) Key Commands

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m pytest tests/ --cov=src
jupyter notebook
```

### 5) Environment and Config

- Config sources: `config.yaml`, env vars with prefix `CAUSAL_RS_`, CLI args `--config.key=value`, hardcoded defaults in `src/config.py`
- Required env vars: `MIND_SMALL_TRAIN_URL`, `MIND_SMALL_DEV_URL`, `MIND_SMALL_TEST_URL`, `MIND_LARGE_TRAIN_URL`, `MIND_LARGE_DEV_URL`, `MIND_LARGE_TEST_URL` (all optional, for download) — `src/data_pipeline/io_utils.py:98-101,176-179`
- Deployment/runtime constraints: Single-machine research project. No production deployment. Requires 4+ GiB VRAM for GPU acceleration. Windows SubprocVecEnv replaced with DummyVecEnv (`src/rl_agent/train_ppo.py:30-32`).

### 6) Evidence

- `README.md`
- `requirements.txt`
- `config.yaml`
- `src/config.py`
- `src/gpu_utils.py`

## Extended Sections (Optional)

### Full dependency taxonomy

- **Deep Learning/GPU**: torch, cupy (optional), cuML (optional), sentence-transformers
- **Data**: pandas, numpy, scipy, scikit-learn, pyarrow, fastparquet
- **Causal Inference**: dowhy, networkx
- **RL**: stable-baselines3, gymnasium
- **NLP**: nltk (VADER sentiment), sentence-transformers
- **Config/IO**: pyyaml, tqdm, pickle, urllib, zipfile
- **Visualization**: matplotlib, tensorboard
- **Dev**: pytest, pytest-cov, black

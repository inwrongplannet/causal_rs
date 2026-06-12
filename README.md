<div align="center">

# causal_rs

**Causal Recommendation System** — A research pipeline combining structural causal models, counterfactual reasoning, and reinforcement learning for diversity-aware news recommendation.

[![Python 3.14](https://img.shields.io/badge/Python-3.14+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.14-7b2ff7?style=flat-square)](https://github.com/py-why/dowhy)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.8-green?style=flat-square)](https://stable-baselines3.readthedocs.io)
[![MIND](https://img.shields.io/badge/Dataset-MIND--small%20%7C%20MIND--large-ff6f00?style=flat-square)](https://msnews.github.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#license)
[![Status](https://img.shields.io/badge/Status-Research-important?style=flat-square)]()

</div>

**causal_rs** explores whether counterfactual diversity scores (CDI) can guide a reinforcement learning policy to recommend content that is both relevant and diverse. The pipeline processes Microsoft News (MIND) data through five sequential phases — from raw TSV files to a trained PPO policy with offline evaluation.

> [!NOTE]
> This is a **research project**, not a production system. After min-max CDI normalization, PPO now shows statistically significant NDCG gains over Random (p=0.017). See `docs/RESEARCH_LOG.md` for full history.

## Features

- **Structural Causal Model** — Builds a NetworkX causal DAG with user PCA features, item-level semantic embeddings (SBERT title + Wikidata entity), and categorical attributes; identifies causal effect of exposure on diversity (ATE ≈ −0.01)
- **Counterfactual Diversity Impact (CDI)** — Fits a DoWhy GCM (StructuralCausalModel) and queries per-item `E[Y_diversity | do(A=1, item_features)]` to compute the causal diversity impact of showing each candidate article
- **Automatic GPU acceleration** — Transparent cupy/numpy fallback for all batch operations (cosine diversity, L2 normalization, PCA); sentence-transformers on GPU; chunked processing to stay within 4 GiB VRAM
- **PPO Reinforcement Learning** — Custom Gymnasium environment with observation = user history embedding + CDI, action = pick K candidates, reward = `w * R_click + (1-w) * CDI`; trained with stable-baselines3
- **Offline Evaluation** — NDCG@K, Precision@K, Intra-List Diversity (ILD), paired t-test and Cohen's d vs Random and Popularity baselines

## Architecture

```text
MIND-small / MIND-large TSV  pipeline/streaming.py         gcm_fit.py
         │                          │                            │
         ▼                          ▼                            ▼
   ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
   │ Phase 1     │ → │ Phase 2          │ → │ Phase 3                  │
   │ Data        │   │ Causal Modeling  │   │ Counterfactual GCM + CDI │
   │ Pipeline    │   │ (DoWhy ATE)      │   │ (SCM fit + query)        │
   └─────────────┘   └──────────────────┘   └──────────────────────────┘
                           │                            │
                           ▼                            ▼
                  ┌──────────────────┐   ┌──────────────────────────┐
                  │ Phase 4          │ ← │ Phase 5                  │
                  │ PPO Training     │   │ Evaluation & Significance│
                  │ (SB3)            │   │ (NDCG, Prec, ILD)        │
                  └──────────────────┘   └──────────────────────────┘
```

## Pipeline Phases

| Phase | Notebook | Module | What it does | Key output |
|-------|----------|--------|-------------|------------|
| 1 | `phase_1_data_pipeline_mind_small` / `phase_1_data_pipeline_mind_large` | `src/data_pipeline/` | Parse MIND TSV → compute news embeddings (SBERT title + Wikidata entity) → build session features → construct SCM rows with Y_diversity → PCA reduction (U_pca, I_entity_pca, I_title_pca) → train/val/test split | `scm_{train,val,test}.parquet`, `phase1_report.json` |
| 2 | `phase_2_causal_modeling` / `phase_2_causal_modeling_mind_large` | `src/causal_model/` | DoWhy `CausalModel` → backdoor identification → IPW + linear ATE estimation → refutation tests (placebo, subset, random common cause) | ATE, refutation pass |
| 3 | `phase_3_counterfactual_gcm` / `phase_3_counterfactual_gcm_mind_large` | `src/counterfactual/` | Build NetworkX causal DAG → auto-assign GCM mechanisms → `gcm.fit()` → per-item CDI via `predict_diversity_counterfactual()` → cache | `gcm_model.pkl`, `cdi_cache.pkl` |
| 4 | `phase_4_ppo_training` / `phase_4_ppo_training_mind_large` | `src/rl_agent/` | `NewsRecommendEnv` (Gymnasium) → PPO training → save checkpoint | `ppo_causal_rs_w03.zip` |
| 5 | `phase_5_evaluation` / `phase_5_evaluation_mind_large` | `src/evaluation/` | Replay evaluation on test sessions → NDCG, Precision, ILD → significance tests vs Random, Popularity | Aggregated metrics table |

## Project Structure

```
causal_rs/
├── config.yaml                 # Editable configuration (YAML + env/CLI overrides)
├── src/                        # Python package (5 subpackages, 25 files)
│   ├── causal_model/           # DoWhy causal inference (ATE, refutation, CDI batch)
│   ├── counterfactual/         # GCM graph, fit, counterfactual query, CDI cache
│   ├── data_pipeline/          # MIND I/O, parsing, features, SCM builder, streaming
│   ├── evaluation/             # NDCG, Precision, ILD, significance testing
│   ├── rl_agent/               # Gymnasium environment, PPO training
│   ├── config.py               # Config loader (reads config.yaml + env + CLI)
│   └── gpu_utils.py            # GPU-accelerated batch ops with cupy/numpy fallback
├── notebooks/                  # 10 phase notebooks (MIND-small + MIND-large)
├── tests/                      # pytest test suite (140 tests, 12 test files)
├── data/                       # Raw MIND-small / MIND-large, interim, processed, SCM parquet parts
├── artifacts/                  # Trained models, CDI cache, PPO checkpoints, TB logs
└── docs/                       # Research log, codebase docs, CausalRS reference
```

## Getting Started

### Prerequisites

- **Python 3.14+** — developed and tested on CPython 3.14.5
- **pip 25+** (comes with Python 3.14)
- **NVIDIA GPU** optional — all operations auto-fallback to CPU
- **MIND-small or MIND-large dataset** — download from [MIND News Dataset](https://msnews.github.io/)

### Virtual Environment Setup

The project was built and tested with the following exact environment. Clone it precisely:

```bash
# 1. Clone
git clone https://github.com/your-username/causal_rs.git
cd causal_rs

# 2. Create venv with Python 3.14+
python -m venv .venv

# 3. Activate
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS

# 4. Upgrade pip inside venv
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

**Key packages (as tested):**

| Package | Version | Role |
|---------|---------|------|
| `python` | 3.14.5 | Runtime |
| `torch` | 2.6.0+cu124 | Deep learning / GPU ops |
| `dowhy` | 0.14 | Causal inference |
| `stable-baselines3` | 2.8.0 | PPO RL agent |
| `gymnasium` | 1.2.3 | RL environment interface |
| `pandas` | 3.0.3 | Data pipeline |
| `numpy` | 2.4.6 | Numerical ops |
| `scikit-learn` | 1.9.0 | PCA, vectorizer fallback |
| `sentence-transformers` | 5.5.1 | Title embeddings (all-mpnet-base-v2) |
| `networkx` | 3.6.1 | Causal DAG |
| `pyyaml` | 6.0.3 | Config file loader |
| `matplotlib` | 3.10.9 | Plots |
| `pytest` | 9.0.3 | Test runner (140 tests) |
| `pytest-cov` | 7.1.0 | Coverage (optional) |

> `requirements.txt` is a full `pip freeze` (767 lines). Only the packages above are actively used.

### Configuration

Edit `config.yaml` in the project root to override defaults. Priority (highest → lowest):

1. CLI args: `--config.seed=99`
2. Env vars: `CAUSAL_RS_SEED=99`
3. YAML: `config.yaml`
4. Hardcoded defaults in `src/config.py`

### Run the Pipeline

Each phase is a Jupyter notebook. Choose the dataset variant:

**MIND-small (default):**
```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_1_data_pipeline_mind_small.ipynb
.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_2_causal_modeling.ipynb
# ... phases 3, 4, 5
```

**MIND-large (edit `config.yaml`: `dataset: "large"`, `max_behavior_rows: null`):**
```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_1_data_pipeline_mind_large.ipynb
.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_2_causal_modeling_mind_large.ipynb
# ... phases 3, 4, 5 (mind_large variants)
```

Or open interactively:

```bash
.venv\Scripts\jupyter notebook
```

### Run Tests

```bash
.venv\Scripts\python -m pytest tests/ -v       # all 140 tests
.venv\Scripts\python -m pytest tests/ --cov=src  # with coverage
```

## Key Results

After min-max CDI normalization (2026-06-11):

| Method | NDCG@10 | Precision@10 | ILD | n |
|--------|:-------:|:------------:|:---:|:-:|
| **PPO (Causal-RL)** | **0.0974** ± 0.230 | 0.0204 ± 0.042 | **0.9589** ± 0.015 | 750 |
| Random | 0.0786 ± 0.183 | 0.0204 ± 0.041 | 0.9587 ± 0.015 | 750 |
| Popularity | **0.2967** ± 0.318 | **0.0657** ± 0.064 | 0.9522 ± 0.016 | 750 |

| Comparison | NDCG | Precision | ILD |
|------------|:----:|:---------:|:---:|
| **PPO vs Random** | **p=0.017** ✅, d=+0.082 | p=1.00, d=0.000 | p=0.73, d=+0.016 |
| PPO vs Popularity | p<0.0001, d=−0.865 | p<0.0001, d=−1.081 | p<0.0001, d=+0.444 |

> [!TIP]
> With min-max CDI normalization, PPO now significantly outperforms Random on NDCG (p=0.017, +23.9%). Precision matches Random (clicks are sparse). ILD is the highest of all methods. Popularity still dominates relevance metrics. See `docs/RESEARCH_LOG.md` for full history.

## Technology Stack

| Category | Version | Tools |
|----------|---------|-------|
| **Runtime** | 3.14.5 | CPython, pip 25+ |
| **Core** | 2.6.0+cu124 / 3.0.3 / 2.4.6 / 1.9.0 | PyTorch, Pandas, NumPy, scikit-learn |
| **Causal** | 0.14 / 3.6.1 | DoWhy, NetworkX |
| **RL** | 2.8.0 / 1.2.3 | stable-baselines3, Gymnasium |
| **NLP** | 5.5.1 | Sentence-Transformers (all-mpnet-base-v2) |
| **Config** | 6.0.3 | PyYAML (config.yaml loader) |
| **GPU** | optional | cuML, cupy (auto-fallback to CPU) |
| **Data** | — | MIND-small / MIND-large, PyArrow/Parquet |
| **Test** | 9.0.3 / 7.1.0 | pytest, pytest-cov (140 tests) |

## Research Log

See [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) for a chronological record of runs, findings, decisions, and metrics.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details (if not present, standard MIT terms apply).

## Acknowledgements

- Microsoft for the [MIND dataset](https://msnews.github.io/)
- PyWhy team for [DoWhy](https://github.com/py-why/dowhy)
- The [stable-baselines3](https://stable-baselines3.readthedocs.io) team
- [Sentence-Transformers](https://www.sbert.net) library

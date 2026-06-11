<div align="center">

# causal_rs

**Causal Recommendation System** — A research pipeline combining structural causal models, counterfactual reasoning, and reinforcement learning for diversity-aware news recommendation.

[![Python 3.13](https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.14-7b2ff7?style=flat-square)](https://github.com/py-why/dowhy)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.8-green?style=flat-square)](https://stable-baselines3.readthedocs.io)
[![MIND](https://img.shields.io/badge/Dataset-MIND--small-ff6f00?style=flat-square)](https://msnews.github.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#license)
[![Status](https://img.shields.io/badge/Status-Research-important?style=flat-square)]()

</div>

**causal_rs** explores whether counterfactual diversity scores (CDI) can guide a reinforcement learning policy to recommend content that is both relevant and diverse. The pipeline processes Microsoft News (MIND) data through five sequential phases — from raw TSV files to a trained PPO policy with offline evaluation.

> [!NOTE]
> This is a **research project**, not a production system. Results show that causal diversity signals improve intra-list diversity compared to popularity baselines, but relevance gains over random are not yet statistically significant with current training budget.

## Features

- **Structural Causal Model** — Builds a NetworkX causal DAG with user PCA features, item-level semantic embeddings (SBERT title + Wikidata entity), and categorical attributes; identifies causal effect of exposure on diversity (ATE ≈ −0.01)
- **Counterfactual Diversity Impact (CDI)** — Fits a DoWhy GCM (StructuralCausalModel) and queries per-item `E[Y_diversity | do(A=1, item_features)]` to compute the causal diversity impact of showing each candidate article
- **Automatic GPU acceleration** — Transparent cupy/numpy fallback for all batch operations (cosine diversity, L2 normalization, PCA); sentence-transformers on GPU; chunked processing to stay within 4 GiB VRAM
- **PPO Reinforcement Learning** — Custom Gymnasium environment with observation = user history embedding + CDI, action = pick K candidates, reward = `w * R_click + (1-w) * CDI`; trained with stable-baselines3
- **Offline Evaluation** — NDCG@K, Precision@K, Intra-List Diversity (ILD), paired t-test and Cohen's d vs Random and Popularity baselines

## Architecture

```text
MIND-small TSV            pipeline/streaming.py         gcm_fit.py
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
| 1 | `phase_1_data_pipeline_mind_small` | `src/data_pipeline/` | Parse MIND-small TSV → compute news embeddings (SBERT title + Wikidata entity) → build session features → construct SCM rows with Y_diversity → PCA reduction (U_pca, I_entity_pca, I_title_pca) → train/val/test split | `scm_{train,val,test}.parquet`, `phase1_report.json` |
| 2 | `phase_2_causal_modeling` | `src/causal_model/` | DoWhy `CausalModel` → backdoor identification → IPW + linear ATE estimation (−0.01) → refutation tests (placebo, subset, random common cause) | ATE ≈ −0.01, refutation pass |
| 3 | `phase_3_counterfactual_gcm` | `src/counterfactual/` | Build NetworkX causal DAG → auto-assign GCM mechanisms → `gcm.fit()` → per-item CDI via `predict_diversity_counterfactual()` → cache | `gcm_model.pkl`, `cdi_cache.pkl` |
| 4 | `phase_4_ppo_training` | `src/rl_agent/` | `NewsRecommendEnv` (Gymnasium) → PPO training (200K steps, 2 envs) → save checkpoint | `ppo_causal_rs_w03.zip` |
| 5 | `phase_5_evaluation` | `src/evaluation/` | Replay evaluation on 750 test sessions → NDCG, Precision, ILD → significance tests vs Random, Popularity | Aggregated metrics table |

## Project Structure

```
causal_rs/
├── src/                        # Python package (5 subpackages)
│   ├── causal_model/           # DoWhy causal inference (ATE, refutation, CDI batch)
│   ├── counterfactual/         # GCM graph, fit, counterfactual query, CDI cache
│   ├── data_pipeline/          # MIND I/O, parsing, features, SCM builder, streaming
│   ├── evaluation/             # NDCG, Precision, ILD, significance testing
│   ├── rl_agent/               # Gymnasium environment, PPO training
│   ├── config.py               # Path constants and pipeline configuration
│   └── gpu_utils.py            # GPU-accelerated batch ops with cupy/numpy fallback
├── notebooks/                  # 5 phase notebooks (execution entry points)
├── tests/                      # pytest test suite (140 tests, 12 test files)
├── data/                       # Raw MIND-small, interim, processed, SCM parquet parts
├── artifacts/                  # Trained models, CDI cache, PPO checkpoints, TB logs
└── docs/                       # Research log, codebase docs, CausalRS reference
```

## Getting Started

### Prerequisites

- **Python 3.13+**
- **NVIDIA GPU** (optional, automatic fallback to CPU)
- **MIND-small dataset** — download from [MIND News Dataset](https://msnews.github.io/)

### Installation

```bash
# Clone
git clone https://github.com/your-username/causal_rs.git
cd causal_rs

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
.venv\Scripts\pip install -r requirements.txt
```

### Run the Pipeline

Each phase is a Jupyter notebook. Execute sequentially:

```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_1_data_pipeline_mind_small.ipynb

.venv\Scripts\python -m jupyter nbconvert --to notebook \
    --execute notebooks/phase_2_causal_modeling.ipynb

# ... phases 3, 4, 5
```

Or open the notebooks interactively:

```bash
.venv\Scripts\jupyter notebook
```

### Run Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

## Key Results

| Metric | PPO (Causal-RL) | Random | Popularity |
|--------|:-:|:-:|:-:|
| NDCG@10 | 0.0875 ± 0.203 | 0.0796 ± 0.178 | **0.2967** ± 0.318 |
| Precision@10 | 0.0213 ± 0.043 | 0.0205 ± 0.040 | **0.0657** ± 0.064 |
| ILD | **0.9586** ± 0.014 | 0.9576 ± 0.015 | 0.9522 ± 0.016 |

> [!TIP]
> PPO consitently outperforms Random on all metrics (NDCG +10%, Precision +4%, ILD +0.0010) but differences are not yet statistically significant (p > 0.05). Popularity dominates relevance. PPO achieves higher ILD than Popularity (d = 0.44, medium effect). See `docs/RESEARCH_LOG.md` for full history.

## Technology Stack

| Category | Tools |
|----------|-------|
| **Core** | Python 3.13, PyTorch 2.6, NumPy, Pandas, scikit-learn |
| **Causal** | DoWhy 0.14, NetworkX 3.4 |
| **RL** | stable-baselines3 2.8, Gymnasium 1.2 |
| **NLP** | Sentence-Transformers 5.5 (all-mpnet-base-v2), Transformers |
| **GPU** | cuML (optional), cupy (optional, auto-fallback) |
| **Data** | MIND-small, PyArrow/Parquet |
| **Test** | pytest 8.1 (140 tests) |

## Research Log

See [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) for a chronological record of runs, findings, decisions, and metrics.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details (if not present, standard MIT terms apply).

## Acknowledgements

- Microsoft for the [MIND dataset](https://msnews.github.io/)
- PyWhy team for [DoWhy](https://github.com/py-why/dowhy)
- The [stable-baselines3](https://stable-baselines3.readthedocs.io) team
- [Sentence-Transformers](https://www.sbert.net) library

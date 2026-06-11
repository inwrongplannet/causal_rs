# Architecture

## Core Sections (Required)

### 1) Architectural Style

- **Primary style**: **Pipeline** (sequential notebook phases) with **functional-modular** internals (each `src/` package exposes stateless functions operating on DataFrames and models).
- **Why this classification**: The system is organized as a 5-phase sequential pipeline (Data → Causal Model → Counterfactual → RL → Evaluation), not a long-running service. Each phase is a Jupyter notebook that imports functions from a corresponding `src/` package. No event loops, request handlers, or dependency injection containers exist.
- **Primary constraints**: (1) MIND-small dataset size (89K behavior rows → 940K SCM records), (2) limited GPU VRAM (4 GiB RTX 3050) requiring chunked batch ops, (3) DoWhy GCM API constraints (parent order, need for low-level `evaluate()` calls after DoWhy 0.14 API changes).

### 2) System Flow

```text
MIND-small TSV files
    │
    ▼
[Phase 1: Data Pipeline]          src/data_pipeline/
    • parse_entities(), parse_history(), parse_impressions()
    • compute_news_features() → I_title_emb_full, I_entity_emb_full, I_sentiment, I_category
    • build_session_user_features() → U_history_emb_full, U_dwell_mean
    • build_scm_dataframe() → one row per (impression, candidate, click), Y_diversity
    • reduce_embedding_columns() → PCA (U_pca_* 32d, I_entity_pca_* 32d)
    • split_by_impression_id() → train/val/test + phase1_report.json
    │
    ▼
[Phase 2: Causal Modeling]        src/causal_model/
    • build_causal_graph_gml() → GML string
    • create_causal_model() → DoWhy CausalModel
    • identify_effect() → backdoor estimand
    • estimate_ate_ipw() / estimate_ate_linear() → ATE ≈ −0.01
    • run_refutations() → placebo/subset/random-common-cause
    │
    ▼
[Phase 3: Counterfactual GCM]     src/counterfactual/
    • build_causal_graph() → NetworkX DiGraph (U_pca_* + I_entity_pca_* as parents of outcomes)
    • auto.assign_causal_mechanisms() → empirical distributions & additive noise models
    • gcm.fit() → fitted StructuralCausalModel
    • predict_diversity_counterfactual() → E[Y_diversity | do(A=1, I_category=x, I_sentiment=y, I_entity_pca=z)]
    • precompute_cdi_cache() → {(user_id, item_id): CDI} for all session × candidate pairs
    │
    ▼
[Phase 4: PPO Training]           src/rl_agent/
    • NewsRecommendEnv (Gymnasium) — state = history_emb + CDI, action = pick candidate, reward = w·click + (1−w)·CDI
    • train_ppo() → stable-baselines3 PPO, saves checkpoint
    │
    ▼
[Phase 5: Evaluation]             src/evaluation/
    • replay_evaluate() → deterministically evaluate PPO on test sessions
    • ndcg_at_k(), precision_at_k(), ild() → per-session metrics
    • aggregate_metrics() → mean/std per method
    • significance_test() → paired t-test vs baselines (Random, Popularity)
```

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `src/data_pipeline/` | Raw data loading (TSV→DF), entity parsing, sentiment, embedding inference (SBERT), feature engineering, SCM record building, streaming chunk processing, PCA reduction, train/val/test split, quality checks | Causal inference, counterfactual reasoning, RL training | `scm_builder.py` builds the SCM DataFrame; `streaming.py` does chunked processing; `io_utils.py` loads MIND |
| `src/causal_model/` | DoWhy model creation, backdoor identification, IPW/linear ATE estimation, refutation tests (placebo, subset, random common cause) | Data loading, GCM fitting, RL environment | `model.py` wraps DoWhy; `refutation.py` runs 3 refuters |
| `src/counterfactual/` | NetworkX causal DAG, gcm.StructuralCausalModel fitting, counterfactual query, CDI precomputation cache | DoWhy refutations, data pipeline, PPO training | `gcm_fit.py`, `queries.py`, `precompute_cdi.py` |
| `src/rl_agent/` | Gymnasium environment (state/action/reward), PPO agent training with SB3 | Causal inference, evaluation metrics, data loading | `environment.py`, `train_ppo.py` |
| `src/evaluation/` | NDCG, Precision, ILD, replay evaluation, aggregate metrics, significance testing | RL training, causal graph construction, data pipeline | `metrics.py` |
| `src/gpu_utils.py` | GPU-accelerated batch ops (L2 normalize, cosine sim, cosine diversity), cupy/numpy fallback, chunked processing | Any domain logic | `gpu_utils.py` — pure numerical ops |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| GPU/CPU transparent fallback | `src/gpu_utils.py`, `src/data_pipeline/embedder.py`, `src/data_pipeline/scm_builder.py` (_try_import_pca) | MIND-small runs on laptops with or without NVIDIA GPU; no code changes needed to toggle |
| Chunked batch processing | `src/gpu_utils.py` (GPU_BATCH_SIZE=4096), `src/data_pipeline/streaming.py` (pd.read_csv chunksize) | Memory constraint: 4 GiB GPU VRAM, limited system RAM |
| Module-level try/except for optional deps | `src/gpu_utils.py` (cupy import), `src/data_pipeline/scm_builder.py` (cuml import), `src/data_pipeline/embedder.py` (sentence-transformers fallback) | No hard dependency on GPU libraries; graceful degradation |
| Lazy parallel environment | `src/rl_agent/train_ppo.py` (SubprocVecEnv on Unix, DummyVecEnv on Windows) | Windows does not support fork-based multiprocessing for large session objects |
| Data frame as cross-module contract | All phases use `pd.DataFrame` for inter-module data exchange | Standard Python data science pattern; enables notebook-inspectable outputs |

### 5) Known Architectural Risks

1. **CDI lacks per-item discrimination** — The GCM counterfactual query produces CDI scores that vary by only ~0.01–0.02 within a session (all items ≈ 0.88–0.90). **Partially mitigated 2026-06-11**: min-max normalization in `NewsRecommendEnv._min_max_cdi()` amplifies the tiny range to [0,1] per step. PPO now significantly beats Random on NDCG (p=0.017). The root cause (raw CDI's narrow range) persists — normalization is a workaround.
2. **Single commit — no iterative history** — The repository has only 1 commit (`5e0ba82`), so no change tracking, blame, or rollback capability exists.
3. **Config as global mutable state** — `src/config.py` still exports module-level constants imported by consumers. This was partially mitigated by the YAML loader (CLI/env/YAML overrides), but the module-level constants remain global and cannot be easily swapped per experiment without `reload()`.

### 6) Evidence

- `src/data_pipeline/scm_builder.py` — SCM DataFrame construction (the central data contract)
- `src/counterfactual/gcm_fit.py` — GCM graph and fit (counterfactual engine)
- `src/rl_agent/environment.py` — RL environment (reward = w·click + (1−w)·CDI)
- `src/evaluation/metrics.py` — Evaluation and significance testing
- `src/gpu_utils.py` — Chunked batch GPU ops

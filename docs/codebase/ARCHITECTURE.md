# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: Pipeline (sequential 5-phase data flow) with layered module boundaries
- Why this classification: Each phase reads the output of the previous phase via parquet files. The pipeline progresses from raw TSV data → causal SCM → GCM counterfactual → RL training → evaluation. Within each phase, modules are organized by layer (data → model → evaluation).
- Primary constraints:
  1. Single-machine execution (no distributed processing)
  2. GPU-optional design with transparent cupy/numpy fallback
  3. Memory-bounded chunked processing for 4 GiB VRAM

### 2) System Flow

```text
Phase 1 (Data Pipeline):
  MIND TSV → parse_impressions/parse_history → compute_news_features (SBERT title + entity)
  → build_session_user_features → build_scm_dataframe (batch Y_diversity via GPU)
  → reduce_embedding_columns (PCA) → split_by_impression_id → save SCM parquet

Phase 2 (Causal Modeling):
  SCM parquet → DoWhy CausalModel → identify_effect → estimate_ate_ipw/linear
  → run_refutations (placebo, subset, random common cause)

Phase 3 (Counterfactual GCM):
  SCM train parquet → build_causal_graph (NetworkX) → fit_gcm (auto.assign_causal_mechanisms)
  → precompute_cdi_cache (per-user, per-item counterfactual queries) → save pickles

Phase 4 (PPO Training):
  SCM train + CDI cache → NewsRecommendEnv (Gymnasium) → train_ppo (SB3)
  → save checkpoint

Phase 5 (Evaluation):
  SCM test + CDI cache + PPO model → replay_evaluate → ndcg_at_k/precision_at_k/ild
  → significance_test (paired t-test, Cohen's d)
```

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `data_pipeline` | TSV parsing, feature extraction, SCM DataFrame construction, streaming chunk processing | Causal inference, RL training | `src/data_pipeline/scm_builder.py`, `src/data_pipeline/features.py` |
| `causal_model` | DoWhy model creation, ATE estimation, refutation | Data I/O, GCM fitting | `src/causal_model/model.py`, `src/causal_model/refutation.py` |
| `counterfactual` | GCM graph, fitting, counterfactual queries, CDI cache | ATE estimation, PPO | `src/counterfactual/gcm_fit.py`, `src/counterfactual/queries.py` |
| `rl_agent` | Gymnasium env, PPO training, reward shaping | Evaluation metrics, causal inference | `src/rl_agent/environment.py`, `src/rl_agent/train_ppo.py` |
| `evaluation` | Metrics computation, significance tests | Training, data pipeline | `src/evaluation/metrics.py` |
| `gpu_utils` | GPU-accelerated batch ops with CPU fallback | Domain-specific logic | `src/gpu_utils.py` |
| `config` | Config loading (YAML + env + CLI + defaults) | Business logic | `src/config.py` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| GPU/CPU transparent fallback | `src/gpu_utils.py:60-76` (chunked batch ops), `src/data_pipeline/embedder.py:22-62` (sentence-transformers → HashingVectorizer) | Single codebase runs on GPU or CPU without changes |
| Chunked batch processing | `src/gpu_utils.py:135-157` (GPU_BATCH_SIZE=4096), `src/data_pipeline/streaming.py:71-121` (chunksize=2000) | Stays within 4 GiB VRAM; no OOM |
| Factory function | `src/rl_agent/train_ppo.py:11-25` (`make_env`), `src/data_pipeline/embedder.py:22-62` (`build_title_encoder`) | Creates configured instances lazily |
| Singleton config | `src/config.py:176` (`_config` at module level) | Single config state accessible via imports |
| Pickle persistence | `src/counterfactual/precompute_cdi.py:73-77`, `src/counterfactual/precompute_cdi.py:82-92` | Serialize fitted GCM and CDI cache between phases |

### 5) Known Architectural Risks

1. **GCM ↔ PPO coupling via CDI**: CDI is precomputed and cached as a static lookup. If the GCM is refitted, the entire CDI cache must be regenerated (~73 min for 656K entries). No cache-invalidation mechanism exists.
2. **Single-machine scaling**: The pipeline is designed for a single workstation. No distributed processing, no sharding. Full MIND-large (65M rows after SCM expansion) would exceed available memory/VRAM.
3. **Sequential phase dependency**: Phases 2-5 depend on phase 1 output on disk. No explicit dependency graph or automated pipeline orchestration — must be run in order manually.
4. **Notebook state coupling**: Phases 2-5 notebooks depend on output parquets from previous phases. No formal contract or schema validation beyond `run_quality_checks()`.

### 6) Evidence

- `README.md` (architecture diagram, lines 32-48)
- `src/data_pipeline/scm_builder.py` (SCM building flow)
- `src/counterfactual/gcm_fit.py` (GCM graph + fitting)
- `src/rl_agent/environment.py` (RL env)
- `src/evaluation/metrics.py` (evaluation flow)
- `notebooks/` (5 sequential notebooks)

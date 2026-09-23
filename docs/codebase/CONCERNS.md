# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| High | **CDI lacks per-item discrimination within sessions** — ~0.0005 variance across items means PPO reward is nearly constant | `docs/RESEARCH_LOG.md:712-713`, `src/counterfactual/queries.py:30-97` | PPO cannot learn meaningful ranking; performs ≈ Random on relevance | [ASK USER] Intended fix direction — min-max normalization (implemented), GCM mechanism tuning, or alternative diversity signal |
| ~~High~~ Resolved | ~~No CI/CD pipeline~~ — **Resolved**: `.github/workflows/ci.yml` runs lint + tests + coverage floor (70%) on every push/PR | `.github/workflows/ci.yml` | N/A | N/A |
| High | **`src/causal_model/` (4 modules) has zero test coverage** | `docs/RESEARCH_LOG.md:193` | Regressions in ATE estimation or refutation go undetected | Add unit tests for model creation, ATE estimation, refutation |
| Medium | **No schema validation across pipeline phases** — Phase 2-5 assume parquet column names from Phase 1 | `src/data_pipeline/scm_builder.py:254-313` (quality checks only within Phase 1) | Silent failures if column names change | Define formal column contract for SCM parquet files |
| Medium | **Arrow MemoryError risk** — full MIND-large parquet loading can exceed 2 GB contiguous allocation | `docs/RESEARCH_LOG.md:467-469` | Pipeline crashes on MIND-large load | Fragment loading implemented; verify on full dataset |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| `src/config.py` global module-level constants | Legacy pattern before YAML/CLI config was added | `src/config.py:181-207` | Cyclic import risk; config not refreshable at runtime cleanly | [ASK USER] Replace with dependency-injected config object |
| `scripts/run_phase4_large.py` duplicates session-building logic | Copied from notebook with minor param changes | `scripts/run_phase4_large.py:27-49` | Session-building logic drifts from notebook code | Extract shared session builder into `src/` |
| `eval_only.py` duplicates evaluation logic | Created for quick iteration | `scripts/eval_only.py:55-105` | Evaluation bugs differ from `metrics.py` | Use `src/evaluation/metrics.py` exclusively |
| Pickle persistence for GCM and CDI | Quick serialization | `artifacts/` | Version skew; large file sizes (gcm_model_full.pkl ~200 MB) | Use joblib or model versioning |
| Windows SubprocVecEnv workaround | SubprocVecEnv fails on Windows with large pickle payloads | `src/rl_agent/train_ppo.py:30-32` | Single-process training only on Windows | Use shared memory or ray |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| `eval()` in old notebook versions | A03:2021-Injection | `docs/RESEARCH_LOG.md:105` (bug #5 — now fixed to `ast.literal_eval`) | All notebooks now use `ast.literal_eval()` | Verify no `eval()` remains in any notebook cell |
| No input validation on MIND TSV loads | N/A (local research project) | `src/data_pipeline/io_utils.py:226-254` | pandas reads with strict dtype | N/A for research context |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| CDI precomputation O(n*m) | `src/counterfactual/precompute_cdi.py:40-69` | 73.5 min for 656K pairs (149 pairs/s) | Full MIND-large would take hours | Parallelize per-session; vectorize across items |
| GCM fit O(rows * nodes) | `src/counterfactual/gcm_fit.py:74-117` | 108 s for 657K rows × 102 nodes | Doubling rows quadruples fit time | Sample training data; use incremental fitting |
| PPO training with large session objects | `src/rl_agent/train_ppo.py:28-32` | DummyVecEnv on Windows (single-process) | Cannot scale to full MIND-large training | Implement multi-GPU or ray-based parallelism |
| Parquet columnar embedding storage | `src/data_pipeline/scm_builder.py:280-287` | String-serialized embeddings inflate file size/load time | MIND-large parquet loads are slow | Store embeddings as float32 numpy arrays in parquet |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `src/data_pipeline/scm_builder.py` | Contains SCM building, PCA, split, quality checks — multiple concerns | 2 commits in last 5 (scan lines 306-327) | Refactor into focused modules; add unit tests |
| `src/data_pipeline/io_utils.py` | Handles dataset discovery across 3 sources (local, zip, env URL) | 3 commits (highest churn, scan line 306) | Add tests for each discovery strategy |
| `notebooks/` (all 5 phases) | Bug-fixed multiple times; `eval()` → `ast.literal_eval()`, import errors, MemoryError | 5 notebooks in top churn (scan lines 314-317) | Critical-path logic should be in `src/`, notebooks as thin orchestrators |

### 6) Resolved Decisions

The following intent-dependent questions have been answered by the team:

1. **CDI discrimination**: Accept current results — no fix planned.
2. **Config refactor**: Replace `src/config.py` global constants with dependency-injected config object.
3. **CI pipeline**: Not needed.
4. **Coverage threshold**: Add coverage tool with threshold.
5. **Session-builder extraction**: Keep as-is (notebooks/scripts).
6. **Windows SubprocVecEnv**: Current DummyVecEnv workaround is acceptable.

### 7) Evidence

- Scan output (TODO/FIXME: none found, lines 295-296)
- Scan output (high-churn files, lines 305-326)
- Scan output (no CI/CD, line 351; no testing config, line 360)
- `docs/RESEARCH_LOG.md` (bug reports, performance metrics)
- `src/data_pipeline/scm_builder.py` (quality checks)
- `src/counterfactual/queries.py` (CDI computation)

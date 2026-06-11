# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| High | **CDI scores lack per-item discrimination** — all items in a session get CDI ≈ 0.88–0.90, range < 0.02. This makes the PPO reward nearly constant, causing PPO ≈ Random in evaluation. | `src/counterfactual/queries.py`, `docs/RESEARCH_LOG.md` (2026-06-11 RL-Guided Hyperparameter Tuning), Phase 5 evaluation: PPO NDCG 0.0918 vs Random 0.0807 (p=0.17, n.s.) | Core research hypothesis (causal RL improves diversity) cannot be validated. Pipeline produces PPO policy that matches random. | Re-architect CDI: normalize within session, use different GCM intervention targets, or shape reward to amplify CDI differences. |
| High | **No git history** — repository has no commits. No change tracking, blame, or rollback. | `git log` returns no commits | Any experimental change is irreversible. Cannot identify when bugs were introduced. | Initialize git repo with initial commit. |
| Medium | **`src/causal_model/` has no tests** — graph.py, model.py, cdi.py, refutation.py are completely untested. | Scan shows no test files covering `src/causal_model/`. 140 total tests, none in this module. | Causal identification and ATE estimation could have silent bugs. Refutation logic could silently skip tests. | Add unit tests for GML graph generation, model creation, IPW estimation, refutation wrappers. |
| Medium | **`src/config.py` is global mutable state** — imported directly as a module constant, not injected. Cannot vary config per experiment without modifying source. | `src/config.py` is imported via `from src.config import *` in notebooks | Experiments with different PCA dims, seed, or NEG_RATIO require editing source code. Risk of accidental commits with stale config. | Extract experiment configs to YAML/JSON files or dataclass with CLI overrides. |
| Medium | **`requirements.txt` is a full pip freeze** (767 lines) — not a minimal dependency declaration. Packages like `2captcha-python`, `celery`, `boto3` are not used but listed. | `requirements.txt` scan shows 767 lines; `pip list` confirms many unused packages. | Bloated install, unclear which packages are truly required. Could hide version conflicts. | Generate minimal `requirements.in` (or `pyproject.toml`) with only direct dependencies. |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| `requirements.txt` is a full `pip freeze` output | Quick setup — `pip freeze > requirements.txt` from working venv | `requirements.txt` (767 lines) | Unclear dependency tree, extraneous packages, version conflicts when collaborating | Generate a proper `pyproject.toml` with only direct dependencies and pinned sub-dependencies |
| No `pyproject.toml` or `setup.py` | Project started as a notebook; never formalized as a package | Project root | Cannot `pip install -e .` to use `src` as an importable package from arbitrary directories | Create `pyproject.toml` with `[project]` and `[build-system]` sections |
| `src/causal_model/` has no tests | Notebook-first development; tests were added only for later modules | `tests/` directory lists no files matching `causal_model` | ATE estimation and refutation logic could have silent bugs | Add `test_graph.py`, `test_model.py`, `test_refutation.py` |
| No linter configuration | Project grew organically from notebooks | Project root | Inconsistent code style across contributors, no automated quality enforcement | Add `ruff` or `flake8` config to `pyproject.toml` |
| Static `docs/codebase/` docs | Generated from a one-time codebase scan | `docs/codebase/` | Will go stale as code changes; no CI check | Add a CI step that re-generates or validates docs against the current source |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Pickle deserialization of untrusted artifacts | A08 (Software and Data Integrity Failures) | `pickle.load(open('artifacts/cdi_cache.pkl', 'rb'))` in `src/counterfactual/precompute_cdi.py` | Artifacts are local files, not user-supplied | No integrity check (hash/signature) on loaded pickle files. If an attacker replaces `gcm_model.pkl` or `cdi_cache.pkl`, arbitrary code execution is possible. |
| Python installer in project root | N/A | `python-3.13.5-amd64.exe` was found in project root (now removed) | N/A — file has been cleaned up | Should not commit executables to version control. `.gitignore` already excludes it. |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Full MIND dataset (~1M training records) causes MemoryError | `src/config.py` sets `MAX_BEHAVIOR_ROWS = 5000` as a hard stopgap; `docs/RESEARCH_LOG.md` (2026-06-10) describes the workaround | Pipeline is capped at 5K behavior rows (~940K SCM records). Full MIND would produce millions of SCM records. | Streaming chunked pipeline exists (`streaming.py`) but streaming only covers Phase 1. Phase 3 GCM fit loads all data into memory. | Extend chunked processing to GPU batch ops and GCM fitting. Add incremental CDI computation. |
| CDI precomputation: 38 pairs/s | `docs/RESEARCH_LOG.md`: 500 pairs in 2.7 s = ~38 pairs/s | Full 3500×150 = 525K pairs would take ~3.8 h. | 525K pairs is an upper bound; realistic candidate pools are smaller, but still hours. | Batch CDI computation (vectorize the GCM evaluate call over candidates) to reduce from O(pairs) to O(sessions). |
| PPO training throughput ~204 steps/s (CPU) | `docs/RESEARCH_LOG.md` (2026-06-11 Phase 3 & 4) | 200K timesteps in ~16 min. | Training a production-quality policy would need 5M+ timesteps → 7+ hours on CPU. | GPU PPO policy (CNN/Transformer encoder) or use stable-baselines3's MlpPolicy on GPU. |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `src/counterfactual/` | Heavily dependent on DoWhy 0.14 internals (`PARENTS_DURING_FIT`, `AdditiveNoiseModel.evaluate`, auto-encoder assignment). Upstream DoWhy API changes could silently break. | 3 significant refactors in last 3 development sessions to work around API changes (counterfactual_samples removed, parent order fix, auto encoder changes) | Pin DoWhy to 0.14; add integration tests that verify GCM fit + CDI query produce valid outputs; wrap DoWhy access behind a thin adapter that can be swapped if API changes. |
| `src/gpu_utils.py` | Chunked batch ops with dual cupy/numpy paths triple the code surface. GPU backend detection is fragile (probes CUDA_PATH, PyTorch cuda, cupy import). | Implemented in a single session with 17 tests; has been refactored once for chunking | Keep test coverage high (17 tests); add a `GPU_ENABLED=False` CI run to ensure CPU path is always exercised. |
| `src/data_pipeline/io_utils.py` | Multiple fallback strategies for data loading (local files, extracted zips, env URL). Zip extraction can hang. | Caused pipeline halts in 2 earlier runs (zip extraction hang, ArrowMemoryError) | Add explicit timeout for zip extraction; add CRC validation on extracted files; prefer pre-extracted directories over runtime extraction. |

### 6) `[ASK USER]` Questions

1. [ASK USER] The CDI scores lack per-item discrimination (all items in a session get CDI ≈ 0.88–0.90). Is the intended fix: (a) normalize CDI within session, (b) add more item features to the GCM graph (e.g., I_title_pca), (c) change the counterfactual intervention target, or (d) accept PPO ≈ Random as the current baseline?
2. [ASK USER] Should `src/config.py` be replaced with a YAML/JSON config file or CLI argument parser to support experiment tracking?
3. [ASK USER] Do you want automated CI (GitHub Actions) for running tests on push, or is the project intended as a local-only research tool?
4. [ASK USER] Should a coverage tool (pytest-cov) be added and a minimum threshold enforced?

### 7) Evidence

- `src/counterfactual/queries.py` — CDI computation logic (root cause of the discriminative-power gap)
- `docs/RESEARCH_LOG.md` (2026-06-11 — Full Pipeline, RL-Guided Hyperparameter Tuning) — metrics and diagnosis
- `src/config.py` — global config module (mutable state concern)
- `tests/` directory listing — missing `test_causal_model*` files
- `requirements.txt` — 767-line full freeze

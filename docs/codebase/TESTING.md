# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- **Primary test framework**: pytest 9.0.3
- **Assertion/mocking tools**: Built-in `assert` statements. No mocking library (unittest.mock, pytest-mock, etc.) found. `pytest-cov==7.1.0` is installed but not configured (no `.coveragerc` or `pyproject.toml` coverage section).
- **Commands**:
```bash
.venv\Scripts\python -m pytest tests/ -v
.venv\Scripts\python -m pytest tests/test_counterfactual.py -v
.venv\Scripts\python -m pytest tests/ -v -k "test_build"
.venv\Scripts\python -m pytest tests/ --cov=src --cov-report=term
```

### 2) Test Layout

- **Test file placement pattern**: All tests in dedicated `tests/` directory (not co-located with source).
- **Naming convention**: `test_<module_name>.py` matches source module name. Test classes use `Test<PascalCaseName>`; test methods use `test_<descriptive_name>`.
- **Setup files and where they run**: `tests/conftest.py` inserts the project root into `sys.path` at import time so tests can `import src.*` directly. No pytest fixtures defined.

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | Yes | Individual functions: graph construction, category encoding, nlp utils, GPU utils, IO utils, parsers, feature engineering, SCM builder, environment, evaluation metrics | All 12 test files are pure unit tests — they construct small inputs and assert on structure/dimensions/values |
| Integration | No | [TODO] — no tests that load real MIND data, fit a GCM, or train a PPO agent | The pipeline notebooks serve as integration smoke tests; no automated integration test suite exists |
| E2E | No | [TODO] — no full-pipeline test | Pipeline must be run manually through 5 notebooks |

### 4) Mocking and Isolation Strategy

- **Main mocking approach**: **No mocking**. Tests construct minimal synthetic DataFrames and graphs directly in test code. For example, `test_counterfactual.py` builds `nx.DiGraph` instances manually. `test_gpu_utils.py` uses small numpy arrays. `test_environment.py` creates a minimal `NewsRecommendEnv` with hardcoded embeddings.
- **Isolation guarantees**: No state shared between tests. No fixtures — each test sets up its own data.
- **Common failure mode in tests**: `PerformanceWarning` for DataFrame fragmentation (benign) and `RuntimeWarning: Precision loss` in significance test for near-identical arrays (benign, handled).

### 5) Coverage and Quality Signals

- **Coverage tool + threshold**: `pytest-cov==7.1.0` installed but not configured (no `.coveragerc` or `[tool.coverage]` section in project root). Running with `--cov` would work ad-hoc.
- **Current reported coverage**: [TODO] — coverage has never been measured.
- **Known gaps/flaky areas**: 
  - `src/causal_model/` (graph.py, model.py, cdi.py, refutation.py) has **zero test coverage**.
  - No tests for `predict_diversity_counterfactual()`, `fit_gcm()`, `precompute_cdi_cache()`.
  - No tests for end-to-end SCM fitting or PPO training.

### 6) Evidence

- `tests/conftest.py` — sys.path setup
- `tests/test_counterfactual.py` — representative unit test (12 tests, 2 classes)
- `tests/test_gpu_utils.py` — 17 tests for batch ops with both GPU and CPU paths
- `tests/test_environment.py` — 7 tests for `NewsRecommendEnv`
- `tests/test_evaluation.py` — 21 tests for metrics
- `pytest-cov==7.1.0` in `.venv` — coverage tool available (not configured)

# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | snake_case | `scm_builder.py`, `gcm_fit.py`, `train_ppo.py` | Any file in `src/` |
| Functions/methods | snake_case | `build_scm_dataframe()`, `precompute_cdi_cache()`, `sample_negative_items()` | `src/data_pipeline/scm_builder.py:28`, `src/counterfactual/precompute_cdi.py:14`, `src/data_pipeline/features.py:95` |
| Classes | PascalCase | `NewsRecommendEnv`, `TestSampleNegativeItems` | `src/rl_agent/environment.py:87`, `tests/test_features.py:13` |
| Constants/env vars | UPPER_CASE | `DATA_DIR`, `SEED`, `GPU_BATCH_SIZE`, `CAUSAL_RS_*` | `src/config.py:181-207`, `src/data_pipeline/io_utils.py:98-101` |
| Private methods/fields | `_` prefix | `_obs()`, `_min_max_cdi()`, `_to_array()`, `_cp`, `_DEFAULTS` | `src/rl_agent/environment.py:135,181`, `src/gpu_utils.py:49,60`, `src/config.py:31` |
| Test classes | `Test` prefix | `TestSampleNegativeItems`, `TestNDCG` | All files in `tests/` |
| Test methods | snake_case | `test_basic_sampling()`, `test_saves_and_loads()` | All test files |

### 2) Formatting and Linting

- Formatter: black 26.1.0 (installed in `requirements.txt`, no config file found)
- Linter: None configured (no `.eslintrc`, `.flake8`, `pyproject.toml` linting config detected)
- Most relevant enforced rules: None — no linting or formatting CI check
- Run commands: None documented

### 3) Import and Module Conventions

- Import grouping/order: Standard library → third-party → local `src` imports (observed in all source files)
- Alias vs relative import policy: Absolute imports only, using `src.` prefix (e.g., `from src.data_pipeline.parsers import parse_history`). No relative imports observed.
- Public exports/barrel policy: No `__all__` definitions found. Each `__init__.py` is empty.

### 4) Error and Logging Conventions

- Error strategy by layer:
  - Data pipeline: Assertions and explicit raises (`raise ValueError`, `raise AssertionError`, `raise RuntimeError`)
  - Causal model: Exceptions caught and printed inline in notebook cells
  - Counterfactual: `log.warning()` on individual failures, continues processing
  - RL agent: GPU fallback catches exceptions and logs via `logger.warning()`
- Logging style: Module-level `logger = logging.getLogger(__name__)` pattern. Used with structured messages indicating operation, context, and error reason.
- Sensitive-data redaction rules: No secrets handling; no redaction logic found.

### 5) Testing Conventions

- Test file naming/location rule: `tests/test_<module_name>.py` (separate `tests/` directory, not co-located)
- Mocking strategy norm: Minimal mocking. Dependencies injected via function parameters or monkeypatch for import failures. No mock framework (unittest.mock) usage.
- Coverage expectation: No coverage threshold enforced. `pytest-cov` optional.

### 6) Evidence

- `src/data_pipeline/parsers.py` (snake_case functions, clear error handling)
- `src/gpu_utils.py` (logging pattern, exception handling, `_` private helpers)
- `tests/test_features.py` (test class/method naming)
- `README.md` (test commands, lines 172-175)
- Scan output (no linting config detected, line 290)

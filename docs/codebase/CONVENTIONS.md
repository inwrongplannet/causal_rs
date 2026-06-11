# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | `snake_case.py` | `gcm_fit.py`, `precompute_cdi.py`, `scm_builder.py` | All files in `src/` |
| Functions/methods | `snake_case` | `build_causal_graph()`, `precompute_cdi_cache()`, `train_ppo()` | All `.py` files in `src/` |
| Types/interfaces | Python has no explicit type aliases; functions use type hints (PEP 484) for list/dict/optional params | `def fit_gcm(df_train: pd.DataFrame, pca_columns: list, entity_pca_columns: list = None) -> gcm.StructuralCausalModel:` | `src/counterfactual/gcm_fit.py`, most `src/` functions |
| Constants/env vars | `UPPER_SNAKE_CASE` in `src/config.py` | `GPU_DEVICE`, `GPU_BATCH_SIZE`, `PCA_COMPONENTS`, `NEG_RATIO` | `src/config.py` |

### 2) Formatting and Linting

- **Formatter**: `black` (version 26.1.0) — installed as dev dependency, no config file found in project root
- **Linter**: No linter config found (no `.flake8`, `pyproject.toml`, `.pylintrc`, or `ruff` config)
- **Most relevant enforced rules**: Black defaults (line length 88, consistent quotes, trailing commas). No custom rules found.
- **Run commands**:
  ```bash
  .venv\Scripts\python -m black src/ tests/
  .venv\Scripts\python -m pytest tests/ -v
  ```

### 3) Import and Module Conventions

- **Import grouping/order**: Standard Python (stdlib → third-party → local). No explicit grouping rules or isort config found.
- **Alias vs relative import policy**: All internal imports use **absolute** package imports from `src`. No relative imports (no `from .foo import bar`).
- **Public exports/barrel policy**: No `__all__` declarations found. Each `__init__.py` is empty.
- **Path aliasing**: None. No `sys.path` manipulation in production code. Only `tests/conftest.py` inserts the project root into `sys.path`.

### 4) Error and Logging Conventions

- **Error strategy by layer**:
  - Data pipeline: `AssertionError` for quality checks. Warnings for partial failures (embedding fallback, parse errors).
  - Causal model: DoWhy exceptions wrapped in `try/except` in refutations to avoid one-failure-blocks-all.
  - Counterfactual: `log.warning` and `continue` on individual (user, item) CDI failures.
  - RL agent: Uses SB3's internal PPO error handling; no custom exception wrappers.
- **Logging style**: Python `logging.getLogger(__name__)` at module level. Informational messages about pipeline progress, warnings about fallback/partial failures. No structured logging (no JSON log format).
- **Sensitive-data redaction rules**: None found. The project processes public MIND dataset with no PII handling.

### 5) Testing Conventions

- **Test file naming/location rule**: All tests in `tests/` directory, named `test_<module>.py` matching the source module name. E.g., `test_counterfactual.py` tests `src/counterfactual/`.
- **Mocking strategy norm**: No mocking framework used. Tests construct small DataFrames or NetworkX graphs with hardcoded values and assert on structure/dimensions. No external API mocking (because there are no external API calls in production code).
- **Coverage expectation**: No coverage tool configured. No `.coveragerc` or `[tool.coverage]` section found.

### 6) Evidence

- `src/counterfactual/gcm_fit.py` — representative function with type hints, logging, docstrings
- `src/data_pipeline/nlp_utils.py` — snake_case naming, try/except fallback at function level
- `src/config.py` — UPPER_SNAKE_CASE constants
- `tests/test_counterfactual.py` — representative test file with class-based test organization

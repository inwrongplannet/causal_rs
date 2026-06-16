# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary test framework: pytest 9.0.3 (`README.md:128`)
- Assertion/mocking tools: pytest `approx`, `raises`, `monkeypatch`. No separate mocking library.
- Commands:

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src
```

### 2) Test Layout

- Test file placement pattern: Separate `tests/` directory (not co-located with source)
- Naming convention: `test_<module_name>.py` (e.g., `test_features.py`, `test_scm_builder.py`)
- Setup files and where they run: `tests/conftest.py` adds project root to `sys.path`; no fixtures beyond `tmp_path` from pytest built-in.

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | Yes | Individual functions: parsers, metrics, features, env methods | 12 test files, ~140 tests |
| Integration | No | No end-to-end pipeline test | No test loads real MIND data |
| E2E | No | No full pipeline test | Notebooks serve as E2E verification |

### 4) Mocking and Isolation Strategy

- Main mocking approach: Dependency injection (passing callables like `title_encoder`), `monkeypatch` for import failures (e.g., `test_embedder.py:8` simulating missing sentence-transformers). No `unittest.mock` usage.
- Isolation guarantees: Each test creates its own data via fixtures. No shared state between tests.
- Common failure mode in tests: Tests that rely on GPU availability skip gracefully (`gpu_available()` returns bool). Empty/edge case matrix coverage is thorough.

### 5) Coverage and Quality Signals

- Coverage tool + threshold: pytest-cov optional (no threshold configured)
- Current reported coverage: [TODO] — no coverage report generated in scan
- Known gaps/flaky areas:
  - `src/causal_model/` (graph, model, refutation, cdi) — **untested** as of `docs/RESEARCH_LOG.md:193`
  - `src/counterfactual/` — only `gcm_fit.py` and `queries.py` partially tested (graph build, category_to_int)
  - `scripts/` — no test coverage
  - `notebooks/` — no automated test coverage (verified manually via notebook execution)

### 6) Evidence

- `tests/conftest.py`
- `tests/test_features.py`
- `tests/test_gpu_utils.py`
- `tests/test_parsers.py`
- `README.md:172-175`
- `docs/RESEARCH_LOG.md:193` (untested modules noted)

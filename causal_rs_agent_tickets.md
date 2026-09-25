# causal_rs Research-Ready Plan — Agent Ticket Sheet (Low-Ambiguity Edition)

This document is written to be executed by a coding agent with **no judgment calls left unspecified**. Every ticket gives exact file paths, exact code to find, exact code to write, and an exact command to verify success. Follow tickets **in numeric order**. Do not skip, reorder, or combine tickets.

---

## RULES FOR THE AGENT — READ FIRST

1. Work through tickets **in the exact order given** (0.1, 0.2, 0.3, ... 7.2). Do not start a later-phase ticket before every ticket in an earlier phase is done and verified.
2. For every ticket, use the **exact code given** in this document. Do not rewrite it, "improve" it, or change variable names, even if you think you have a better approach.
3. Every ticket has a "Verify" step with an exact command and an exact expected result. Run it. If the actual result does not match, **stop and report the exact error text** — do not attempt to guess a fix that isn't described in this document.
4. Only touch files explicitly named in a ticket. Do not delete, rename, or edit any other file.
5. After each ticket passes its Verify step, make one git commit with the message given in that ticket (`Commit message:` line).
6. When a ticket says "find the exact text block", search for it verbatim (character-for-character, including whitespace) in the named file. If you cannot find an exact match, stop and report — do not approximate.
7. When a ticket provides a full new file, create that file with **exactly** the given content — do not add, remove, or reformat anything.
8. Some tickets (marked **[REQUIRES MIND DATA]**) can only be *run* after the MIND dataset has been downloaded and Phases 1–3 of the existing pipeline have been executed (per the README). You can still create the files for these tickets, but you cannot verify them by running them until that data exists. Say so explicitly when you reach such a ticket instead of skipping it silently.
9. Never invent a person's name, email, or copyright holder. If a ticket needs one and it isn't given, insert the literal placeholder text shown (e.g. `[COPYRIGHT HOLDER NAME]`) and tell the user to fill it in — do not make one up.
10. If any package version given in this document fails to install, do not silently substitute a different version. Report the exact `pip` error, then use the fallback procedure described in that ticket (only Ticket 0.3 has a fallback procedure; every other ticket has fixed, verified version numbers with no fallback needed).

---

## TICKET INDEX

- **Phase 0 — Repo hygiene & broken dependency file:** 0.1, 0.2, 0.3, 0.4, 0.5
- **Phase 1 — Fix real bugs, close critical test gaps:** 1.1, 1.2, 1.3, 1.4, 1.5
- **Phase 2 — CI:** 2.1, 2.2
- **Phase 3 — Determinism/seeding:** 3.1, 3.2, 3.3, 3.4
- **Phase 4 — Statistical rigor infrastructure:** 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
- **Phase 5 — Real baseline (Logistic-CF):** 5.1, 5.2, 5.3, 5.4, 5.5
- **Phase 6 — Narrative decision rule + README rewrite:** 6.1, 6.2, 6.3
- **Phase 7 — Paper-support artifacts:** 7.1, 7.2

---

# PHASE 0 — Repo Hygiene & Broken Dependency File

### Ticket 0.1 — Delete the stray `aa.md` file

**Action:** Delete the file `aa.md` at the repository root (it contains only the text "hi" and serves no purpose).

**Verify:**
```bash
ls aa.md
```
Expected: `ls: cannot access 'aa.md': No such file or directory` (i.e. the command fails because the file is gone).

**Commit message:** `chore: remove stray aa.md file`

---

### Ticket 0.2 — Add a real LICENSE file

**Action:** The README displays an MIT license badge but no `LICENSE` file exists in the repository. Create a new file `LICENSE` at the repository root with exactly this content (replace `[COPYRIGHT HOLDER NAME]` with the name the user gives you — if you don't know it, leave the placeholder exactly as-is and tell the user to fill it in; do not invent a name):

```
MIT License

Copyright (c) 2026 [COPYRIGHT HOLDER NAME]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Verify:**
```bash
test -f LICENSE && echo "LICENSE exists"
```
Expected output: `LICENSE exists`

**Commit message:** `docs: add MIT LICENSE file (matches README badge claim)`

---

### Ticket 0.3 — Fix the broken `requirements.txt`

**Background (do not re-investigate this — it is already confirmed):** The current `requirements.txt` has two serious problems:
1. It is encoded as **UTF-16** (a byte-order-mark `FF FE` at the start of the file), which happens when `pip freeze > requirements.txt` is run in Windows PowerShell. Many tools mishandle UTF-16 requirements files.
2. It is **missing the project's most important dependencies entirely** — `torch`, `dowhy`, `stable-baselines3`, `gymnasium`, `sentence-transformers`, and `nltk` do not appear anywhere in the file, even though every phase of the pipeline requires them. A fresh `pip install -r requirements.txt` followed by running any notebook will fail with `ModuleNotFoundError`.

**Action:** Delete the existing `requirements.txt` and replace it with a new file (plain UTF-8 — do not use PowerShell `>` redirection to create it) containing exactly this content:

```
# Core dependencies for causal_rs.
#
# torch, dowhy, stable-baselines3, gymnasium, sentence-transformers, and nltk
# were MISSING ENTIRELY from the previous requirements.txt (it had been
# frozen from an environment that did not actually have these installed).
# Their version numbers below come from README.md's "Key packages (as
# tested)" table and MUST be confirmed by successfully completing the
# Verify step of this ticket before being trusted further.
torch==2.6.0
dowhy==0.14
stable-baselines3==2.8.0
gymnasium==1.2.3
sentence-transformers==5.5.1
nltk>=3.8,<4.0

# These versions were directly confirmed present in the previous
# requirements.txt's pip freeze and are trusted as-is.
numpy==1.24.4
pandas==2.3.2
scikit-learn==1.7.2
scipy==1.16.2
networkx==3.4.2
PyYAML==6.0.2
matplotlib==3.10.6
pyarrow==21.0.0
joblib==1.5.2
tqdm==4.67.1

# Needed for scripts/analyze_results.py (Phase 4 of the research-ready plan).
statsmodels>=0.14,<1.0

# Optional GPU acceleration. Install manually only if an NVIDIA GPU is
# available — all code in src/ auto-falls-back to CPU/numpy without these.
# cupy-cuda12x
# cuml
```

Also create a new file `requirements-dev.txt` (plain UTF-8) with exactly this content:
```
pytest==8.1.1
pytest-cov>=5.0,<6.0
ruff>=0.4,<1.0
```

**Verify (run all of these commands in order; every one must succeed):**
```bash
python3 -m venv /tmp/causal_rs_test_venv
/tmp/causal_rs_test_venv/bin/pip install --upgrade pip
/tmp/causal_rs_test_venv/bin/pip install -r requirements.txt
/tmp/causal_rs_test_venv/bin/pip install -r requirements-dev.txt
/tmp/causal_rs_test_venv/bin/python -c "import torch, dowhy, stable_baselines3, gymnasium, sentence_transformers, nltk, pandas, numpy, sklearn, scipy, networkx, yaml, matplotlib, pyarrow, statsmodels, pytest; print('ALL IMPORTS OK')"
```
Expected: the final line printed is `ALL IMPORTS OK`, and no command in the sequence exits with an error.

**Fallback procedure (only if a `pip install` step fails on a specific package/version):**
1. Read the exact pip error for that one package.
2. If it says the version doesn't exist (e.g. "no matching distribution"), run `pip index versions <package_name>` to see available versions, pick the closest available version to the one in this ticket, and update only that one line in `requirements.txt`.
3. Add a one-line comment directly above the changed line explaining what you changed and why (e.g. `# changed from 2.6.0 -> 2.6.1: 2.6.0 not available for this platform`).
4. Re-run the full Verify sequence from the top.
5. Do not change any other line while doing this.

**Commit message:** `fix: rebuild requirements.txt as UTF-8 and add previously-missing core ML dependencies`

---

### Ticket 0.4 — Confirm `.gitignore` covers generated artifacts

**Action:** Open `.gitignore`. Confirm it excludes `data/`, `artifacts/`, `tb_logs/`, `*.pyc`, `__pycache__/`, and `.venv/`. If any of these five patterns is missing, append the missing ones (each on its own line) to the end of `.gitignore`. If all five are already present in any form, make no change and say so.

**Verify:**
```bash
grep -c "data/" .gitignore
```
Expected: a number greater than or equal to 1.

**Commit message:** `chore: ensure .gitignore excludes generated data/artifacts`

---

### Ticket 0.5 — Full Phase 0 regression check

**Action:** None — this is a checkpoint.

**Verify:**
```bash
pytest tests/ -v
```
Expected: all tests that passed before Phase 0 still pass (140 tests, same as before — Phase 0 did not touch any file under `src/` or `tests/`, so this should be unchanged). If any test now fails, stop and report — something in Phase 0 broke an import path.

**Commit message:** (none — this ticket makes no changes)

---

# PHASE 1 — Fix Real Bugs, Close Critical Test Gaps

### Ticket 1.1 — Fix the missing `torch` import and add a misuse guard in `src/evaluation/metrics.py`

**Background (already confirmed, do not re-investigate):** `src/evaluation/metrics.py::replay_evaluate()` calls `torch.from_numpy(...)` and `torch.argsort(...)` but the file never imports `torch`, so calling this function raises `NameError`. Separately, the function calls `.parameters()` and `.get_distribution()` on its `policy` argument — these methods exist on `model.policy` (an SB3 `ActorCriticPolicy`, which is a `torch.nn.Module`), **not** on the top-level SB3 `model` object itself. Callers must pass `model.policy`, not `model`.

**Step A — Fix the import.** Open `src/evaluation/metrics.py`. Find this exact text block:
```python
import numpy as np
import pandas as pd
from scipy import stats

from src.rl_agent.environment import NewsRecommendEnv, cosine_similarity
```
Replace it with exactly:
```python
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.rl_agent.environment import NewsRecommendEnv, cosine_similarity

ALPHA = 0.05  # Significance threshold used consistently across this module.
```

**Step B — Fix the docstring and add a misuse guard.** In the same file, find this exact text block:
```python
def replay_evaluate(policy, test_sessions, news_df, cdi_cache, w=0.6, K=10, T=10):
    """Run offline replay evaluation of a trained policy.

    Inserts the trained policy into historical test sessions. At each
    step, the agent recommends an article; metrics are computed against
    the ground-truth clicks.

    Args:
        policy: Trained stable-baselines3 policy.
        test_sessions: List of test session objects.
        news_df: News DataFrame indexed by item_id.
        cdi_cache: Dict mapping (user_id, item_id) -> CDI score.
        w: Click reward weight used during training.
        K: Number of top recommendations to evaluate.
        T: Number of steps per session.

    Returns:
        Aggregated metrics dict from aggregate_metrics().
    """
    results = []
```
Replace it with exactly:
```python
def replay_evaluate(policy, test_sessions, news_df, cdi_cache, w=0.6, K=10, T=10):
    """Run offline replay evaluation of a trained policy.

    Inserts the trained policy into historical test sessions. At each
    step, the agent recommends an article; metrics are computed against
    the ground-truth clicks.

    Args:
        policy: The `.policy` attribute of a trained stable-baselines3
            model (e.g. pass `model.policy`, NOT the top-level `model`
            object itself). Must expose `.predict()`, `.parameters()`,
            and `.get_distribution()` (all present on SB3's
            `ActorCriticPolicy`, which is a `torch.nn.Module`).
        test_sessions: List of test session objects.
        news_df: News DataFrame indexed by item_id.
        cdi_cache: Dict mapping (user_id, item_id) -> CDI score.
        w: Click reward weight used during training.
        K: Number of top recommendations to evaluate.
        T: Number of steps per session.

    Returns:
        Aggregated metrics dict from aggregate_metrics().

    Raises:
        TypeError: If `policy` does not look like an SB3 `ActorCriticPolicy`
            (missing `.get_distribution`) — this usually means the caller
            passed `model` instead of `model.policy`.
    """
    if not hasattr(policy, "get_distribution"):
        raise TypeError(
            "replay_evaluate() expects model.policy (an SB3 ActorCriticPolicy "
            "exposing .get_distribution()), not the top-level SB3 model. "
            "Call replay_evaluate(model.policy, ...) instead of "
            "replay_evaluate(model, ...)."
        )
    results = []
```

**Verify:**
```bash
python3 -c "import ast; ast.parse(open('src/evaluation/metrics.py').read()); print('SYNTAX OK')"
grep -c "^import torch" src/evaluation/metrics.py
grep -c "ALPHA = 0.05" src/evaluation/metrics.py
```
Expected: `SYNTAX OK` printed, and both `grep -c` commands print `1`.

**Commit message:** `fix: add missing torch import and misuse guard to replay_evaluate()`

---

### Ticket 1.2 — Fix the significance threshold inconsistency in `significance_test()`

**Background:** `significance_test()` hardcodes `p_val < 0.01`, but the project's README and research log describe `p=0.017` as "significant" (i.e. they use the standard `p < 0.05` convention). This ticket makes the code consistent with the `ALPHA` constant added in Ticket 1.1.

**Action:** In `src/evaluation/metrics.py`, find this exact text block:
```python
    Returns:
        True if p < 0.01 (statistically significant difference).
    """
    t_stat, p_val = stats.ttest_rel(causal_rl_scores, baseline_scores)
    d = (np.mean(causal_rl_scores) - np.mean(baseline_scores)) / (
        np.std(causal_rl_scores) + 1e-10
    )
    print(
        f"{metric_name}: t={t_stat:.3f}, p={p_val:.4f}, Cohen_d={d:.3f}"
    )
    return bool(p_val < 0.01)
```
Replace it with exactly:
```python
    Returns:
        True if p < ALPHA (0.05 by default — see the module-level ALPHA
        constant defined at the top of this file).
    """
    t_stat, p_val = stats.ttest_rel(causal_rl_scores, baseline_scores)
    d = (np.mean(causal_rl_scores) - np.mean(baseline_scores)) / (
        np.std(causal_rl_scores) + 1e-10
    )
    print(
        f"{metric_name}: t={t_stat:.3f}, p={p_val:.4f}, Cohen_d={d:.3f}"
    )
    return bool(p_val < ALPHA)
```

**Verify:**
```bash
grep -c "p_val < ALPHA" src/evaluation/metrics.py
grep -c "p_val < 0.01" src/evaluation/metrics.py
```
Expected: first command prints `1`, second command prints `0`.

**Commit message:** `fix: significance_test() now uses the consistent ALPHA=0.05 threshold`

---

### Ticket 1.3 — Create `tests/test_replay_evaluate.py`

**Action:** Create a new file `tests/test_replay_evaluate.py` with exactly this content:

```python
"""Tests for src.evaluation.metrics.replay_evaluate — previously untested
and containing a missing-import bug (fixed in Ticket 1.1 of the
research-ready plan). These tests exercise the real function end-to-end
against a tiny synthetic session and a tiny real PPO model (trained for a
handful of timesteps), rather than mocking internal SB3 APIs.
"""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl_agent.environment import NewsRecommendEnv
from src.evaluation.metrics import replay_evaluate


def _make_candidate(item_id, emb):
    return type(
        "Candidate", (), {"item_id": item_id, "title_emb": np.array(emb, dtype=np.float32)}
    )()


def _make_session():
    candidates = [
        _make_candidate("N1", [1.0, 0.0, 0.0]),
        _make_candidate("N2", [0.0, 1.0, 0.0]),
        _make_candidate("N3", [0.0, 0.0, 1.0]),
    ]
    return type(
        "Session",
        (),
        {
            "user_id": "U1",
            "initial_history_emb": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "candidates": [candidates],
            "clicks": [[1, 0, 0]],
            "clicked_items": {"N1"},
        },
    )()


def _make_news_df():
    return pd.DataFrame(
        {
            "I_title_emb_full": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        },
        index=["N1", "N2", "N3"],
    )


def _train_tiny_policy(session):
    vec_env = DummyVecEnv(
        [lambda: NewsRecommendEnv([session], pd.DataFrame(), {}, w=0.6, K=3, T=1)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=64)
    return model


class TestReplayEvaluateWrongArgument:
    def test_raises_typeerror_when_passed_model_instead_of_model_policy(self):
        session = _make_session()
        news_df = _make_news_df()
        model = _train_tiny_policy(session)
        with pytest.raises(TypeError):
            replay_evaluate(model, [session], news_df, {}, w=0.6, K=3, T=1)


class TestReplayEvaluateCorrectUsage:
    def test_runs_without_raising_and_returns_expected_keys(self):
        session = _make_session()
        news_df = _make_news_df()
        model = _train_tiny_policy(session)

        result = replay_evaluate(model.policy, [session], news_df, {}, w=0.6, K=3, T=1)

        assert isinstance(result, dict)
        for key in (
            "ndcg_mean", "ndcg_std",
            "precision_mean", "precision_std",
            "ild_mean", "ild_std",
            "n_sessions",
        ):
            assert key in result
        assert result["n_sessions"] == 1
```

**Verify:**
```bash
pytest tests/test_replay_evaluate.py -v
```
Expected: `2 passed` (both tests pass). This may take up to ~1 minute due to the tiny PPO training runs — that is normal.

**Commit message:** `test: add end-to-end coverage for replay_evaluate() (previously 0% covered)`

---

### Ticket 1.4 — Create `tests/test_causal_model.py`

**Background:** `src/causal_model/` (ATE estimation, refutation tests — the paper's central causal-inference claim) has zero test coverage. This ticket adds tests against **synthetic data with a known, true treatment effect**, so the tests validate that the estimator recovers the correct answer, not just that it runs.

**Action:** Create a new file `tests/test_causal_model.py` with exactly this content:

```python
"""Tests for src.causal_model — previously 0% test coverage despite being
the module implementing this project's central causal-inference claim.

These tests use SYNTHETIC data with a KNOWN true average treatment effect
(ATE), so a passing test is evidence the estimator is actually correct,
not just that it runs without crashing.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display needed for tests

import numpy as np
import pandas as pd
import pytest
from dowhy import CausalModel

from src.causal_model.model import (
    create_causal_model,
    identify_effect,
    estimate_ate_ipw,
    estimate_ate_linear,
    positivity_check,
)
from src.causal_model.refutation import run_refutations


def _make_synthetic_confounded_data(n=2000, true_ate=2.0, seed=42):
    """Generate data with a KNOWN true ATE for validating the estimator.

    W is a confounder affecting both treatment assignment and outcome.
    Y = true_ate * A + W + noise, so a correctly backdoor-adjusted
    estimator should recover an ATE close to `true_ate`.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=n)
    propensity = 1 / (1 + np.exp(-W))
    A = rng.binomial(1, propensity)
    noise = rng.normal(0, 0.5, size=n)
    Y = true_ate * A + W + noise
    return pd.DataFrame({"W": W, "A": A, "Y": Y})


@pytest.fixture(scope="module")
def synthetic_df():
    return _make_synthetic_confounded_data()


class TestCreateCausalModel:
    def test_returns_causal_model_instance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        assert isinstance(model, CausalModel)


class TestIdentifyEffect:
    def test_returns_nonnull_estimand(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        assert estimand is not None


class TestEstimateATE:
    def test_ipw_recovers_true_ate_within_tolerance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_ipw(model, estimand)
        # Generous tolerance (abs=0.3) because IPW on n=2000 synthetic
        # samples still has real estimation noise — this is checking
        # correctness of direction and rough magnitude, not exactness.
        assert estimate.value == pytest.approx(2.0, abs=0.3)

    def test_linear_recovers_true_ate_within_tolerance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        assert estimate.value == pytest.approx(2.0, abs=0.3)


class TestPositivityCheck:
    def test_returns_scores_in_unit_interval(self, synthetic_df):
        ps = positivity_check(synthetic_df, common_causes=["W"], treatment="A")
        assert (ps >= 0.0).all() and (ps <= 1.0).all()


class TestRefutations:
    def test_all_three_refutations_run_without_raising(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        results = run_refutations(model, estimand, estimate, num_simulations=5)
        assert "placebo" in results
        assert "subset" in results
        assert "random_common_cause" in results

    def test_placebo_effect_is_near_zero(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        results = run_refutations(model, estimand, estimate, num_simulations=5)
        if results["placebo"] is not None:
            assert abs(results["placebo"].new_effect) < 0.5
```

**Verify:**
```bash
pytest tests/test_causal_model.py -v
```
Expected: `7 passed`. This may take 1–2 minutes (DoWhy fitting is not instant).

**Commit message:** `test: add synthetic ground-truth test coverage for src/causal_model/ (previously 0%)`

---

### Ticket 1.5 — Static bug scan across `src/`

**Action:** Run this command:
```bash
pip install pyflakes --quiet
python3 -m pyflakes src/
```
For every line of output that says `F821 undefined name '<name>'`, open the named file, locate the undefined name, and add the missing `import` statement for it at the top of that file (following the same import style already used in that file — standard library imports first, then third-party, then `src.` imports, matching the existing files you've already seen in this project). Do not change anything else in the file.

If `pyflakes` reports zero `F821` findings, make no changes and say so explicitly.

**Verify:**
```bash
python3 -m pyflakes src/ | grep F821
```
Expected: no output (empty) — i.e. the `grep` finds nothing.

**Commit message:** `fix: resolve remaining undefined-name (F821) issues found by pyflakes`

---

# PHASE 2 — CI

### Ticket 2.1 — Add GitHub Actions CI workflow

**Action:** Create the directory `.github/workflows/` if it does not exist, and create a new file `.github/workflows/ci.yml` with exactly this content:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  MPLBACKEND: Agg

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          # NOTE: pinned to 3.11 for broad wheel availability across
          # torch/dowhy/stable-baselines3/gymnasium. The README describes
          # the project as developed on Python 3.14 — if that is a hard
          # requirement, change this value, but first confirm every package
          # in requirements.txt publishes a 3.14 wheel, or CI will fail.
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Run tests with coverage
        run: pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70
```

**Verify:** Push this file to a branch and open a pull request (or push directly to `main` if that's the project's workflow), then check the GitHub Actions tab for this repository shows a run for this workflow that completes (pass or fail is informative either way — a fail here means Phase 2 surfaced a real problem to fix, which is expected on the first run; a fail because of ruff lint errors is fine to fix by running `ruff check src/ tests/ --fix` locally and committing the result).

Locally, before pushing, you can approximate the same check with:
```bash
ruff check src/ tests/
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70
```
Expected: both commands exit with code 0. If `ruff check` reports errors, run `ruff check src/ tests/ --fix` to auto-fix what it can, then re-run and manually fix anything left.

**Commit message:** `ci: add GitHub Actions workflow (lint + test + coverage floor)`

---

### Ticket 2.2 — Update `docs/codebase/CONCERNS.md` to reflect the CI fix

**Action:** Open `docs/codebase/CONCERNS.md`. Find this exact text block:
```
| High | **No CI/CD pipeline** — no automated tests, linting, or coverage enforcement | Scan output line 351 (no CI/CD detected) | Regressions not caught before execution | [ASK USER] Add GitHub Actions for automated test run |
```
Replace it with exactly:
```
| ~~High~~ Resolved | ~~No CI/CD pipeline~~ — **Resolved**: `.github/workflows/ci.yml` runs lint + tests + coverage floor (70%) on every push/PR | `.github/workflows/ci.yml` | N/A | N/A |
```

**Verify:**
```bash
grep -c "Resolved.*No CI/CD pipeline" docs/codebase/CONCERNS.md
```
Expected: `1`

**Commit message:** `docs: update CONCERNS.md — CI/CD gap resolved`

---

# PHASE 3 — Determinism & Seeding

### Ticket 3.1 — Add a `seed` parameter to `train_ppo()`

**Background:** `train_ppo()` in `src/rl_agent/train_ppo.py` never sets a random seed anywhere, so training is non-reproducible, and the saved checkpoint filename does not include the seed — meaning if you later train with multiple seeds (Phase 4), each run will **overwrite the previous checkpoint file**. This ticket fixes both problems.

**Step A.** In `src/rl_agent/train_ppo.py`, find this exact text block:
```python
    net_arch=None,
    tensorboard_log="./tb_logs/",
    checkpoint_dir="./checkpoints/",
    verbose=1,
    model_name=None,
):
```
Replace it with exactly:
```python
    net_arch=None,
    tensorboard_log="./tb_logs/",
    checkpoint_dir="./checkpoints/",
    verbose=1,
    model_name=None,
    seed=None,
):
```

**Step B.** In the same file, find this exact text block (in the docstring):
```python
        checkpoint_dir: Directory to save model checkpoints.
        verbose: Verbosity level (0 = silent, 1 = info).

    Returns:
        Tuple of (trained PPO model, checkpoint save path).
    """
```
Replace it with exactly:
```python
        checkpoint_dir: Directory to save model checkpoints.
        verbose: Verbosity level (0 = silent, 1 = info).
        seed: RNG seed for reproducibility. When set, passed directly to
            SB3's PPO constructor, which seeds the policy network
            initialization, the environment's RNG (via `env.reset(seed=...)`
            internally), and PyTorch/NumPy global RNGs used during training.
            Default None (non-deterministic).

    Returns:
        Tuple of (trained PPO model, checkpoint save path).
    """
```

**Step C.** In the same file, find this exact text block:
```python
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
```
Replace it with exactly:
```python
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )
```

**Step D.** In the same file, find this exact text block:
```python
    if model_name:
        save_path = str(checkpoint_path / model_name)
    else:
        save_path = str(checkpoint_path / f"ppo_causal_rs_w{int(w*10):02d}")
```
Replace it with exactly:
```python
    if model_name:
        save_path = str(checkpoint_path / model_name)
    else:
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        save_path = str(checkpoint_path / f"ppo_causal_rs_w{int(w*10):02d}{seed_suffix}")
```

**Verify:**
```bash
python3 -c "import ast; ast.parse(open('src/rl_agent/train_ppo.py').read()); print('SYNTAX OK')"
grep -c "seed=None," src/rl_agent/train_ppo.py
grep -c "seed_suffix" src/rl_agent/train_ppo.py
```
Expected: `SYNTAX OK` printed, first grep prints `1`, second grep prints `2`.

**Commit message:** `fix: thread seed parameter through train_ppo() for reproducibility and non-clobbering checkpoint names`

---

### Ticket 3.2 — Create `tests/test_determinism.py`

**Action:** Create a new file `tests/test_determinism.py` with exactly this content:

```python
"""Verifies that PPO training is reproducible given a fixed seed — this is
a prerequisite for Phase 4's multi-seed statistical analysis to be
meaningful (if training weren't seed-reproducible, "same seed, different
result" would silently corrupt every downstream statistic).
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl_agent.environment import NewsRecommendEnv


def _make_candidate(item_id, emb):
    return type(
        "Candidate", (), {"item_id": item_id, "title_emb": np.array(emb, dtype=np.float32)}
    )()


def _make_session():
    candidates = [
        _make_candidate("N1", [1.0, 0.0, 0.0]),
        _make_candidate("N2", [0.0, 1.0, 0.0]),
        _make_candidate("N3", [0.0, 0.0, 1.0]),
    ]
    return type(
        "Session",
        (),
        {
            "user_id": "U1",
            "initial_history_emb": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "candidates": [candidates],
            "clicks": [[1, 0, 0]],
        },
    )()


def _train_once(seed):
    session = _make_session()
    vec_env = DummyVecEnv(
        [lambda: NewsRecommendEnv([session], pd.DataFrame(), {}, w=0.6, K=3, T=1)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=64)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    return action


class TestDeterminism:
    def test_same_seed_gives_same_action(self):
        action_a = _train_once(seed=123)
        action_b = _train_once(seed=123)
        np.testing.assert_array_equal(action_a, action_b)
```

**Verify:**
```bash
pytest tests/test_determinism.py -v
```
Expected: `1 passed`.

**Commit message:** `test: verify PPO training is seed-reproducible`

---

### Ticket 3.3 — Mark `scripts/eval_only.py` as not the source of truth

**Background:** `scripts/eval_only.py` is a second, independent implementation of the evaluation logic (duplicating, and diverging from, `src/evaluation/metrics.py::replay_evaluate`). It also never seeds `np.random` (so its Random baseline and candidate subsampling change every run) and hardcodes `device="cuda"` (so it will crash on any machine without an NVIDIA GPU). Rather than rewriting this script (out of scope for this ticket), mark it clearly so nobody reports numbers from it going forward — Phase 4 introduces the real source of truth (`scripts/multi_seed_experiment.py`).

**Action:** In `scripts/eval_only.py`, find this exact text block (the first line of the file):
```python
"""Quick evaluation script — re-uses already-trained model with min-max CDI."""
```
Replace it with exactly:
```python
"""Quick evaluation script — re-uses already-trained model with min-max CDI.

WARNING — NOT THE SOURCE OF TRUTH FOR REPORTED RESULTS.
This script duplicates evaluation logic that also exists in
src/evaluation/metrics.py::replay_evaluate(), and the two have drifted out
of sync. It also never seeds np.random (results are non-reproducible run
to run) and hardcodes device="cuda" (will crash without an NVIDIA GPU).
Use scripts/multi_seed_experiment.py + src/evaluation/metrics.py for any
number that will be reported in a paper or README.
"""
```

**Verify:**
```bash
grep -c "NOT THE SOURCE OF TRUTH" scripts/eval_only.py
```
Expected: `1`

**Commit message:** `docs: mark scripts/eval_only.py as deprecated/non-authoritative`

---

### Ticket 3.4 — Full Phase 3 regression check

**Verify:**
```bash
pytest tests/ -v
```
Expected: all tests pass, including the two new files from Tickets 1.3, 1.4, 3.2 (total test count should now be higher than the original 140 — roughly 140 + 2 (replay_evaluate) + 7 (causal_model) + 1 (determinism) = 150).

**Commit message:** (none — checkpoint only)

---

# PHASE 4 — Statistical Rigor Infrastructure

### Ticket 4.1 — Create `docs/PREREGISTRATION.md`

**Action:** Create a new file `docs/PREREGISTRATION.md` with exactly this content:

```markdown
# Pre-Registration: Confirmatory Evaluation Configuration

This document fixes the exact configuration used for the **one** confirmatory
result reported in the paper/README, decided **before** looking at its
results. Everything in `docs/RESEARCH_LOG.md` before this document's
creation date is **exploratory** work (tuning, debugging, ablations) and
must not be cited as the paper's evidence — only the run described here can
be.

## Fixed Configuration

| Parameter | Value |
|---|---|
| Click weight `w` | 0.3 |
| Candidates per step `K` | 10 |
| Episode length `T` | 1 |
| Total training timesteps | 200,000 |
| Number of independent training seeds | 5 (`seeds = [1, 2, 3, 4, 5]`) |
| GCM mechanism | `fit_gcm_item_sensitive()` (HistGradientBoostingRegressor), per `docs/RESEARCH_LOG.md` 2026-06-17 entry |
| CDI normalization | Min-max normalized within each step's candidate pool (`NewsRecommendEnv._min_max_cdi()`) |
| Reported metrics | NDCG@10, Precision@10, ILD |
| Baselines compared | Random, Popularity, Logistic-CF (Phase 5) |
| Significance threshold (α) | 0.05 |
| Multiple-comparison correction | Holm-Bonferroni, applied across all comparisons in the same confirmatory table |
| Effect size | Cohen's d, computed across seed-level means (not across test-session-level scores) |
| Confidence intervals | Bootstrap, 95%, 10,000 resamples, computed across seed-level means |

## What Counts as Confirmatory

Only the output of `scripts/multi_seed_experiment.py` (Phase 4 of the
research-ready plan) run with the exact configuration above, analyzed by
`scripts/analyze_results.py`, may be reported as the paper's result. Any
number from `docs/RESEARCH_LOG.md`, `scripts/eval_only.py`, or any other ad
hoc run is exploratory and must be labeled as such if shown in the paper
(e.g. in an ablation/appendix section).

## Decision Rule (fixed in advance, not to be changed after seeing results)

The PPO-vs-Random comparison on NDCG is treated as a genuine positive result
**if and only if**:
1. `p_adj < 0.05` (Holm-Bonferroni corrected) for the PPO-vs-Random NDCG comparison, **and**
2. The corresponding effect size label is not `"negligible"` (i.e. `|d| >= 0.2`).

If both conditions hold: report a positive result (README/paper "Path A").
If either condition fails: report a diagnostic/negative result explaining
why the causal reward signal did not translate into a measurable
improvement (README/paper "Path B"). See Phase 6 of the implementation
plan for the exact template text for each path.
```

**Verify:**
```bash
test -f docs/PREREGISTRATION.md && echo "exists"
```
Expected: `exists`

**Commit message:** `docs: add pre-registration document fixing the confirmatory experiment configuration`

---

### Ticket 4.2 — Add `bootstrap_ci()` and `cohens_d_label()` to `src/evaluation/metrics.py`

**Action:** Open `src/evaluation/metrics.py`. Go to the **very end of the file** (after the last line of the existing `significance_test` function) and append exactly this text (with one blank line separating it from the existing last line of the file):

```python


def cohens_d_label(d: float) -> str:
    """Label a Cohen's d effect size using standard conventions.

    Args:
        d: Cohen's d value (can be negative; magnitude is used for labeling).

    Returns:
        One of "negligible" (< 0.2), "small" (0.2-0.5), "medium" (0.5-0.8),
        or "large" (>= 0.8).
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def bootstrap_ci(values: list, n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> dict:
    """Compute a bootstrap confidence interval for the mean of a list of values.

    Resamples `values` with replacement `n_boot` times, computes the mean of
    each resample, and returns the percentile confidence interval. Intended
    to be called on PER-SEED aggregate scores (e.g. one mean NDCG per
    independent training run), not on per-test-session scores from a single
    trained policy — those answer different questions.

    Args:
        values: List or array of per-seed scores.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for a 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys "mean", "ci_low", "ci_high", "excludes_zero" (bool,
        True if the interval does not contain zero).
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(boot_means, alpha))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha))
    return {
        "mean": float(values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }
```

**Verify:**
```bash
python3 -c "from src.evaluation.metrics import bootstrap_ci, cohens_d_label; print('IMPORT OK')"
```
Expected: `IMPORT OK`

**Commit message:** `feat: add bootstrap_ci() and cohens_d_label() to src/evaluation/metrics.py`

---

### Ticket 4.3 — Create `tests/test_bootstrap_and_effect_size.py`

**Action:** Create a new file `tests/test_bootstrap_and_effect_size.py` with exactly this content:

```python
import pytest

from src.evaluation.metrics import bootstrap_ci, cohens_d_label


class TestCohensDLabel:
    def test_negligible(self):
        assert cohens_d_label(0.05) == "negligible"

    def test_negligible_negative(self):
        assert cohens_d_label(-0.1) == "negligible"

    def test_small(self):
        assert cohens_d_label(0.3) == "small"

    def test_medium(self):
        assert cohens_d_label(0.6) == "medium"

    def test_large(self):
        assert cohens_d_label(1.2) == "large"

    def test_boundary_0_2_is_small(self):
        assert cohens_d_label(0.2) == "small"

    def test_boundary_0_8_is_large(self):
        assert cohens_d_label(0.8) == "large"


class TestBootstrapCI:
    def test_returns_expected_keys(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(values, n_boot=500, seed=1)
        for key in ("mean", "ci_low", "ci_high", "excludes_zero"):
            assert key in result

    def test_mean_is_correct(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(values, n_boot=500, seed=1)
        assert result["mean"] == pytest.approx(3.0)

    def test_ci_low_le_ci_high(self):
        values = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19]
        result = bootstrap_ci(values, n_boot=1000, seed=1)
        assert result["ci_low"] <= result["ci_high"]

    def test_excludes_zero_when_all_positive(self):
        values = [5.0, 6.0, 5.5, 6.5, 5.8]
        result = bootstrap_ci(values, n_boot=1000, seed=1)
        assert result["excludes_zero"] is True

    def test_does_not_exclude_zero_when_centered_on_zero(self):
        values = [-1.0, 1.0, -0.5, 0.5, 0.1, -0.1, 0.0]
        result = bootstrap_ci(values, n_boot=2000, seed=1)
        assert result["excludes_zero"] is False

    def test_deterministic_with_same_seed(self):
        values = [1.0, 2.0, 3.0, 2.5, 1.5]
        r1 = bootstrap_ci(values, n_boot=500, seed=7)
        r2 = bootstrap_ci(values, n_boot=500, seed=7)
        assert r1 == r2
```

**Verify:**
```bash
pytest tests/test_bootstrap_and_effect_size.py -v
```
Expected: `13 passed`.

**Commit message:** `test: add coverage for bootstrap_ci() and cohens_d_label()`

---

### Ticket 4.4 — Create `scripts/multi_seed_experiment.py` **[REQUIRES MIND DATA]**

**Background:** This script trains PPO across 5 independent seeds (per `docs/PREREGISTRATION.md`) and evaluates each trained policy plus Random and Popularity baselines on the same held-out test set, saving everything to a single JSON file for later analysis. This is the confirmatory experiment.

**Action:** Create a new file `scripts/multi_seed_experiment.py` with exactly this content:

```python
"""Multi-seed PPO training + evaluation for confirmatory statistical analysis.

Trains PPO across multiple independent seeds using the exact configuration
fixed in docs/PREREGISTRATION.md, evaluates each trained policy plus the
Random and Popularity baselines on the same held-out test set, and saves
all per-seed, per-method results to artifacts/multi_seed_results.json.

This script requires the MIND data pipeline to have already been run
(data/scm_train.parquet, data/scm_test.parquet, and
artifacts/cdi_cache.pkl must exist). It does NOT run in CI — run it
manually once that data is prepared.

Usage:
    python scripts/multi_seed_experiment.py
"""
import ast
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rl_agent.train_ppo import train_ppo
from src.evaluation.metrics import replay_evaluate, ndcg_at_k, precision_at_k, ild

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "multi_seed_results.json"

# --- Pre-registered configuration — see docs/PREREGISTRATION.md ---
# Do NOT change these values without also updating docs/PREREGISTRATION.md.
SEEDS = [1, 2, 3, 4, 5]
W = 0.3
K = 10
T = 1
TOTAL_TIMESTEPS = 200_000


def _require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "This script requires the MIND data pipeline to have already "
            "been run (see README.md 'Run the Pipeline'). Run phases 1-3 "
            "first, then re-run this script."
        )


def _to_array(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=np.float32)
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    raise TypeError(f"Unexpected type {type(val)}")


def _build_sessions(df):
    """Build session objects from a SCM dataframe grouped by impression_id."""
    sessions = []
    for imp_id, group in df.groupby("impression_id", sort=False):
        sessions.append(type("Session", (), {
            "user_id": group["user_id"].iloc[0],
            "initial_history_emb": _to_array(group["U_history_emb_full"].iloc[0]).flatten(),
            "candidates": [[type("C", (), {"item_id": iid, "title_emb": emb})()
                           for iid, emb in zip(group["item_id"].tolist(),
                                               group["I_title_emb_full"].apply(_to_array).tolist())]],
            "clicks": [group["Y_click"].tolist()],
            "candidate_pool": group["item_id"].tolist(),
            "clicked_items": set(group[group["Y_click"] == 1]["item_id"].tolist()),
        })())
    return sessions


def _evaluate_random_baseline(sessions, news_df, K, rng):
    """Score-based Random baseline using a seeded RNG (not global np.random,
    unlike scripts/eval_only.py — see Ticket 3.3)."""
    results = []
    for s in sessions:
        pool = list(s.candidate_pool)
        rng.shuffle(pool)
        rec_items = pool[:K]
        rec_embs = [_to_array(news_df.loc[iid, "I_title_emb_full"])
                    for iid in rec_items if iid in news_df.index]
        results.append({
            "ndcg": ndcg_at_k(rec_items, s.clicked_items, K),
            "precision": precision_at_k(rec_items, s.clicked_items, K),
            "ild": ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0,
        })
    return pd.DataFrame(results)


def _evaluate_popularity_baseline(sessions, news_df, pop_counter, K):
    results = []
    for s in sessions:
        scored = sorted(
            [(iid, pop_counter.get(iid, 0)) for iid in s.candidate_pool],
            key=lambda x: -x[1],
        )
        rec_items = [iid for iid, _ in scored[:K]]
        rec_embs = [_to_array(news_df.loc[iid, "I_title_emb_full"])
                    for iid in rec_items if iid in news_df.index]
        results.append({
            "ndcg": ndcg_at_k(rec_items, s.clicked_items, K),
            "precision": precision_at_k(rec_items, s.clicked_items, K),
            "ild": ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0,
        })
    return pd.DataFrame(results)


def _metrics_row(seed, method, df):
    return {
        "seed": seed, "method": method,
        "ndcg_mean": float(df["ndcg"].mean()), "ndcg_std": float(df["ndcg"].std()),
        "precision_mean": float(df["precision"].mean()), "precision_std": float(df["precision"].std()),
        "ild_mean": float(df["ild"].mean()), "ild_std": float(df["ild"].std()),
        "n_sessions": len(df),
    }


def main():
    train_path = DATA / "scm_train.parquet"
    test_path = DATA / "scm_test.parquet"
    cdi_path = ARTIFACTS / "cdi_cache.pkl"
    _require_file(train_path)
    _require_file(test_path)
    _require_file(cdi_path)

    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    with open(cdi_path, "rb") as f:
        cdi_cache = pickle.load(f)

    train_sessions = _build_sessions(train_df)
    test_sessions = _build_sessions(test_df)
    news_df = test_df[["item_id", "I_title_emb_full"]].drop_duplicates("item_id").set_index("item_id")
    pop_counter = Counter(train_df["item_id"])

    all_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

        print("Training PPO...")
        model, save_path = train_ppo(
            train_sessions,
            news_df,
            cdi_cache,
            total_timesteps=TOTAL_TIMESTEPS,
            n_envs=1,
            w=W,
            K=K,
            T=T,
            checkpoint_dir=str(ARTIFACTS / "checkpoints"),
            model_name=f"ppo_multiseed_seed{seed}",
            seed=seed,
            verbose=0,
        )
        print(f"Saved to {save_path}")

        print("Evaluating PPO...")
        ppo_metrics = replay_evaluate(model.policy, test_sessions, news_df, cdi_cache, w=W, K=K, T=T)
        all_results.append({"seed": seed, "method": "PPO", **ppo_metrics})

        print("Evaluating Random...")
        rng = np.random.default_rng(seed)
        rand_df = _evaluate_random_baseline(test_sessions, news_df, K, rng)
        all_results.append(_metrics_row(seed, "Random", rand_df))

        print("Evaluating Popularity...")
        pop_df = _evaluate_popularity_baseline(test_sessions, news_df, pop_counter, K)
        all_results.append(_metrics_row(seed, "Popularity", pop_df))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} result rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

**Verify:** This ticket requires the MIND data pipeline to already have been run. If `data/scm_train.parquet`, `data/scm_test.parquet`, and `artifacts/cdi_cache.pkl` exist, run:
```bash
python3 scripts/multi_seed_experiment.py
```
Expected: the script runs to completion (this can take a long time — 5 seeds × 200,000 timesteps each — do not interrupt it) and ends by printing `Saved 15 result rows to .../artifacts/multi_seed_results.json`.

If those data files do not exist yet, do not attempt to run this script. Instead run only a syntax check:
```bash
python3 -c "import ast; ast.parse(open('scripts/multi_seed_experiment.py').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`. State explicitly that full execution is pending until the MIND data pipeline (Phases 1-3) has been run.

**Commit message:** `feat: add scripts/multi_seed_experiment.py — the single source of truth for confirmatory results`

---

### Ticket 4.5 — Create `scripts/analyze_results.py` **[REQUIRES MIND DATA — depends on Ticket 4.4's output]**

**Action:** Create a new file `scripts/analyze_results.py` with exactly this content:

```python
"""Analyze multi-seed experiment results: bootstrap CIs, corrected
significance tests, and effect-size labeling.

Reads artifacts/multi_seed_results.json (produced by
scripts/multi_seed_experiment.py) and produces:
  - artifacts/final_results_table.csv
  - a printed summary to stdout

Usage:
    python scripts/analyze_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import bootstrap_ci, cohens_d_label

ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "multi_seed_results.json"
OUTPUT_CSV = ARTIFACTS / "final_results_table.csv"

METRICS = ["ndcg_mean", "precision_mean", "ild_mean"]
ALPHA = 0.05


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run scripts/multi_seed_experiment.py first."
        )
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def per_seed_matrix(df, method, metric):
    """Return an array of one value per seed for (method, metric), sorted by seed."""
    sub = df[df["method"] == method].sort_values("seed")
    return sub[metric].to_numpy()


def main():
    df = load_results()
    methods_present = sorted(df["method"].unique())
    seeds = sorted(df["seed"].unique())
    print(f"Loaded results for {len(seeds)} seeds: {seeds}")
    print(f"Methods present: {methods_present}")

    summary_rows = []
    raw_pvalues = []
    comparison_labels = []

    baseline_methods = [m for m in methods_present if m != "PPO"]

    for metric in METRICS:
        ppo_vals = per_seed_matrix(df, "PPO", metric)

        for method_name in methods_present:
            vals = per_seed_matrix(df, method_name, metric)
            ci = bootstrap_ci(vals.tolist(), n_boot=10000, seed=42)
            summary_rows.append({
                "metric": metric,
                "row_type": "mean_ci",
                "method": method_name,
                "mean_across_seeds": ci["mean"],
                "ci_95_low": ci["ci_low"],
                "ci_95_high": ci["ci_high"],
                "n_seeds": len(vals),
            })

        for baseline_name in baseline_methods:
            baseline_vals = per_seed_matrix(df, baseline_name, metric)
            diffs = ppo_vals - baseline_vals
            t_stat, p_val = stats.ttest_rel(ppo_vals, baseline_vals)
            d = float(np.mean(diffs) / (np.std(ppo_vals, ddof=1) + 1e-10))
            raw_pvalues.append(p_val)
            comparison_labels.append(f"PPO vs {baseline_name} ({metric})")
            summary_rows.append({
                "metric": metric,
                "row_type": "comparison",
                "method": f"PPO vs {baseline_name}",
                "t_stat": float(t_stat),
                "p_raw": float(p_val),
                "cohens_d": d,
                "effect_label": cohens_d_label(d),
            })

    reject, p_corrected, _, _ = multipletests(raw_pvalues, alpha=ALPHA, method="holm")

    print("\n" + "=" * 70)
    print(f"CORRECTED SIGNIFICANCE RESULTS (Holm-Bonferroni, alpha={ALPHA:.2f})")
    print("=" * 70)
    idx = 0
    for row in summary_rows:
        if row.get("row_type") == "comparison":
            row["p_adj"] = float(p_corrected[idx])
            row["significant_after_correction"] = bool(reject[idx])
            print(f"{comparison_labels[idx]:40s} p_raw={raw_pvalues[idx]:.4f}  "
                  f"p_adj={p_corrected[idx]:.4f}  d={row['cohens_d']:.3f} "
                  f"({row['effect_label']})  "
                  f"{'SIGNIFICANT' if reject[idx] else 'not significant'} (corrected)")
            idx += 1

    result_df = pd.DataFrame(summary_rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull table written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
```

**Verify:** If `artifacts/multi_seed_results.json` exists (from Ticket 4.4), run:
```bash
python3 scripts/analyze_results.py
```
Expected: the script prints a table of corrected significance results and ends with `Full table written to .../artifacts/final_results_table.csv`.

If that file does not exist yet, run only a syntax check:
```bash
python3 -c "import ast; ast.parse(open('scripts/analyze_results.py').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`. State explicitly that full execution is pending on Ticket 4.4's output.

**Commit message:** `feat: add scripts/analyze_results.py — Holm-Bonferroni-corrected, bootstrap-CI, effect-size-labeled analysis`

---

### Ticket 4.6 — Full Phase 4 regression check

**Verify:**
```bash
pytest tests/ -v
```
Expected: all tests pass, count now roughly 150 + 13 (Ticket 4.3) = 163.

**Commit message:** (none — checkpoint only)

---

# PHASE 5 — Real Baseline (Logistic-CF)

### Ticket 5.1 — Create `src/baselines/__init__.py`

**Action:** Create the directory `src/baselines/` and inside it a new file `src/baselines/__init__.py` with exactly this content:
```python
from src.baselines.logistic_cf import (
    build_feature_matrix,
    train_logistic_cf,
    score_candidates,
)

__all__ = ["build_feature_matrix", "train_logistic_cf", "score_candidates"]
```

**Verify:**
```bash
test -f src/baselines/__init__.py && echo "exists"
```
Expected: `exists`. (This will not yet import successfully — `logistic_cf.py` is created in the next ticket. That's expected; do not try to fix it yet.)

**Commit message:** `feat: scaffold src/baselines package`

---

### Ticket 5.2 — Create `src/baselines/logistic_cf.py`

**Background:** The project's original plan (per `docs/resources/CausalRS_Complete_Documentation_v2.md`) called for a real correlational "CF baseline" to compare against the causal-RL system, but only Random and Popularity were ever implemented. This ticket adds a logistic-regression click-prediction baseline using the existing SCM feature columns.

**Action:** Create a new file `src/baselines/logistic_cf.py` with exactly this content:

```python
"""Logistic regression click-prediction baseline (a minimal, correlational
'CF-style' baseline) for comparison against the causal-RL PPO agent.

This is the baseline the original project plan called for ("<5% degradation
vs a CF baseline") but was never implemented before this module — only
Random and Popularity baselines existed previously.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_PREFIXES = ("U_pca_", "I_entity_pca_", "I_title_pca_")
EXTRA_NUMERIC_FEATURES = ("U_dwell_mean", "I_sentiment")
CATEGORICAL_FEATURES = ("I_category",)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Find all PCA + numeric feature columns present in the dataframe."""
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    cols += [c for c in EXTRA_NUMERIC_FEATURES if c in df.columns]
    return cols


def build_feature_matrix(df: pd.DataFrame, fit_columns: Optional[List[str]] = None):
    """Build a numeric feature matrix (PCA dims + numeric + one-hot category).

    Args:
        df: SCM dataframe with PCA/numeric/categorical columns.
        fit_columns: If provided (from a previously-built training matrix),
            the returned matrix's columns are reindexed to match exactly,
            filling any missing one-hot category with 0. Pass this when
            transforming test data so train/test have identical columns.

    Returns:
        Tuple of (numpy feature matrix, list of column names used).
    """
    numeric_cols = _select_feature_columns(df)
    numeric_part = df[numeric_cols].fillna(0.0)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if cat_cols:
        cat_part = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
    else:
        cat_part = pd.DataFrame(index=df.index)

    full = pd.concat([numeric_part, cat_part], axis=1)

    if fit_columns is not None:
        full = full.reindex(columns=fit_columns, fill_value=0.0)
        columns = fit_columns
    else:
        columns = list(full.columns)

    return full.to_numpy(dtype=np.float64), columns


def train_logistic_cf(train_df: pd.DataFrame, seed: int = 42) -> dict:
    """Train a logistic regression click-prediction model.

    Args:
        train_df: SCM training dataframe with a "Y_click" column.
        seed: Random seed for the LogisticRegression solver.

    Returns:
        Dict with keys "model" (fitted LogisticRegression) and "columns"
        (the exact feature column order used — required for scoring test
        data with a matching feature matrix shape).
    """
    X, columns = build_feature_matrix(train_df)
    y = train_df["Y_click"].to_numpy()
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X, y)
    return {"model": model, "columns": columns}


def score_candidates(fitted: dict, candidates_df: pd.DataFrame) -> np.ndarray:
    """Score a set of candidate rows with the fitted model.

    Args:
        fitted: Output of train_logistic_cf().
        candidates_df: Dataframe of candidate rows with the same feature
            columns as the training data (an extra Y_click column, if
            present, is ignored).

    Returns:
        1-D array of predicted click probabilities, one per row, in the
        same row order as candidates_df.
    """
    X, _ = build_feature_matrix(candidates_df, fit_columns=fitted["columns"])
    return fitted["model"].predict_proba(X)[:, 1]
```

**Verify:**
```bash
python3 -c "from src.baselines.logistic_cf import build_feature_matrix, train_logistic_cf, score_candidates; print('IMPORT OK')"
```
Expected: `IMPORT OK`

**Commit message:** `feat: implement Logistic-CF baseline (src/baselines/logistic_cf.py)`

---

### Ticket 5.3 — Create `tests/test_baselines.py`

**Action:** Create a new file `tests/test_baselines.py` with exactly this content:

```python
import numpy as np
import pandas as pd

from src.baselines.logistic_cf import (
    build_feature_matrix,
    train_logistic_cf,
    score_candidates,
)


def _make_fake_scm_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "U_pca_0": rng.normal(size=n),
        "U_pca_1": rng.normal(size=n),
        "I_entity_pca_0": rng.normal(size=n),
        "I_title_pca_0": rng.normal(size=n),
        "U_dwell_mean": rng.normal(size=n),
        "I_sentiment": rng.uniform(-1, 1, size=n),
        "I_category": rng.choice(["sports", "news", "finance"], size=n),
    })
    # Y_click is weakly dependent on I_sentiment so the model has signal to learn.
    prob = 1 / (1 + np.exp(-df["I_sentiment"].to_numpy()))
    df["Y_click"] = rng.binomial(1, prob)
    return df


class TestBuildFeatureMatrix:
    def test_returns_matrix_and_columns(self):
        df = _make_fake_scm_df()
        X, columns = build_feature_matrix(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(columns)
        assert not np.isnan(X).any()

    def test_test_set_reindexes_to_train_columns(self):
        train_df = _make_fake_scm_df(n=100, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        _, train_columns = build_feature_matrix(train_df)
        X_test, test_columns = build_feature_matrix(test_df, fit_columns=train_columns)
        assert test_columns == train_columns
        assert X_test.shape[1] == len(train_columns)


class TestTrainAndScore:
    def test_train_returns_model_and_columns(self):
        df = _make_fake_scm_df()
        fitted = train_logistic_cf(df, seed=42)
        assert "model" in fitted
        assert "columns" in fitted

    def test_score_candidates_returns_probabilities_in_unit_interval(self):
        train_df = _make_fake_scm_df(n=200, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        fitted = train_logistic_cf(train_df, seed=42)
        scores = score_candidates(fitted, test_df)
        assert len(scores) == len(test_df)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_deterministic_with_same_seed(self):
        train_df = _make_fake_scm_df(n=200, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        fitted_a = train_logistic_cf(train_df, seed=42)
        fitted_b = train_logistic_cf(train_df, seed=42)
        scores_a = score_candidates(fitted_a, test_df)
        scores_b = score_candidates(fitted_b, test_df)
        np.testing.assert_array_almost_equal(scores_a, scores_b)
```

**Verify:**
```bash
pytest tests/test_baselines.py -v
```
Expected: `5 passed`.

**Commit message:** `test: add coverage for src/baselines/logistic_cf.py`

---

### Ticket 5.4 — Wire Logistic-CF into `scripts/multi_seed_experiment.py` and `scripts/analyze_results.py`

**Step A.** In `scripts/multi_seed_experiment.py`, find this exact text block:
```python
from src.rl_agent.train_ppo import train_ppo
from src.evaluation.metrics import replay_evaluate, ndcg_at_k, precision_at_k, ild
```
Replace it with exactly:
```python
from src.rl_agent.train_ppo import train_ppo
from src.evaluation.metrics import replay_evaluate, ndcg_at_k, precision_at_k, ild
from src.baselines.logistic_cf import train_logistic_cf, score_candidates
```

**Step B.** In the same file, find this exact text block:
```python
def _evaluate_popularity_baseline(sessions, news_df, pop_counter, K):
```
Replace it with exactly:
```python
def _evaluate_logistic_cf_baseline(sessions, test_df, fitted_cf, K):
    """Score-based Logistic-CF baseline (see src/baselines/logistic_cf.py)."""
    results = []
    candidates_by_item = test_df.drop_duplicates("item_id").set_index("item_id")
    for s in sessions:
        pool = s.candidate_pool
        rows = candidates_by_item.loc[[iid for iid in pool if iid in candidates_by_item.index]]
        if len(rows) == 0:
            results.append({"ndcg": 0.0, "precision": 0.0, "ild": 0.0})
            continue
        scores = score_candidates(fitted_cf, rows)
        ranked_item_ids = rows.index.to_numpy()[np.argsort(-scores)]
        rec_items = ranked_item_ids[:K].tolist()
        news_df = test_df[["item_id", "I_title_emb_full"]].drop_duplicates("item_id").set_index("item_id")
        rec_embs = [_to_array(news_df.loc[iid, "I_title_emb_full"])
                    for iid in rec_items if iid in news_df.index]
        results.append({
            "ndcg": ndcg_at_k(rec_items, s.clicked_items, K),
            "precision": precision_at_k(rec_items, s.clicked_items, K),
            "ild": ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0,
        })
    return pd.DataFrame(results)


def _evaluate_popularity_baseline(sessions, news_df, pop_counter, K):
```

**Step C.** In the same file, find this exact text block:
```python
    train_sessions = _build_sessions(train_df)
    test_sessions = _build_sessions(test_df)
    news_df = test_df[["item_id", "I_title_emb_full"]].drop_duplicates("item_id").set_index("item_id")
    pop_counter = Counter(train_df["item_id"])

    all_results = []
```
Replace it with exactly:
```python
    train_sessions = _build_sessions(train_df)
    test_sessions = _build_sessions(test_df)
    news_df = test_df[["item_id", "I_title_emb_full"]].drop_duplicates("item_id").set_index("item_id")
    pop_counter = Counter(train_df["item_id"])

    print("Training Logistic-CF baseline (trained once — deterministic given "
          "a fixed seed, so it does not need retraining per PPO seed)...")
    fitted_cf = train_logistic_cf(train_df, seed=42)

    all_results = []
```

**Step D.** In the same file, find this exact text block:
```python
        print("Evaluating Popularity...")
        pop_df = _evaluate_popularity_baseline(test_sessions, news_df, pop_counter, K)
        all_results.append(_metrics_row(seed, "Popularity", pop_df))
```
Replace it with exactly:
```python
        print("Evaluating Popularity...")
        pop_df = _evaluate_popularity_baseline(test_sessions, news_df, pop_counter, K)
        all_results.append(_metrics_row(seed, "Popularity", pop_df))

        print("Evaluating Logistic-CF...")
        # Logistic-CF is deterministic given seed=42 (fixed above) and does
        # not vary across PPO training seeds. We repeat the same result
        # under each seed label so scripts/analyze_results.py's per-seed
        # bootstrap logic works unmodified for every method, including this
        # one — this is intentional, not a bug: it correctly shows zero
        # across-run variance for a deterministic method.
        cf_df = _evaluate_logistic_cf_baseline(test_sessions, test_df, fitted_cf, K)
        all_results.append(_metrics_row(seed, "Logistic-CF", cf_df))
```

**Verify:**
```bash
python3 -c "import ast; ast.parse(open('scripts/multi_seed_experiment.py').read()); print('SYNTAX OK')"
grep -c "Logistic-CF" scripts/multi_seed_experiment.py
```
Expected: `SYNTAX OK` printed, and the grep count is `4` or more.

No changes are needed to `scripts/analyze_results.py` — it already loops over `methods_present = sorted(df["method"].unique())`, so `"Logistic-CF"` will be picked up automatically once it appears in `multi_seed_results.json`.

**Commit message:** `feat: wire Logistic-CF baseline into the multi-seed confirmatory experiment`

---

### Ticket 5.5 — Full Phase 5 regression check

**Verify:**
```bash
pytest tests/ -v
```
Expected: all tests pass, count now roughly 163 + 5 (Ticket 5.3) = 168.

**Commit message:** (none — checkpoint only)

---

# PHASE 6 — Narrative Decision Rule + README Rewrite

**This entire phase requires `artifacts/final_results_table.csv` to exist (Ticket 4.5's output, which requires Ticket 4.4's output, which requires the MIND data pipeline). Do not attempt Phase 6 until that file exists.**

### Ticket 6.1 — Determine which narrative path applies

**Action:** Open `artifacts/final_results_table.csv`. Find the row where `metric == "ndcg_mean"` and `method == "PPO vs Random"`. Read its `p_adj` and `effect_label` values.

- **If `p_adj < 0.05` AND `effect_label` is NOT `"negligible"`:** the applicable path is **Path A**.
- **Otherwise:** the applicable path is **Path B**.

This is a mechanical rule — do not use judgment beyond reading these two values from the CSV. State explicitly in your response which path applies and cite the exact `p_adj` and `effect_label` values you read.

**Verify:** Re-read the same row a second time and confirm your stated `p_adj`/`effect_label` values match exactly what is in the file.

**Commit message:** (none — this ticket is a decision, not a code change)

---

### Ticket 6.2 — Rewrite the README "Key Results" section

**Action:** Open `README.md`. Find the section starting with the line `## Key Results` and ending right before the line `## Technology Stack`. Replace that entire section with **one** of the two templates below, chosen per Ticket 6.1's outcome. Fill in every `{{PLACEHOLDER}}` with the exact numeric value from `artifacts/final_results_table.csv` (do not round differently than the CSV shows; copy the digits exactly).

**If Path A applies, use this template:**
```markdown
## Key Results

Confirmatory evaluation (5 independent training seeds, pre-registered
configuration in `docs/PREREGISTRATION.md`, Holm-Bonferroni-corrected
significance, bootstrap 95% CIs across seeds):

| Method | NDCG@10 (mean, 95% CI) |
|---|---|
| PPO (Causal-RL) | {{PPO_NDCG_MEAN}} [{{PPO_NDCG_CI_LOW}}, {{PPO_NDCG_CI_HIGH}}] |
| Random | {{RANDOM_NDCG_MEAN}} [{{RANDOM_NDCG_CI_LOW}}, {{RANDOM_NDCG_CI_HIGH}}] |
| Popularity | {{POP_NDCG_MEAN}} [{{POP_NDCG_CI_LOW}}, {{POP_NDCG_CI_HIGH}}] |
| Logistic-CF | {{CF_NDCG_MEAN}} [{{CF_NDCG_CI_LOW}}, {{CF_NDCG_CI_HIGH}}] |

PPO significantly outperforms Random on NDCG@10 after correction for
multiple comparisons (p_adj={{P_ADJ}}, Cohen's d={{COHENS_D}}, a
{{EFFECT_LABEL}} effect), across 5 independent training seeds. See
`docs/RESEARCH_LOG.md` for the full exploratory tuning history that led to
this configuration, and `artifacts/final_results_table.csv` for the
complete results table.
```

**If Path B applies, use this template instead:**
```markdown
## Key Results

Confirmatory evaluation (5 independent training seeds, pre-registered
configuration in `docs/PREREGISTRATION.md`, Holm-Bonferroni-corrected
significance, bootstrap 95% CIs across seeds):

| Method | NDCG@10 (mean, 95% CI) |
|---|---|
| PPO (Causal-RL) | {{PPO_NDCG_MEAN}} [{{PPO_NDCG_CI_LOW}}, {{PPO_NDCG_CI_HIGH}}] |
| Random | {{RANDOM_NDCG_MEAN}} [{{RANDOM_NDCG_CI_LOW}}, {{RANDOM_NDCG_CI_HIGH}}] |
| Popularity | {{POP_NDCG_MEAN}} [{{POP_NDCG_CI_LOW}}, {{POP_NDCG_CI_HIGH}}] |
| Logistic-CF | {{CF_NDCG_MEAN}} [{{CF_NDCG_CI_LOW}}, {{CF_NDCG_CI_HIGH}}] |

After correction for multiple comparisons across 5 independent training
seeds, PPO does **not** show a statistically significant or practically
meaningful improvement over Random on NDCG@10 (p_adj={{P_ADJ}}, Cohen's
d={{COHENS_D}}, a {{EFFECT_LABEL}} effect). This project's main finding is
diagnostic rather than a positive result: `docs/RESEARCH_LOG.md` documents
why — the counterfactual diversity signal (CDI) has very low within-session
variance, and the click signal is too sparse (~0.8% positive rate) to
compensate, so the RL reward is close to constant regardless of which item
is chosen. See `docs/RESEARCH_LOG.md`'s 2026-06-11 "RL-Guided Hyperparameter
Tuning" entry for the full root-cause analysis, and
`artifacts/final_results_table.csv` for the complete results table.
```

**Verify:**
```bash
grep -c "{{" README.md
```
Expected: `0` (i.e. every placeholder was actually filled in — none left as literal `{{...}}` text).

**Commit message:** `docs: replace README Key Results with corrected, multi-seed confirmatory numbers`

---

### Ticket 6.3 — Update the README "Note" callout near the top

**Action:** Open `README.md`. Find this exact text block:
```
Note

This is a **research project**, not a production system. After min-max CDI normalization, PPO now shows statistically significant NDCG gains over Random (p=0.017). See `docs/RESEARCH_LOG.md` for full history.
```
Replace it with **one** of the following, matching the path chosen in Ticket 6.1:

**If Path A:**
```
Note

This is a **research project**, not a production system. Across 5 independent training seeds (the pre-registered confirmatory configuration in `docs/PREREGISTRATION.md`), PPO shows a statistically significant, Holm-Bonferroni-corrected NDCG improvement over Random. See `docs/PREREGISTRATION.md` for the exact protocol and `docs/RESEARCH_LOG.md` for the full exploratory history.
```

**If Path B:**
```
Note

This is a **research project**, not a production system, and its main result is a **diagnostic finding, not a positive result**: after correcting for multiple comparisons across 5 independent training seeds, the causal-RL agent does not reliably outperform a Random baseline. See `docs/PREREGISTRATION.md` for the exact protocol and `docs/RESEARCH_LOG.md`'s root-cause analysis of why the counterfactual reward signal did not translate into a measurable improvement.
```

**Verify:**
```bash
grep -c "p=0.017" README.md
```
Expected: `0` (the old, since-superseded number no longer appears anywhere in the README).

**Commit message:** `docs: update README note callout to match corrected confirmatory result`

---

# PHASE 7 — Paper-Support Artifacts

### Ticket 7.1 — Create `scripts/generate_paper_figures.py` **[REQUIRES Ticket 4.5's output]**

**Action:** Create a new file `scripts/generate_paper_figures.py` with exactly this content:

```python
"""Generate a bar chart (mean +/- 95% CI, per method, per metric) from
artifacts/final_results_table.csv for direct use in a paper.

Usage:
    python scripts/generate_paper_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
FIGURES_DIR = ROOT / "docs" / "figures"
INPUT_CSV = ARTIFACTS / "final_results_table.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run scripts/analyze_results.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    mean_ci_rows = df[df["row_type"] == "mean_ci"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for metric in sorted(mean_ci_rows["metric"].unique()):
        sub = mean_ci_rows[mean_ci_rows["metric"] == metric].sort_values("method")
        methods = sub["method"].tolist()
        means = sub["mean_across_seeds"].tolist()
        lower_err = [m - lo for m, lo in zip(means, sub["ci_95_low"])]
        upper_err = [hi - m for m, hi in zip(means, sub["ci_95_high"])]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(methods, means, yerr=[lower_err, upper_err], capsize=5)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — mean across seeds, 95% bootstrap CI")
        fig.tight_layout()

        out_path = FIGURES_DIR / f"{metric}_bar_chart.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
```

**Verify:** If `artifacts/final_results_table.csv` exists, run:
```bash
python3 scripts/generate_paper_figures.py
ls docs/figures/
```
Expected: one `.png` file per metric (`ndcg_mean_bar_chart.png`, `precision_mean_bar_chart.png`, `ild_mean_bar_chart.png`) listed.

If the CSV does not exist yet, run only:
```bash
python3 -c "import ast; ast.parse(open('scripts/generate_paper_figures.py').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`.

**Commit message:** `feat: add scripts/generate_paper_figures.py for reproducible paper figures`

---

### Ticket 7.2 — Create `docs/METHODS.md`

**Action:** Create a new file `docs/METHODS.md` with exactly this content:

```markdown
# Methods

This document mirrors `docs/PREREGISTRATION.md` in prose form, suitable for
direct inclusion in a paper's Methods section.

## Data
Microsoft News (MIND) dataset (MIND-small and/or MIND-large variants).
Impressions are split 70/15/15 into train/val/test by impression ID, seeded
deterministically (see `src/data_pipeline/scm_builder.py`).

## Causal Model
A structural causal model is fit over user PCA features (32 dims), item
entity-embedding PCA features (32 dims), item title-embedding PCA features
(32 dims), category, sentiment, and dwell time, with treatment `A`
(exposure) and outcome `Y_diversity`. The average treatment effect (ATE) is
estimated via inverse propensity weighting and linear regression
(`src/causal_model/model.py`), and validated with placebo, data-subset, and
random-common-cause refutation tests (`src/causal_model/refutation.py`).

## Counterfactual Diversity Impact (CDI)
A DoWhy `StructuralCausalModel` (GCM) is fit over the same feature set,
using `HistGradientBoostingRegressor` mechanisms for `Y_diversity` and
`Y_click` (`fit_gcm_item_sensitive()`), chosen because linear mechanisms
were dominated by the 32 user-PCA features and produced near-constant CDI
scores (see `docs/RESEARCH_LOG.md`, 2026-06-17 entry). CDI is computed as
`E[Y_diversity | do(A=1, item features)]` and normalized via min-max
scaling within each step's candidate pool (`NewsRecommendEnv._min_max_cdi`).

## RL Agent
A Gymnasium environment (`NewsRecommendEnv`) with observation = user history
embedding (EMA-updated on click) + current diversity score, action = choice
among `K` candidate articles, reward = `w * click + (1 - w) * CDI`. Trained
with PPO (stable-baselines3) using the hyperparameters fixed in
`docs/PREREGISTRATION.md`.

## Evaluation Protocol
Offline replay evaluation on held-out test sessions
(`src/evaluation/metrics.py::replay_evaluate`), computing NDCG@10,
Precision@10, and Intra-List Diversity (ILD) against ground-truth clicks.
Baselines: Random (seeded), Popularity (train-set frequency), and
Logistic-CF (`src/baselines/logistic_cf.py`, a click-probability model
trained on the same feature set).

## Statistical Procedure
PPO and the Logistic-CF baseline are each evaluated across 5 independent
seeds (Random and Popularity are deterministic given a seeded RNG). For
each metric, a 95% bootstrap confidence interval (10,000 resamples) is
computed across the 5 per-seed means (`src/evaluation/metrics.py::bootstrap_ci`).
Paired t-tests compare PPO against each baseline at the seed level, with
Holm-Bonferroni correction applied across all comparisons reported in the
same table (`scripts/analyze_results.py`). Effect sizes are reported as
Cohen's d with standard-convention labels (negligible/small/medium/large;
`src/evaluation/metrics.py::cohens_d_label`).
```

**Verify:**
```bash
test -f docs/METHODS.md && echo "exists"
```
Expected: `exists`

**Commit message:** `docs: add docs/METHODS.md paper-ready methods writeup`

---

# FINAL CHECKLIST

Run this single command as the last step:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
ruff check src/ tests/
python3 -m pyflakes src/ | grep F821
```
Expected: all tests pass, `ruff check` reports no errors, and the `pyflakes` grep is empty. If `artifacts/final_results_table.csv` exists, also confirm `README.md` contains no `{{` placeholders and no `p=0.017` string (Ticket 6.2/6.3). At this point every ticket in this document has been completed and verified.

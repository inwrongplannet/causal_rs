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

from stable_baselines3 import PPO

from src.rl_agent.train_ppo import train_ppo
from src.evaluation.metrics import replay_evaluate, ndcg_at_k, precision_at_k, ild
from src.baselines.logistic_cf import train_logistic_cf, score_candidates

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

    print("Training Logistic-CF baseline (trained once — deterministic given "
          "a fixed seed, so it does not need retraining per PPO seed)...")
    fitted_cf = train_logistic_cf(train_df, seed=42)

    all_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

        checkpoint_zip = ARTIFACTS / "checkpoints" / f"ppo_multiseed_seed{seed}.zip"
        if checkpoint_zip.exists():
            print(f"Loading existing checkpoint from {checkpoint_zip}...")
            model = PPO.load(str(checkpoint_zip))
        else:
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

        print("Evaluating Logistic-CF...")
        # Logistic-CF is deterministic given seed=42 (fixed above) and does
        # not vary across PPO training seeds. We repeat the same result
        # under each seed label so scripts/analyze_results.py's per-seed
        # bootstrap logic works unmodified for every method, including this
        # one — this is intentional, not a bug: it correctly shows zero
        # across-run variance for a deterministic method.
        cf_df = _evaluate_logistic_cf_baseline(test_sessions, test_df, fitted_cf, K)
        all_results.append(_metrics_row(seed, "Logistic-CF", cf_df))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} result rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

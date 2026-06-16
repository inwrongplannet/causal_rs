"""Quick evaluation script — re-uses already-trained model with min-max CDI."""
import ast
import logging
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rl_agent.environment import NewsRecommendEnv
from src.evaluation.metrics import ndcg_at_k, precision_at_k, ild

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "checkpoints" / "ppo_causal_rs_w03"


def _to_array(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=np.float32)
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    raise TypeError(f"Unexpected type {type(val)}")


def build_sessions(df):
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


def evaluate_policy(policy, sessions, news_df, cdi_cache, K=10, label="policy"):
    results = []
    for i, s in enumerate(sessions):
        if policy is None:
            ranked = list(range(len(s.candidate_pool)))
            np.random.shuffle(ranked)
            rec_items = [s.candidate_pool[a] for a in ranked[:K]]
        else:
            K_train = int(policy.action_space.n)
            pool = s.candidate_pool
            chosen = list(np.random.choice(len(pool), K_train, replace=False)) if len(pool) > K_train else list(range(len(pool)))
            cand_items = [pool[a] for a in chosen]
            cand_embs = [s.candidates[0][a].title_emb for a in chosen]
            mock = type("Session", (), {
                "user_id": s.user_id,
                "initial_history_emb": s.initial_history_emb,
                "candidates": [[type("C", (), {"item_id": iid, "title_emb": emb})()
                               for iid, emb in zip(cand_items, cand_embs)]],
                "clicks": [[s.clicks[0][a] for a in chosen]],
                "candidate_pool": cand_items,
                "clicked_items": s.clicked_items,
            })()
            env = NewsRecommendEnv([mock], news_df, cdi_cache, w=0.6, K=K_train, T=1)
            obs, _ = env.reset()
            with torch.no_grad():
                logits = policy.policy.get_distribution(
                    torch.as_tensor(obs[None], dtype=torch.float32)
                ).distribution.logits[0]
            ranked = np.argsort(logits.detach().cpu().numpy())[::-1]
            rec_items = [cand_items[a] for a in ranked[:K]]
            env.close()

        rec_embs = []
        for iid in rec_items:
            try:
                rec_embs.append(_to_array(news_df.loc[iid, "I_title_emb_full"]))
            except KeyError:
                rec_embs.append(np.zeros(768, dtype=np.float32))

        results.append({
            "ndcg": ndcg_at_k(rec_items, s.clicked_items, K),
            "precision": precision_at_k(rec_items, s.clicked_items, K),
            "ild": ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0,
        })
        if (i + 1) % 200 == 0:
            interim = pd.DataFrame(results)
            print(f"  [{label}] {i+1}/{len(sessions)} — "
                  f"NDCG: {interim['ndcg'].mean():.4f}, "
                  f"Prec: {interim['precision'].mean():.4f}, "
                  f"ILD: {interim['ild'].mean():.4f}")
    return pd.DataFrame(results)


# ---- Load data ----
print("Loading data...")
train_df = pd.read_parquet(DATA / "scm_train.parquet")
test_df = pd.read_parquet(DATA / "scm_test.parquet")
cdi_cache = pickle.load(open(ARTIFACTS / "cdi_cache.pkl", "rb"))
test_sessions = build_sessions(test_df)
news_df = test_df[["item_id", "I_title_emb_full"]].drop_duplicates("item_id").set_index("item_id")

model = PPO.load(str(MODEL_PATH), device="cuda")
print(f"Model loaded. Test sessions: {len(test_sessions)}, News items: {len(news_df)}")

# ---- Evaluate ----
print("\n--- PPO ---")
ppo_df = evaluate_policy(model, test_sessions, news_df, cdi_cache, label="PPO")

print("\n--- Random ---")
rand_df = evaluate_policy(None, test_sessions, news_df, cdi_cache, label="Random")

print("\n--- Popularity ---")
pop_counter = Counter(train_df["item_id"])
pop_results = []
for i, s in enumerate(test_sessions):
    scored = sorted([(iid, pop_counter.get(iid, 0)) for iid in s.candidate_pool], key=lambda x: -x[1])
    rec_items = [s[0] for s in scored[:10]]
    rec_embs = []
    for iid in rec_items:
        try:
            rec_embs.append(_to_array(news_df.loc[iid, "I_title_emb_full"]))
        except KeyError:
            rec_embs.append(np.zeros(768))
    pop_results.append({
        "ndcg": ndcg_at_k(rec_items, s.clicked_items, 10),
        "precision": precision_at_k(rec_items, s.clicked_items, 10),
        "ild": ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0,
    })
    if (i + 1) % 200 == 0:
        interim = pd.DataFrame(pop_results)
        print(f"  [Popularity] {i+1}/{len(test_sessions)} — "
              f"NDCG: {interim['ndcg'].mean():.4f}, "
              f"Prec: {interim['precision'].mean():.4f}, "
              f"ILD: {interim['ild'].mean():.4f}")
pop_df = pd.DataFrame(pop_results)

# ---- Summary ----
print("\n" + "=" * 65)
print("AGGREGATED METRICS")
print("=" * 65)
print(f"{'Method':<18s} {'NDCG@10':<20s} {'Precision@10':<20s} {'ILD':<20s} {'n':>4s}")
print("-" * 65)
for label, df in [("PPO (Causal-RL)", ppo_df), ("Random", rand_df), ("Popularity", pop_df)]:
    print(f"{label:<18s} {df['ndcg'].mean():.4f} \u00b1 {df['ndcg'].std():.4f}  "
          f"{df['precision'].mean():.4f} \u00b1 {df['precision'].std():.4f}  "
          f"{df['ild'].mean():.4f} \u00b1 {df['ild'].std():.4f}  {len(df):>4d}")

print("\n" + "=" * 65)
print("SIGNIFICANCE TESTS (paired t-test)")
print("=" * 65)
for metric in ["ndcg", "precision", "ild"]:
    print(f"\n--- {metric.upper()} ---")
    t, p = stats.ttest_rel(ppo_df[metric], rand_df[metric])
    d = (ppo_df[metric].mean() - rand_df[metric].mean()) / (ppo_df[metric].std() + 1e-10)
    print(f"  PPO vs Random:     t={t:.3f}, p={p:.4f}, d={d:.3f}  {'SIGNIFICANT' if p<0.05 else 'n.s.'}")
    t, p = stats.ttest_rel(ppo_df[metric], pop_df[metric])
    d = (ppo_df[metric].mean() - pop_df[metric].mean()) / (ppo_df[metric].std() + 1e-10)
    print(f"  PPO vs Popularity: t={t:.3f}, p={p:.4f}, d={d:.3f}  {'SIGNIFICANT' if p<0.05 else 'n.s.'}")

print("\nEvaluation complete.")

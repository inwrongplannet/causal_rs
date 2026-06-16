"""Phase 4 (PPO training) + Phase 5 (evaluation) with min-max CDI normalization."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_agent.environment import NewsRecommendEnv
from src.rl_agent.train_ppo import train_ppo
from src.evaluation.metrics import ndcg_at_k, precision_at_k, ild

warnings.filterwarnings("ignore")
logging.getLogger("src.rl_agent").setLevel(logging.WARNING)
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "checkpoints" / "ppo_causal_rs_w03"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_array(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=np.float32)
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    raise TypeError(f"Embedding has unexpected type {type(val)}")


def build_sessions(df):
    sessions = []
    for imp_id, group in df.groupby("impression_id", sort=False):
        user_id = group["user_id"].iloc[0]
        history_emb = _to_array(group["U_history_emb_full"].iloc[0]).flatten()
        item_ids = group["item_id"].tolist()
        clicks = group["Y_click"].tolist()
        title_embs = group["I_title_emb_full"].apply(_to_array).tolist()
        clicked_items = set(group[group["Y_click"] == 1]["item_id"].tolist())
        candidates = [
            type("C", (), {"item_id": iid, "title_emb": emb})()
            for iid, emb in zip(item_ids, title_embs)
        ]
        session = type("Session", (), {
            "user_id": user_id,
            "initial_history_emb": history_emb,
            "candidates": [candidates],
            "clicks": [clicks],
            "candidate_pool": item_ids,
            "clicked_items": clicked_items,
        })()
        sessions.append(session)
    return sessions


# ---------------------------------------------------------------------------
# Phase 4 — PPO Training
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 4: PPO Training (min-max CDI)")
print("=" * 60)

print("\nLoading training data...")
train_df = pd.read_parquet(DATA / "scm_train.parquet")
print(f"  Rows: {len(train_df):,}, Users: {train_df['user_id'].nunique()}, "
      f"Impressions: {train_df['impression_id'].nunique()}")

cdi_path = ARTIFACTS / "cdi_cache.pkl"
with open(cdi_path, "rb") as f:
    cdi_cache = pickle.load(f)
print(f"  CDI cache: {len(cdi_cache)} entries")

print("\nBuilding train sessions...")
train_sessions = build_sessions(train_df)
print(f"  Sessions: {len(train_sessions)}")

cdi_user_items = set(cdi_cache.keys())
train_sessions = [
    s for s in train_sessions
    if any((s.user_id, iid) in cdi_user_items for iid in s.candidate_pool)
]
print(f"  Sessions with CDI coverage: {len(train_sessions)}")

K = 10
T = 1
total_timesteps = 200000
n_envs = min(2, len(train_sessions))

print(f"\nTraining PPO: {total_timesteps} timesteps, {n_envs} envs, "
      f"K={K}, T={T}, w=0.3, net_arch=[512,256]")
t0 = time.time()
model, save_path = train_ppo(
    train_sessions,
    train_df,
    cdi_cache,
    total_timesteps=total_timesteps,
    n_envs=n_envs,
    w=0.3,
    K=K,
    T=T,
    learning_rate=3e-4,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    tensorboard_log=str(ARTIFACTS / "tb_logs"),
    checkpoint_dir=str(ARTIFACTS / "checkpoints"),
    net_arch=[512, 256],
    verbose=0,
)
t_train = time.time() - t0
print(f"\nTraining completed in {t_train:.1f}s")
print(f"Model saved to: {save_path}")

# ---------------------------------------------------------------------------
# Phase 5 — Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 5: Evaluation")
print("=" * 60)

print("\nLoading test data...")
test_df = pd.read_parquet(DATA / "scm_test.parquet")
print(f"  Rows: {len(test_df):,}, Users: {test_df['user_id'].nunique()}, "
      f"Impressions: {test_df['impression_id'].nunique()}")

print("\nBuilding test sessions...")
test_sessions = build_sessions(test_df)
print(f"  Sessions: {len(test_sessions)}")

news_df = test_df[["item_id", "I_title_emb_full", "I_category"]].drop_duplicates("item_id")
news_df = news_df.set_index("item_id")
print(f"  News lookup: {len(news_df)} unique items")

model = PPO.load(str(MODEL_PATH), device="cuda")
print(f"\nPPO model loaded from {MODEL_PATH}.zip")

# --- Evaluation function ---
def evaluate_policy(policy, sessions, news_df, cdi_cache, w=0.6, K=10, T=1, label="policy"):
    results = []
    for i, session in enumerate(sessions):
        if policy is None:
            ranked = list(range(len(session.candidate_pool)))
            np.random.shuffle(ranked)
            rec_items = [session.candidate_pool[a] for a in ranked[:K]]
        else:
            K_train = int(policy.action_space.n)
            pool = session.candidate_pool
            if len(pool) > K_train:
                chosen = list(np.random.choice(len(pool), K_train, replace=False))
            else:
                chosen = list(range(len(pool)))
            cand_items = [pool[a] for a in chosen]
            cand_embs = [session.candidates[0][a].title_emb for a in chosen]
            cand_clicks = [session.clicks[0][a] for a in chosen]
            mock_sess = type("Session", (), {
                "user_id": session.user_id,
                "initial_history_emb": session.initial_history_emb,
                "candidates": [[
                    type("C", (), {"item_id": iid, "title_emb": emb})()
                    for iid, emb in zip(cand_items, cand_embs)
                ]],
                "clicks": [cand_clicks],
                "candidate_pool": cand_items,
                "clicked_items": session.clicked_items,
            })()
            env = NewsRecommendEnv([mock_sess], news_df, cdi_cache, w=w, K=K_train, T=T)
            obs, _ = env.reset()
            with torch.no_grad():
                dist = policy.policy.get_distribution(
                    torch.as_tensor(obs[None], dtype=torch.float32)
                )
                logits = dist.distribution.logits[0]
            ranked = np.argsort(logits.detach().cpu().numpy())[::-1]
            rec_items = [cand_items[a] for a in ranked[:K]]
            env.close()

        rec_embs = []
        for iid in rec_items:
            try:
                rec_embs.append(_to_array(news_df.loc[iid, "I_title_emb_full"]))
            except KeyError:
                rec_embs.append(np.zeros(768, dtype=np.float32))

        ndcg = ndcg_at_k(rec_items, session.clicked_items, K)
        prec = precision_at_k(rec_items, session.clicked_items, K)
        ild_val = ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0
        results.append({"ndcg": ndcg, "precision": prec, "ild": ild_val})

        if (i + 1) % 100 == 0:
            interim = pd.DataFrame(results)
            print(f"  [{label}] {i+1}/{len(sessions)} — "
                  f"NDCG: {interim['ndcg'].mean():.4f}, "
                  f"Prec: {interim['precision'].mean():.4f}, "
                  f"ILD: {interim['ild'].mean():.4f}")
    return results

# --- PPO ---
print("\nEvaluating PPO agent...")
t0 = time.time()
ppo_results = evaluate_policy(model, test_sessions, news_df, cdi_cache, label="PPO")
t_ppo = time.time() - t0
print(f"  PPO done in {t_ppo:.1f}s, n={len(ppo_results)}")

# --- Random ---
print("\nEvaluating Random baseline...")
t0 = time.time()
rand_results = evaluate_policy(None, test_sessions, news_df, cdi_cache, label="Random")
t_rand = time.time() - t0
print(f"  Random done in {t_rand:.1f}s, n={len(rand_results)}")

# --- Popularity ---
print("\nEvaluating Popularity baseline...")
pop_counter = Counter(train_df["item_id"])
pop_results = []
for i, session in enumerate(test_sessions):
    scored = [(iid, pop_counter.get(iid, 0)) for iid in session.candidate_pool]
    scored.sort(key=lambda x: -x[1])
    rec_items = [s[0] for s in scored[:K]]
    rec_embs = []
    for iid in rec_items:
        try:
            rec_embs.append(_to_array(news_df.loc[iid, "I_title_emb_full"]))
        except KeyError:
            rec_embs.append(np.zeros(768, dtype=np.float32))
    ndcg = ndcg_at_k(rec_items, session.clicked_items, K)
    prec = precision_at_k(rec_items, session.clicked_items, K)
    ild_val = ild(np.array(rec_embs)) if len(rec_embs) >= 2 else 0.0
    pop_results.append({"ndcg": ndcg, "precision": prec, "ild": ild_val})
    if (i + 1) % 100 == 0:
        interim = pd.DataFrame(pop_results)
        print(f"  [Popularity] {i+1}/{len(test_sessions)} — "
              f"NDCG: {interim['ndcg'].mean():.4f}, "
              f"Prec: {interim['precision'].mean():.4f}, "
              f"ILD: {interim['ild'].mean():.4f}")
print(f"  Popularity done, n={len(pop_results)}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def summarize(results):
    df = pd.DataFrame(results)
    return {
        "NDCG@K": f"{df['ndcg'].mean():.4f} ± {df['ndcg'].std():.4f}",
        "Precision@K": f"{df['precision'].mean():.4f} ± {df['precision'].std():.4f}",
        "ILD": f"{df['ild'].mean():.4f} ± {df['ild'].std():.4f}",
        "n": len(results),
    }

print("\n" + "=" * 60)
print("AGGREGATED METRICS")
print("=" * 60)
headers = ["Method", "NDCG@K", "Precision@K", "ILD", "n"]
rows = [headers]
if ppo_results:
    rows.append(["PPO (Causal-RL)"] + list(summarize(ppo_results).values()))
rows.append(["Random"] + list(summarize(rand_results).values()))
rows.append(["Popularity"] + list(summarize(pop_results).values()))

for row in rows:
    print(f"  {row[0]:20s}  {row[1]:20s}  {row[2]:20s}  {row[3]:20s}  {row[4]:>4s}")

print("\n" + "=" * 60)
print("SIGNIFICANCE TESTS (paired t-test)")
print("=" * 60)
ppo_df = pd.DataFrame(ppo_results)
rand_df = pd.DataFrame(rand_results)
pop_df = pd.DataFrame(pop_results)

for metric in ["ndcg", "precision", "ild"]:
    print(f"\n--- {metric.upper()} ---")
    t_stat, p_val = stats.ttest_rel(ppo_df[metric], rand_df[metric])
    d = (ppo_df[metric].mean() - rand_df[metric].mean()) / (ppo_df[metric].std() + 1e-10)
    print(f"  PPO vs Random:     t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

    t_stat, p_val = stats.ttest_rel(ppo_df[metric], pop_df[metric])
    d = (ppo_df[metric].mean() - pop_df[metric].mean()) / (ppo_df[metric].std() + 1e-10)
    print(f"  PPO vs Popularity: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

print("\nPhase 4 + Phase 5 complete.")

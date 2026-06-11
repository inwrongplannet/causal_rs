"""
Refit GCM on ALL 657K training rows, then compute CDI for ALL training sessions.

Steps:
  1. Load full scm_train.parquet (657K rows)
  2. Fit GCM on all rows
  3. Save GCM to artifacts/gcm_model_full.pkl
  4. Build sessions for all 3,500 impressions
  5. Compute CDI for ALL (user, item) pairs
  6. Save CDI cache to artifacts/cdi_cache_full.pkl
"""

import ast
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counterfactual.gcm_fit import fit_gcm
from src.counterfactual.precompute_cdi import precompute_cdi_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("refit")

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load training data")
print("=" * 60)
t0 = time.time()
df = pd.read_parquet(DATA / "scm_train.parquet")
print(f"  Shape: {df.shape}, users={df['user_id'].nunique()}, "
      f"impressions={df['impression_id'].nunique()}")
print(f"  Loaded in {time.time() - t0:.1f}s")

pca_cols = [c for c in df.columns if c.startswith("U_pca_")]
entity_cols = [c for c in df.columns if c.startswith("I_entity_pca_")]
title_cols = [c for c in df.columns if c.startswith("I_title_pca_")]
print(f"  U_pca: {len(pca_cols)}, I_entity_pca: {len(entity_cols)}, I_title_pca: {len(title_cols)}")

# ---------------------------------------------------------------------------
# 2. Fit GCM on ALL data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Fit GCM on ALL rows")
print("=" * 60)
t0 = time.time()
gcm_model = fit_gcm(df, pca_cols, entity_cols, title_cols)
t_gcm = time.time() - t0
print(f"  GCM fitted on {len(df)} rows in {t_gcm:.1f}s")

gcm_path = ARTIFACTS / "gcm_model_full.pkl"
gcm_path.parent.mkdir(parents=True, exist_ok=True)
with open(gcm_path, "wb") as f:
    pickle.dump(gcm_model, f)
print(f"  GCM saved to {gcm_path}")

# ---------------------------------------------------------------------------
# 3. Build sessions from ALL impressions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Build sessions from ALL impressions")
print("=" * 60)
t0 = time.time()

def _to_array(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=np.float32)
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    raise TypeError(f"Embedding has unexpected type {type(val)}")

sessions = []
# Pre-compute news lookup for CDI (category + sentiment + all PCA columns)
entity_cols_all = [c for c in df.columns if c.startswith("I_entity_pca_")]
title_cols_all = [c for c in df.columns if c.startswith("I_title_pca_")]
select_cols = ["item_id", "I_category", "I_sentiment"] + entity_cols_all + title_cols_all
news_df = df[select_cols].drop_duplicates("item_id")
news_df = news_df.set_index("item_id")
categories = sorted(df["I_category"].unique().tolist())
print(f"  News lookup: {len(news_df)} items, categories: {len(categories)}")

# Build session objects
for imp_id, group in df.groupby("impression_id", sort=False):
    user_id = group["user_id"].iloc[0]
    history_emb = _to_array(group["U_history_emb_full"].iloc[0])
    item_ids = group["item_id"].tolist()
    sessions.append(
        type("Session", (), {
            "user_id": user_id,
            "initial_history_emb": history_emb,
            "candidate_pool": item_ids,
        })()
    )
print(f"  Built {len(sessions)} sessions in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 4. Compute CDI for ALL sessions × ALL candidates
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Compute CDI for ALL sessions × ALL candidates")
print(f"  Sessions: {len(sessions)}, total candidate pairs: {sum(len(s.candidate_pool) for s in sessions)}")
print("=" * 60)

# Optimisation: index df by user_id for O(1) lookup
user_index = {}
for uid, row in df.groupby("user_id").first().iterrows():
    user_index[uid] = row
# Also need category_to_int
from src.counterfactual.precompute_cdi import category_to_int

# Manual CDI computation with progress
from tqdm import tqdm
from dowhy.gcm.fitting_sampling import PARENTS_DURING_FIT
from dowhy import gcm

cdi_cache = {}
parent_order = gcm_model.graph.nodes["Y_diversity"].get(
    PARENTS_DURING_FIT,
    sorted(gcm_model.graph.predecessors("Y_diversity")),
)
mech = gcm_model.causal_mechanism("Y_diversity")
n_draws = 50

t0 = time.time()
n_skipped = 0
n_total = sum(len(s.candidate_pool) for s in sessions)

for session in tqdm(sessions, desc="CDI"):
    if session.user_id not in user_index:
        n_skipped += 1
        continue
    user_row = user_index[session.user_id]

    # Get unique items in this session's pool (deduplication)
    unique_items = list(set(session.candidate_pool))

    for item_id in unique_items:
        if (session.user_id, item_id) in cdi_cache:
            continue
        try:
            item = news_df.loc[item_id]
        except KeyError:
            continue

        # Build parent DataFrame just like predict_diversity_counterfactual
        parent_df = pd.DataFrame([user_row], columns=parent_order)
        parent_df["A"] = 1
        parent_df["I_category"] = item["I_category"]
        parent_df["I_sentiment"] = item["I_sentiment"]

        # Override item PCA values with actual
        for col in entity_cols_all:
            if col in parent_df.columns:
                parent_df[col] = item[col]
        for col in title_cols_all:
            if col in parent_df.columns:
                parent_df[col] = item[col]

        parent_values = parent_df.to_numpy()
        noise = mech.draw_noise_samples(num_samples=n_draws)
        tiled = np.repeat(parent_values, n_draws, axis=0)
        evals = mech.evaluate(tiled, noise)
        cdi_cache[(session.user_id, item_id)] = float(np.mean(evals))

t_cdi = time.time() - t0
print(f"\n  CDI computed: {len(cdi_cache)} entries in {t_cdi:.1f}s ({t_cdi/max(1,len(cdi_cache)):.3f}s/pair)")
print(f"  Skipped sessions (user not in training data): {n_skipped}")

cdi_path = ARTIFACTS / "cdi_cache_full.pkl"
with open(cdi_path, "wb") as f:
    pickle.dump(cdi_cache, f)
print(f"  CDI cache saved to {cdi_path}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  GCM model:  {gcm_path} (fitted on {len(df)} rows in {t_gcm:.1f}s)")
print(f"  CDI cache:  {cdi_path} ({len(cdi_cache)} entries in {t_cdi:.1f}s)")
print(f"  Sessions:   {len(sessions)}")
print(f"  Coverage:   {len(cdi_cache)} / {n_total} possible pairs")
print("\nDone.")

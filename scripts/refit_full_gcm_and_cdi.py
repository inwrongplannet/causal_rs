"""
Refit GCM on ALL training rows, then compute CDI for ALL training sessions.

Supports both MIND-small (default) and MIND-large (pass --large or set
dataset=large in config.yaml).  Steps:
  1. Load full scm_train.parquet
  2. Fit GCM on all rows (sampled to 500K if MIND-large to avoid OOM)
  3. Save GCM to artifacts/gcm_model_full.pkl
  4. Build sessions for all impressions
  5. Compute CDI for ALL (user, item) pairs
  6. Save CDI cache to artifacts/cdi_cache_full.pkl
"""

import argparse
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

from src.counterfactual.gcm_fit import fit_gcm, fit_gcm_item_sensitive
from src.counterfactual.precompute_cdi import precompute_cdi_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("refit")

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

parser = argparse.ArgumentParser(description="Refit GCM and compute full CDI cache")
parser.add_argument("--large", action="store_true", help="Use MIND-large settings")
parser.add_argument("--gcm-sample", type=int, default=0,
                    help="Sample N rows for GCM fit (0 = use all)")
parser.add_argument("--skip-gcm", action="store_true",
                    help="Skip GCM fitting, load existing model from artifacts")
parser.add_argument("--cdi-sessions", type=int, default=5000,
                    help="Max sessions for CDI computation (default 5000)")
parser.add_argument("--cdi-checkpoint-interval", type=int, default=500,
                    help="Save CDI cache checkpoint every N sessions (default 500)")
args = parser.parse_args()

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

is_large = args.large or (len(df) > 1_500_000)
if is_large:
    print("  Detected MIND-large dataset")

pca_cols = [c for c in df.columns if c.startswith("U_pca_")]
entity_cols = [c for c in df.columns if c.startswith("I_entity_pca_")]
title_cols = [c for c in df.columns if c.startswith("I_title_pca_")]
print(f"  U_pca: {len(pca_cols)}, I_entity_pca: {len(entity_cols)}, I_title_pca: {len(title_cols)}")

# ---------------------------------------------------------------------------
# 2. Fit GCM (or load existing)
# ---------------------------------------------------------------------------
gcm_path = ARTIFACTS / "gcm_model_full.pkl"
if args.skip_gcm and gcm_path.exists():
    print("\n" + "=" * 60)
    print("STEP 2: Load existing GCM (--skip-gcm)")
    print("=" * 60)
    t0 = time.time()
    with open(gcm_path, "rb") as f:
        gcm_model = pickle.load(f)
    print(f"  GCM loaded from {gcm_path} in {time.time() - t0:.1f}s")
    t_gcm = 0
else:
    print("\n" + "=" * 60)
    print("STEP 2: Fit GCM")
    print("=" * 60)
    t0 = time.time()
    gcm_sample = args.gcm_sample
    if gcm_sample == 0:
        # Auto: use all rows for small, 500K sample for large
        gcm_sample = len(df) if not is_large else 500_000
    gcm_data = df.sample(n=min(gcm_sample, len(df)), random_state=42) if gcm_sample < len(df) else df
    gcm_model, n_draws = fit_gcm_item_sensitive(gcm_data, pca_cols, entity_cols, title_cols)
    t_gcm = time.time() - t0
    print(f"  GCM fitted on {len(gcm_data)} rows in {t_gcm:.1f}s (n_draws={n_draws})")
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
n_sessions_limit = args.cdi_sessions  # use CLI arg
session_checkpoint_interval = args.cdi_checkpoint_interval

# Try loading existing CDI cache
cdi_path = ARTIFACTS / "cdi_cache_full.pkl"
cdi_cache = {}
if cdi_path.exists():
    try:
        with open(cdi_path, "rb") as f:
            cdi_cache = pickle.load(f)
        print(f"  Loaded existing CDI cache: {len(cdi_cache)} entries")
    except Exception as e:
        print(f"  Could not load CDI cache: {e}")
        cdi_cache = {}
for imp_id, group in df.groupby("impression_id", sort=False):
    if n_sessions_limit and len(sessions) >= n_sessions_limit:
        break
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
print(f"  Sessions: {len(sessions)} (limited to {args.cdi_sessions}), "
      f"total candidate pairs: ~{sum(len(s.candidate_pool) for s in sessions[:max(1,args.cdi_sessions)])}")
print("=" * 60)

from src.counterfactual.precompute_cdi import precompute_cdi_cache, category_to_int

cdi_path = ARTIFACTS / "cdi_cache_full.pkl"

t0 = time.time()
cdi_cache = precompute_cdi_cache(
    gcm_model,
    sessions[:args.cdi_sessions],
    news_df,
    df,
    categories=categories,
    cache_path=cdi_path,
    n_draws=200,
)
t_cdi = time.time() - t0

print(f"\n  CDI computed: {len(cdi_cache)} entries in {t_cdi:.1f}s ({t_cdi/max(1,len(cdi_cache)):.3f}s/pair)")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  GCM model:  {gcm_path}")
if t_gcm > 0:
    print(f"  GCM fitting: {len(gcm_data)} rows in {t_gcm:.1f}s")
print(f"  CDI cache:  {cdi_path} ({len(cdi_cache)} entries in {t_cdi:.1f}s)")
print(f"  Sessions:   {len(sessions)}")
print(f"  Coverage:   {len(cdi_cache)} / {n_total} possible pairs")
print("\nDone.")

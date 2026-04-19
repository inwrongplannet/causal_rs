# causal_rs

## Current Problems and Bottlenecks (Post Latest Push)

- `notebooks/phase_1_data_pipeline_mind_small.ipynb` still has a high-memory interaction build path because each row stores full vectors (`U_history_emb_full`, `I_title_emb_full`, `I_entity_emb_full`), which causes large RAM pressure and heavy SSD paging on full runs.
- Negative sampling remains expensive: the current approach builds per-session candidate pools and repeatedly samples from large arrays, which becomes slow at MIND scale.
- Runtime is dominated by memory + disk I/O, not pure compute. CPU/GPU utilization can look low while RAM and SSD are near full because the workload is object-heavy and I/O-bound.
- Current `data/scm_parts` outputs can become very large (many parquet parts), and interrupted runs may leave partial artifacts that are expensive to validate and rerun.
- The notebook currently mixes long-running build logic and output inspection in one flow, which makes restarts/recovery harder after kernel crashes.

## Practical Next Steps

1. Normalize storage: keep interaction rows lightweight (IDs, treatment/outcomes, scalar features, PCA columns), and keep full embeddings in separate sidecar lookup tables keyed by `item_id` / `impression_id`.
2. Replace the current negative sampler with an index-based/rejection sampler that avoids rebuilding large candidate lists each session.
3. Keep chunked processing, but tune chunk size based on memory headroom to avoid swap thrashing.
4. Add strict stage checkpoints (news features -> streamed parts -> PCA/reduced exports -> report), so each stage can resume independently after failure.
5. Use reduced `NEG_RATIO` for full-pipeline stability runs, then increase only for final benchmark runs.
6. Ensure embedding inference uses GPU when available (helps feature generation), while still optimizing the CPU/I/O-heavy SCM build path.

## What Is Already In Place

- Phase 1 and Phase 2 notebook workflows exist:
  - `notebooks/phase_1_data_pipeline_mind_small.ipynb`
  - `notebooks/phase_2_causal_modeling.ipynb`
- `.gitignore` already excludes heavy generated parquet artifacts (`/data/scm_parts/` and parquet files under `data/`).

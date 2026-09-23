import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.rl_agent.environment import NewsRecommendEnv, cosine_similarity

ALPHA = 0.05  # Significance threshold used consistently across this module.


def ndcg_at_k(recommended: list, clicked: set, K: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: Ranked list of recommended item IDs.
        clicked: Set of item IDs that were clicked.
        K: Truncation depth.

    Returns:
        NDCG@K in [0, 1].
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:K])
        if item in clicked
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(clicked), K)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, clicked: set, K: int) -> float:
    """Compute Precision at K.

    Args:
        recommended: Ranked list of recommended item IDs.
        clicked: Set of item IDs that were clicked.
        K: Truncation depth.

    Returns:
        Precision@K in [0, 1].
    """
    return len(set(recommended[:K]) & clicked) / K


def ild(recommended_embeddings: np.ndarray) -> float:
    """Compute Intra-List Diversity (ILD) as mean pairwise 1 - cosine sim.

    Args:
        recommended_embeddings: Array of shape (N, D) with N item embeddings.

    Returns:
        ILD score in [0, 2] (higher = more diverse).
    """
    N = len(recommended_embeddings)
    if N < 2:
        return 0.0
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sim = cosine_similarity(
                recommended_embeddings[i].flatten(),
                recommended_embeddings[j].flatten(),
            )
            total += 1.0 - sim
    return total / (N * (N - 1) / 2)


def homogeneity_trend(
    session_history_embs: list, recommended_embs: list
) -> list:
    """Compute homogeneity (cosine sim) between history and each rec step.

    A healthy causal-RL system should show a DECREASING trend over steps
    (agent explores diverse content), while a CF baseline shows FLAT or
    INCREASING trend.

    Args:
        session_history_embs: List of history embeddings per step (1-D).
        recommended_embs: List of recommended item embeddings per step (1-D).

    Returns:
        List of cosine similarities, one per step.
    """
    return [
        float(cosine_similarity(h.flatten(), r.flatten()))
        for h, r in zip(session_history_embs, recommended_embs)
    ]


def compute_session_metrics(session, rec_lists, news_df, K=10):
    """Compute evaluation metrics for a single session.

    Args:
        session: Session object with .clicked_items attribute.
        rec_lists: List of recommended lists from each step.
        news_df: News DataFrame indexed by item_id with I_title_emb_full.
        K: Truncation depth for NDCG/Precision.

    Returns:
        Dict with ndcg, precision, and ild scores.
    """
    clicked = set(session.clicked_items)
    recommended = rec_lists[-1] if rec_lists else []

    recommended_embs = []
    for item_id in recommended:
        row = news_df.loc[item_id]
        emb = row["I_title_emb_full"] if "I_title_emb_full" in row else np.zeros(768)
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        recommended_embs.append(emb)

    metrics = {
        "ndcg": ndcg_at_k(recommended, clicked, K),
        "precision": precision_at_k(recommended, clicked, K),
        "ild": ild(np.array(recommended_embs)) if len(recommended_embs) >= 2 else 0.0,
    }
    return metrics


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
    for session in test_sessions:
        env = NewsRecommendEnv([session], news_df, cdi_cache, w=w, K=20, T=T)
        obs, _ = env.reset()
        rec_lists = []
        for step in range(min(T, len(session.candidates))):
            action, _ = policy.predict(obs, deterministic=True)
            device = next(policy.parameters()).device
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            dist = policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits
            ranked = torch.argsort(logits[0], descending=True)[:K].cpu().numpy()
            cands = session.candidates[step]
            rec_items = [cands[i].item_id for i in ranked]
            rec_lists.append(rec_items)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        results.append(compute_session_metrics(session, rec_lists, news_df, K))
    return aggregate_metrics(results)


def aggregate_metrics(results: list) -> dict:
    """Aggregate per-session metrics into summary statistics.

    Args:
        results: List of per-session metric dicts from
                 compute_session_metrics().

    Returns:
        Dict with mean and std for ndcg, precision, ild, and n_sessions.
    """
    df = pd.DataFrame(results)
    return {
        "ndcg_mean": float(df["ndcg"].mean()),
        "ndcg_std": float(df["ndcg"].std()),
        "precision_mean": float(df["precision"].mean()),
        "precision_std": float(df["precision"].std()),
        "ild_mean": float(df["ild"].mean()),
        "ild_std": float(df["ild"].std()),
        "n_sessions": len(results),
    }


def significance_test(
    causal_rl_scores: list, baseline_scores: list, metric_name: str
) -> bool:
    """Test statistical significance between two sets of scores.

    Uses paired t-test with Cohen's d effect size reporting.

    Args:
        causal_rl_scores: Scores from the causal-RL system.
        baseline_scores: Scores from a baseline system.
        metric_name: Name of the metric for display.

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

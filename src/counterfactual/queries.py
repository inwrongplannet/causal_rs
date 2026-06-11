import numpy as np
import pandas as pd
from dowhy import gcm


def category_to_int(category: str, categories: list = None) -> int:
    """Map a category string to an integer for the GCM.

    Uses an optional lookup list for consistent mapping, otherwise falls
    back to a hash-based conversion.

    Args:
        category: Category string (e.g. 'news', 'sports').
        categories: Optional ordered list of all known categories.

    Returns:
        Integer index for the category.
    """
    if categories is not None:
        try:
            return categories.index(category)
        except ValueError:
            return 0
    try:
        return int(hash(category) % 1000)
    except Exception:
        return 0


def predict_diversity_counterfactual(
    scm: gcm.StructuralCausalModel,
    user_row: pd.Series,
    new_item_category: str,
    new_item_sentiment: float,
    new_item_entity_pca: dict = None,
    new_item_title_pca: dict = None,
    categories: list = None,
) -> float:
    """Predict counterfactual Y_diversity under intervention.

    Answers: "What would the diversity score be if this user is shown an
    article with the given category and sentiment?"  Uses the fitted GCM's
    causal mechanism for Y_diversity to draw samples under an intervention
    on A, I_category, I_sentiment, I_entity_pca_*, and I_title_pca_* (if
    provided), while keeping observed user confounders at their factual
    values.

    The item PCA columns give the GCM item-level content features so
    different items in the same session produce different CDI scores.

    The parent column order must match the order used during :func:`gcm.fit`,
    which is ``sorted(graph.predecessors(node))``, stored in the graph's
    ``PARENTS_DURING_FIT`` attribute.

    Args:
        scm: Fitted StructuralCausalModel.
        user_row: Observed user state as a pandas Series (must include
                  all graph node columns).
        new_item_category: Category of the candidate article.
        new_item_sentiment: Sentiment score of the candidate article.
        new_item_entity_pca: Optional dict of {col_name: value} for the
                             candidate item's I_entity_pca_* features.
        new_item_title_pca: Optional dict of {col_name: value} for the
                            candidate item's I_title_pca_* features.
        categories: Optional list of known categories for consistent encoding.

    Returns:
        Expected Y_diversity under the intervention (float in [0, 1]).
    """
    from dowhy.gcm.fitting_sampling import PARENTS_DURING_FIT

    parent_order = scm.graph.nodes["Y_diversity"].get(
        PARENTS_DURING_FIT,
        sorted(scm.graph.predecessors("Y_diversity")),
    )

    # Build parent DataFrame with intervention values
    parent_df = pd.DataFrame([user_row], columns=parent_order)
    parent_df["A"] = 1
    parent_df["I_category"] = new_item_category
    parent_df["I_sentiment"] = new_item_sentiment

    # Override item PCA values with the candidate item's actual values
    for pca_dict in (new_item_entity_pca, new_item_title_pca):
        if pca_dict:
            for col, val in pca_dict.items():
                if col in parent_df.columns:
                    parent_df[col] = val

    parent_values = parent_df.to_numpy()
    mech = scm.causal_mechanism("Y_diversity")

    n_draws = 50
    noise = mech.draw_noise_samples(num_samples=n_draws)
    tiled = np.repeat(parent_values, n_draws, axis=0)
    evals = mech.evaluate(tiled, noise)
    return float(np.mean(evals))

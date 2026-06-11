import logging
import networkx as nx
import pandas as pd
from dowhy import gcm
from dowhy.gcm import auto

log = logging.getLogger(__name__)


def build_causal_graph(
    pca_columns: list,
    entity_pca_columns: list = None,
    title_pca_columns: list = None,
) -> nx.DiGraph:
    """Build the causal DAG for the news recommendation SCM.

    Creates a NetworkX DiGraph with fixed nodes (I_category, I_sentiment,
    U_dwell_mean, A, Y_diversity, Y_click) and dynamic PCA embedding nodes
    with edges following the tiered causal structure:
      Tier 1 (user/item features) -> Tier 2 (treatment A) -> Tier 3 (outcomes)

    Includes item-level I_entity_pca_* and I_title_pca_* columns as parents
    of outcomes so the GCM can discriminate between items with different
    semantic content.

    Args:
        pca_columns: List of PCA-reduced user embedding column names.
        entity_pca_columns: Optional list of item entity PCA column names.
        title_pca_columns: Optional list of item title PCA column names.

    Returns:
        A NetworkX DiGraph representing the causal structure.
    """
    g = nx.DiGraph()
    g.add_node("I_category")
    g.add_node("I_sentiment")
    g.add_node("U_dwell_mean")
    g.add_node("A")
    g.add_node("Y_diversity")
    g.add_node("Y_click")

    for col in pca_columns:
        g.add_node(col)
        g.add_edge(col, "A")
        g.add_edge(col, "Y_diversity")
        g.add_edge(col, "Y_click")

    for col_list in (entity_pca_columns or [], title_pca_columns or []):
        for col in col_list:
            g.add_node(col)
            g.add_edge(col, "Y_diversity")
            g.add_edge(col, "Y_click")

    g.add_edge("U_dwell_mean", "A")
    g.add_edge("I_category", "A")
    g.add_edge("I_category", "Y_diversity")
    g.add_edge("I_sentiment", "Y_diversity")
    g.add_edge("A", "Y_diversity")
    g.add_edge("A", "Y_click")

    return g


def _discover_entity_pca(df_train: pd.DataFrame) -> list:
    """Discover I_entity_pca_* columns in the training DataFrame."""
    return [c for c in df_train.columns if c.startswith("I_entity_pca_")]


def _discover_title_pca(df_train: pd.DataFrame) -> list:
    """Discover I_title_pca_* columns in the training DataFrame."""
    return [c for c in df_train.columns if c.startswith("I_title_pca_")]


def fit_gcm(
    df_train: pd.DataFrame,
    pca_columns: list,
    entity_pca_columns: list = None,
    title_pca_columns: list = None,
) -> gcm.StructuralCausalModel:
    """Build and fit a DoWhy StructuralCausalModel on training data.

    Constructs the causal graph from PCA columns and item-level semantic
    PCA columns (entity + title), auto-assigns causal mechanisms, and
    fits the SCM.

    The entity and title PCA columns give the GCM item-level content
    features beyond just I_category and I_sentiment, enabling the
    counterfactual query to produce discriminative CDI scores across
    items in the same session.

    Args:
        df_train: Training DataFrame with columns matching the causal graph
                  nodes (PCA columns + I_category, I_sentiment, etc.).
        pca_columns: List of PCA-reduced user embedding column names.
        entity_pca_columns: Optional list of item entity PCA column names.
                            Discovered automatically if None.
        title_pca_columns: Optional list of item title PCA column names.
                           Discovered automatically if None.

    Returns:
        A fitted gcm.StructuralCausalModel ready for counterfactual queries.
    """
    if entity_pca_columns is None:
        entity_pca_columns = _discover_entity_pca(df_train)
    if title_pca_columns is None:
        title_pca_columns = _discover_title_pca(df_train)
    causal_graph = build_causal_graph(pca_columns, entity_pca_columns, title_pca_columns)
    scm = gcm.StructuralCausalModel(causal_graph)
    auto.assign_causal_mechanisms(scm, df_train, override_models=True)

    for node in causal_graph.nodes:
        mechanism = scm.causal_mechanism(node)
        log.info("Node %s assigned mechanism: %s", node, type(mechanism).__name__)

    gcm.fit(scm, df_train)
    log.info("GCM fitted successfully on %d rows", len(df_train))
    return scm

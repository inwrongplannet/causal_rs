import networkx as nx

from src.counterfactual.gcm_fit import build_causal_graph
from src.counterfactual.queries import category_to_int


class TestBuildCausalGraph:
    def test_basic_structure(self):
        g = build_causal_graph(["U_pca_0", "U_pca_1"])
        assert isinstance(g, nx.DiGraph)
        assert "A" in g.nodes
        assert "Y_diversity" in g.nodes
        assert "Y_click" in g.nodes
        assert "I_category" in g.nodes
        assert "I_sentiment" in g.nodes
        assert "U_dwell_mean" in g.nodes
        assert "U_pca_0" in g.nodes
        assert "U_pca_1" in g.nodes

    def test_edges_from_pca(self):
        g = build_causal_graph(["U_pca_0"])
        assert ("U_pca_0", "A") in g.edges
        assert ("U_pca_0", "Y_diversity") in g.edges
        assert ("U_pca_0", "Y_click") in g.edges

    def test_edges_fixed(self):
        g = build_causal_graph([])
        assert ("U_dwell_mean", "A") in g.edges
        assert ("I_category", "A") in g.edges
        assert ("I_category", "Y_diversity") in g.edges
        assert ("I_sentiment", "Y_diversity") in g.edges
        assert ("A", "Y_diversity") in g.edges
        assert ("A", "Y_click") in g.edges

    def test_node_count(self):
        g = build_causal_graph(["U_pca_0", "U_pca_1", "U_pca_2"])
        assert len(g.nodes) == 9  # 6 fixed + 3 PCA


class TestCategoryToInt:
    def test_same_category_same_int(self):
        assert category_to_int("news") == category_to_int("news")

    def test_different_categories_different(self):
        assert category_to_int("news") != category_to_int("sports")

    def test_with_categories_list(self):
        cats = ["sports", "news", "lifestyle"]
        assert category_to_int("news", cats) == 1
        assert category_to_int("sports", cats) == 0

    def test_unknown_category_defaults(self):
        cats = ["sports", "news"]
        assert category_to_int("unknown", cats) == 0

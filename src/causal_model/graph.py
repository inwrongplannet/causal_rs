from typing import List


def build_causal_graph_gml(pca_columns: List[str]) -> str:
    u_pca_nodes = "\n  ".join(
        [f'node [id "{col}" label "{col}"]' for col in pca_columns]
    )
    u_pca_edges = "\n  ".join(
        [
            f'edge [source "{col}" target "A"]\n  edge [source "{col}" target "Y_click"]\n  edge [source "{col}" target "Y_diversity"]'
            for col in pca_columns
        ]
    )

    return f"""
graph [
  directed 1
  {u_pca_nodes}

  node [id "U_dwell_mean" label "U_dwell_mean"]
  node [id "I_category" label "I_category"]
  node [id "I_sentiment" label "I_sentiment"]
  node [id "A" label "A"]
  node [id "Y_diversity" label "Y_diversity"]
  node [id "Y_click" label "Y_click"]

  {u_pca_edges}

  edge [source "U_dwell_mean" target "A"]

  edge [source "I_category" target "A"]
  edge [source "I_category" target "Y_diversity"]

  edge [source "I_sentiment" target "Y_diversity"]

  edge [source "A" target "Y_diversity"]
  edge [source "A" target "Y_click"]
]
"""

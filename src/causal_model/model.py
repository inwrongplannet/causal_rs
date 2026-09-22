import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

import dowhy
from dowhy import CausalModel

warnings.filterwarnings("ignore")
logging.getLogger("dowhy").setLevel(logging.WARNING)
logging.getLogger("statsmodels").setLevel(logging.WARNING)


def create_causal_model(
    df: pd.DataFrame,
    treatment: str = "A",
    outcome: str = "Y_diversity",
    graph_gml: str = None,
    common_causes: list = None,
) -> CausalModel:
    return CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=graph_gml,
        common_causes=common_causes,
    )


def identify_effect(
    model: CausalModel,
    method_name: str = "maximal-adjustment",
    proceed_when_unidentifiable: bool = False,
):
    return model.identify_effect(
        proceed_when_unidentifiable=proceed_when_unidentifiable,
        method_name=method_name,
    )


def estimate_ate_ipw(model: CausalModel, estimand, **kwargs):
    params = {
        "weighting_scheme": "ips_weight",
        "min_ps_threshold": 0.05,
        "max_ps_threshold": 0.95,
    }
    params.update(kwargs)
    return model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate",
        method_params=params,
    )


def estimate_ate_linear(model: CausalModel, estimand):
    return model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
    )


def positivity_check(df: pd.DataFrame, common_causes: list, treatment: str = "A"):
    features = df[common_causes].copy()
    features = pd.get_dummies(features, drop_first=True)
    target = df[treatment]

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(features, target)

    df_copy = df.copy()
    df_copy["ps"] = ps_model.predict_proba(features)[:, 1]

    plt.figure(figsize=(8, 5))
    plt.hist(
        df_copy[df_copy[treatment] == 1]["ps"], bins=30, alpha=0.5,
        label=f"Treatment ({treatment}=1)", density=True,
    )
    plt.hist(
        df_copy[df_copy[treatment] == 0]["ps"], bins=30, alpha=0.5,
        label=f"Control ({treatment}=0)", density=True,
    )
    plt.title("Propensity Score Overlap (Positivity Check)")
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return df_copy["ps"]

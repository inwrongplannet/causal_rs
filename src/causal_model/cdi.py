def compute_cdi_batch(model, estimand, data):
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    cdi_map = {}
    ate_val = estimate.value
    for idx, row in data.iterrows():
        cdi_map[(row["user_id"], row["item_id"])] = ate_val
    return cdi_map

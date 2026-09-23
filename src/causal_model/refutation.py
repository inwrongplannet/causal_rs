def run_refutations(model, estimand, estimate, num_simulations: int = 10):
    results = {}

    try:
        refute_placebo = model.refute_estimate(
            estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=num_simulations,
        )
        results["placebo"] = refute_placebo
        print("\n1. Placebo Treatment:\n", refute_placebo)
        if abs(refute_placebo.new_effect) > 0.05:
            print("WARNING: Placebo effect is non-zero. The graph may be misspecified.")
    except Exception as e:  # noqa: BLE001
        print("Placebo refutation failed:", e)
        results["placebo"] = None

    try:
        refute_subset = model.refute_estimate(
            estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=num_simulations,
        )
        results["subset"] = refute_subset
        print("\n2. Data Subset:\n", refute_subset)
    except Exception as e:  # noqa: BLE001
        print("Data subset refutation failed:", e)
        results["subset"] = None

    try:
        refute_rcc = model.refute_estimate(
            estimand, estimate,
            method_name="random_common_cause",
            num_simulations=num_simulations,
        )
        results["random_common_cause"] = refute_rcc
        print("\n3. Random Common Cause:\n", refute_rcc)
    except Exception as e:  # noqa: BLE001
        print("Random common cause refutation failed:", e)
        results["random_common_cause"] = None

    return results

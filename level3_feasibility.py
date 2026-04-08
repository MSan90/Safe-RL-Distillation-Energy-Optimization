"""
LEVEL 3: Feasibility Analysis — Data-Driven Constraint Tuning

Determines xd_min so that RL operates in a physically reachable region.
Without this, episodes die in 1 step and RL learns nothing.

DESIGN DECISION:
    The key parameter is feasible_fraction, NOT "quantile" in the
    statistical sense. Its meaning is:

        feasible_fraction = 0.90  ->  "Choose xd_min such that ~90% of
                                       the HeatDuty operating range can
                                       achieve XD >= xd_min."

    Implementation:
        xd_min = np.quantile(xd_values, 1.0 - feasible_fraction)

        With feasible_fraction=0.90:
            np.quantile(..., 0.10)  ->  10th percentile of XD distribution
            -> ~90% of XD values lie ABOVE this threshold
            -> Agent has a large feasible region to learn in

    Recommended progression:
        - Start training:   feasible_fraction = 0.90 or 0.95 (easy)
        - Intermediate:     feasible_fraction = 0.80
        - Hard / fine-tune: feasible_fraction = 0.97 to 0.99

    For report / defense:
        "xd_min is chosen as the (1 - f)-th percentile of the surrogate's
         XD distribution, where f is the desired feasible fraction of the
         operating range. This ensures that the RL agent begins training
         in a region where the purity constraint is achievable for at
         least f x 100% of possible HeatDuty values."
"""
import numpy as np


def analyze_feasibility(surrogate,
                        n_samples=500,
                        feasible_fraction=0.90):
    """
    Sample the surrogate across the full HeatDuty range.
    Extract the distribution of achievable XD values.
    Choose xd_min so that feasible_fraction of the range is feasible.

    Args:
        surrogate:         Level 2 SurrogateModel instance
        n_samples:         Number of HeatDuty points to sample
        feasible_fraction: Target fraction of HeatDuty range that should
                           be feasible (i.e., achieve XD >= xd_min).

                           Examples:
                             0.90 -> 90% of range feasible (easy start)
                             0.95 -> 95% feasible (very easy)
                             0.80 -> 80% feasible (moderate)
                             0.50 -> 50% feasible (hard)

                           Must be in (0, 1).

    Returns:
        dict with xd_min, xd_max, xd_mean, heat_range,
             xd_distribution, achievable_pct, feasible_fraction
    """
    assert 0.0 < feasible_fraction < 1.0, \
        f"feasible_fraction must be in (0, 1), got {feasible_fraction}"

    # Sample surrogate across entire HeatDuty range
    heats = np.linspace(surrogate.heat_min, surrogate.heat_max, n_samples)
    xd_values = np.array([surrogate.predict(h)["XD_ss"] for h in heats])

    # Core calculation:
    # We want ~feasible_fraction of xd_values >= xd_min
    # Therefore: xd_min = (1 - feasible_fraction)-th percentile
    #
    # Example: feasible_fraction=0.90
    #   -> percentile = 0.10
    #   -> xd_min = 10th percentile of XD distribution
    #   -> ~90% of values are above xd_min
    percentile = 1.0 - feasible_fraction
    xd_min = float(np.quantile(xd_values, percentile))

    xd_max = float(np.max(xd_values))
    xd_mean = float(np.mean(xd_values))

    # Verification: actual achievable fraction
    achievable_count = int(np.sum(xd_values >= xd_min))
    achievable_pct = achievable_count / n_samples * 100

    # Report
    print(f"[FEASIBILITY] XD distribution over {n_samples} samples:")
    print(f"   XD min={xd_values.min():.4f}  mean={xd_mean:.4f}  max={xd_max:.4f}")
    print(f"")
    print(f"   Target feasible fraction: {feasible_fraction:.0%}")
    print(f"   -> Using percentile:      {percentile:.2f}"
          f" (i.e., {percentile:.0%}-th percentile)")
    print(f"   -> xd_min:                {xd_min:.4f}")
    print(f"   -> Actual achievable:     {achievable_pct:.1f}%"
          f" ({achievable_count}/{n_samples} points)")

    # Warnings
    if achievable_pct < 50:
        print(f"   WARNING: Only {achievable_pct:.1f}% feasible!")
        print(f"       Consider INCREASING feasible_fraction"
              f" (currently {feasible_fraction})")

    if achievable_pct > 99:
        print(f"   NOTE: {achievable_pct:.1f}% feasible — constraint is very loose.")
        print(f"       Consider DECREASING feasible_fraction for a harder challenge.")

    return {
        "xd_min": xd_min,
        "xd_max": xd_max,
        "xd_mean": xd_mean,
        "heat_range": (surrogate.heat_min, surrogate.heat_max),
        "xd_distribution": xd_values,
        "achievable_pct": achievable_pct,
        "feasible_fraction": feasible_fraction,
    }
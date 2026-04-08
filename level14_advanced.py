"""
LEVEL 14: Advanced Analysis -- PhD-Level Contributions

Sections:
    1. Pure Dead-Time Sweep (how much delay can the policy handle?)
    2. Constraint Tightening (XD spec from 0.95 to 0.99)
    3. Worst-Case Adversarial Shock Sequences
    4. Safety Shield (Action Projection Layer)
    5. Empirical Stability Analysis (Lyapunov-like convergence)
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================================================================
# SECTION 1: Pure Dead-Time Sweep
# ==================================================================

def run_deadtime_sweep(build_env_fn, surrogate, base_config, model,
                       deadtime_values=None, n_episodes=200, seed=42):
    """
    Train was done with dead_time=3. Test the SAME policy
    under increasing dead-time values to find the limit.

    Args:
        build_env_fn:    Function(surrogate, config) -> GymWrapper
        surrogate:       SurrogateModel
        base_config:     Baseline config dict
        model:           Trained SAC model
        deadtime_values: List of dead_time_steps to test
        n_episodes:      Episodes per value
        seed:            Random seed

    Returns:
        List of result dicts
    """
    if deadtime_values is None:
        deadtime_values = [0, 1, 3, 5, 8, 12, 16, 20]

    results = []
    print("\n" + "=" * 65)
    print("  DEAD-TIME SWEEP")
    print("  Testing policy tolerance to increasing transport delay")
    print("=" * 65)

    for dt_val in deadtime_values:
        config = base_config.copy()
        config["dead_time_steps"] = dt_val

        env_test = build_env_fn(surrogate, config)

        ep_rewards = []
        ep_violations = []
        ep_min_xds = []
        ep_lengths = []

        for ep in range(n_episodes):
            obs, _ = env_test.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_viol = 0
            xds = []

            for t in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_test.step(action)
                ep_reward += reward
                xds.append(info.get("xd", 0.0))
                if not info.get("safe", True):
                    ep_viol += 1
                if terminated or truncated:
                    break

            ep_rewards.append(ep_reward)
            ep_violations.append(ep_viol)
            ep_min_xds.append(np.min(xds) if len(xds) > 0 else 0.0)
            ep_lengths.append(len(xds))

        result = {
            "dead_time": dt_val,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "mean_min_xd": float(np.mean(ep_min_xds)),
            "total_violations": int(np.sum(ep_violations)),
            "mean_length": float(np.mean(ep_lengths)),
        }
        results.append(result)

        status = "OK" if result["total_violations"] == 0 else "VIOLATIONS"
        print("  dead_time=%2d | Reward: %8.2f +/- %5.2f | "
              "Min XD: %.4f | Violations: %d | Length: %.0f | %s" %
              (dt_val, result["mean_reward"], result["std_reward"],
               result["mean_min_xd"], result["total_violations"],
               result["mean_length"], status))

    # Find limit
    limit = None
    for r in results:
        if r["total_violations"] > 0 or r["mean_length"] < 150:
            limit = r["dead_time"]
            break

    if limit is not None:
        print("\n  DEAD-TIME LIMIT: %d steps" % limit)
        print("  Policy fails beyond dead_time=%d" % limit)
    else:
        print("\n  Policy survived all tested dead-time values")

    print("=" * 65)
    return results


def plot_deadtime_sweep(dt_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dead-Time Tolerance Sweep",
                 fontsize=14, fontweight="bold")

    dts = [r["dead_time"] for r in dt_results]
    rewards = [r["mean_reward"] for r in dt_results]
    min_xds = [r["mean_min_xd"] for r in dt_results]
    violations = [r["total_violations"] for r in dt_results]

    axes[0].plot(dts, rewards, "bo-", lw=2, markersize=8)
    axes[0].set_xlabel("Dead-Time (steps)")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs Dead-Time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dts, min_xds, "go-", lw=2, markersize=8)
    axes[1].axhline(0.95, color="red", ls="--", lw=1.5, label="Spec=0.95")
    axes[1].set_xlabel("Dead-Time (steps)")
    axes[1].set_ylabel("Mean Min XD")
    axes[1].set_title("Purity Floor vs Dead-Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    colors = ["steelblue" if v == 0 else "red" for v in violations]
    axes[2].bar(range(len(dts)), violations, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(dts)))
    axes[2].set_xticklabels([str(d) for d in dts])
    axes[2].set_xlabel("Dead-Time (steps)")
    axes[2].set_ylabel("Total Violations")
    axes[2].set_title("Violations vs Dead-Time")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_deadtime_sweep.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_deadtime_sweep.png")


# ==================================================================
# SECTION 2: Constraint Tightening
# ==================================================================

def run_constraint_tightening(build_env_fn, surrogate, base_config, model,
                              xd_specs=None, n_episodes=200, seed=42):
    """
    Test the policy under progressively tighter purity constraints.
    The policy was trained with XD_min=0.95. How high can we push it?

    Args:
        build_env_fn:  Function(surrogate, config) -> GymWrapper
        surrogate:     SurrogateModel
        base_config:   Baseline config dict
        model:         Trained SAC model
        xd_specs:      List of XD_min values to test
        n_episodes:    Episodes per spec
        seed:          Random seed

    Returns:
        List of result dicts
    """
    if xd_specs is None:
        xd_specs = [0.950, 0.960, 0.970, 0.980, 0.985, 0.990, 0.995]

    results = []
    print("\n" + "=" * 65)
    print("  CONSTRAINT TIGHTENING")
    print("  Testing policy under stricter purity requirements")
    print("=" * 65)

    for spec in xd_specs:
        config = base_config.copy()
        config["xd_min"] = spec

        env_test = build_env_fn(surrogate, config)

        ep_rewards = []
        ep_violations = []
        ep_energies = []
        ep_mean_xds = []
        ep_lengths = []

        for ep in range(n_episodes):
            obs, _ = env_test.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_viol = 0
            xds = []
            heats = []

            for t in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_test.step(action)
                ep_reward += reward
                xds.append(info.get("xd", 0.0))
                heats.append(info.get("heat_duty", 0.0))
                if not info.get("safe", True):
                    ep_viol += 1
                if terminated or truncated:
                    break

            ep_rewards.append(ep_reward)
            ep_violations.append(ep_viol)
            ep_mean_xds.append(np.mean(xds))
            ep_energies.append(np.mean(heats))
            ep_lengths.append(len(xds))

        result = {
            "xd_spec": spec,
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_xd": float(np.mean(ep_mean_xds)),
            "mean_energy": float(np.mean(ep_energies)),
            "total_violations": int(np.sum(ep_violations)),
            "mean_length": float(np.mean(ep_lengths)),
        }
        results.append(result)

        status = "OK" if result["total_violations"] == 0 else "VIOLATIONS"
        print("  Spec=%.3f | Reward: %8.2f | "
              "Mean XD: %.4f | Energy: %.2f | "
              "Violations: %d | Length: %.0f | %s" %
              (spec, result["mean_reward"],
               result["mean_xd"], result["mean_energy"],
               result["total_violations"],
               result["mean_length"], status))

    print("=" * 65)
    return results


def plot_constraint_tightening(ct_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Constraint Tightening Analysis",
                 fontsize=14, fontweight="bold")

    specs = [r["xd_spec"] for r in ct_results]
    rewards = [r["mean_reward"] for r in ct_results]
    energies = [r["mean_energy"] for r in ct_results]
    violations = [r["total_violations"] for r in ct_results]

    axes[0].plot(specs, rewards, "bo-", lw=2, markersize=8)
    axes[0].set_xlabel("XD Specification")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs Purity Requirement")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(specs, energies, "ro-", lw=2, markersize=8)
    axes[1].set_xlabel("XD Specification")
    axes[1].set_ylabel("Mean Energy (kW)")
    axes[1].set_title("Energy Cost vs Purity Requirement")
    axes[1].grid(True, alpha=0.3)

    colors = ["steelblue" if v == 0 else "red" for v in violations]
    axes[2].bar(range(len(specs)), violations, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(specs)))
    axes[2].set_xticklabels(["%.3f" % s for s in specs], rotation=45)
    axes[2].set_xlabel("XD Specification")
    axes[2].set_ylabel("Total Violations")
    axes[2].set_title("Violations vs Purity Requirement")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_constraint_tightening.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_constraint_tightening.png")


# ==================================================================
# SECTION 3: Worst-Case Adversarial Shock Sequence
# ==================================================================

def run_adversarial_sequence(env, model, sequences=None,
                             total_steps=200, seed=42):
    """
    Apply carefully designed worst-case shock sequences.
    These are NOT random -- they are engineered to maximally
    stress the controller.

    Args:
        env:        GymWrapper environment
        model:      Trained SAC model
        sequences:  List of named sequences, each a list of (step, shock_kw)
        total_steps: Max steps
        seed:       Random seed

    Returns:
        List of result dicts
    """
    if sequences is None:
        sequences = [
            ("Alternating", [(40, 200.0), (60, -200.0), (80, 200.0), (100, -200.0)]),
            ("Rapid Fire",  [(50, 150.0), (55, -150.0), (60, 150.0), (65, -150.0), (70, 150.0)]),
            ("Ramp Up",     [(40, 50.0), (60, 100.0), (80, 150.0), (100, 200.0)]),
            ("Slam Down",   [(50, -300.0)]),
            ("Double Slam", [(50, 300.0), (70, -300.0)]),
        ]

    results = []
    print("\n" + "=" * 65)
    print("  WORST-CASE ADVERSARIAL SHOCK SEQUENCES")
    print("=" * 65)

    for name, shocks in sequences:
        obs, _ = env.reset(seed=seed)
        core = env.core
        shock_dict = {s: v for s, v in shocks}

        data = {
            "xd": [], "heat_duty": [], "reward": [],
            "violation": [], "step": []
        }

        for t in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)

            if t in shock_dict:
                core.heat_duty = np.clip(
                    core.heat_duty + shock_dict[t],
                    core.surrogate.heat_min,
                    core.surrogate.heat_max
                )

            obs, reward, terminated, truncated, info = env.step(action)

            data["xd"].append(info["xd"])
            data["heat_duty"].append(info["heat_duty"])
            data["reward"].append(reward)
            data["violation"].append(0 if info.get("safe", True) else 1)
            data["step"].append(t)

            if terminated or truncated:
                break

        for k in data:
            data[k] = np.array(data[k])

        min_xd = float(np.min(data["xd"]))
        total_violations = int(data["violation"].sum())
        survived = len(data["xd"]) >= total_steps * 0.9

        result = {
            "name": name,
            "shocks": shocks,
            "min_xd": min_xd,
            "violations": total_violations,
            "episode_length": len(data["xd"]),
            "survived": survived,
            "trajectory": data,
        }
        results.append(result)

        status = "SURVIVED" if survived and total_violations == 0 else "FAILED"
        print("  %-16s | Min XD: %.4f | Violations: %d | "
              "Length: %d | %s" %
              (name, min_xd, total_violations,
               len(data["xd"]), status))

    print("=" * 65)
    return results


def plot_adversarial_sequences(adv_results, xd_target=0.95):
    n = len(adv_results)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Worst-Case Adversarial Shock Sequences",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    for i, r in enumerate(adv_results):
        traj = r["trajectory"]
        label = "%s (min=%.4f)" % (r["name"], r["min_xd"])
        axes[0].plot(traj["step"], traj["xd"], color=colors[i],
                     lw=1.5, label=label)
        axes[1].plot(traj["step"], traj["heat_duty"], color=colors[i],
                     lw=1.5, label=r["name"])

    axes[0].axhline(xd_target, color="black", ls=":", lw=1.5, label="Spec=0.95")
    axes[0].set_ylabel("XD (Purity)")
    axes[0].set_title("XD Under Adversarial Sequences")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("HeatDuty (kW)")
    axes[1].set_title("HeatDuty Control Under Adversarial Sequences")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_adversarial.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_adversarial.png")


# ==================================================================
# SECTION 4: Safety Shield (Action Projection)
# ==================================================================

def run_with_safety_shield(env, model, n_episodes=200,
                           total_steps=200, seed=42):
    """
    Run episodes with a safety shield that clips actions
    before they can cause constraint violations.

    The shield works by:
        1. Agent proposes action
        2. Shield simulates the next state using surrogate
        3. If predicted XD < threshold, reduce the action magnitude
        4. Execute the clipped action

    This is NOT retraining -- it is a post-hoc safety layer.

    Returns:
        dict comparing shielded vs unshielded performance
    """
    shield_threshold = 0.96  # Shield activates above spec (margin)

    print("\n" + "=" * 65)
    print("  SAFETY SHIELD ANALYSIS (Action Projection)")
    print("  Shield threshold: XD >= %.3f" % shield_threshold)
    print("=" * 65)

    # Run WITHOUT shield
    noshield_rewards = []
    noshield_violations = []
    noshield_xds = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_viol = 0
        xds = []

        for t in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            xds.append(info.get("xd", 0.0))
            if not info.get("safe", True):
                ep_viol += 1
            if terminated or truncated:
                break

        noshield_rewards.append(ep_reward)
        noshield_violations.append(ep_viol)
        noshield_xds.append(np.mean(xds))

    # Run WITH shield
    shield_rewards = []
    shield_violations = []
    shield_xds = []
    shield_interventions = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        core = env.core
        ep_reward = 0.0
        ep_viol = 0
        ep_interventions = 0
        xds = []

        for t in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            raw_delta = float(action[0]) * core.delta_heat_max

            # Shield: check if proposed action would reduce XD below threshold
            proposed_heat = np.clip(
                core.heat_duty + raw_delta,
                core.surrogate.heat_min,
                core.surrogate.heat_max
            )
            pred = core.surrogate.predict(proposed_heat)

            if pred["XD_ss"] < shield_threshold:
                # Find the maximum safe delta
                # Binary search for the best safe action
                safe_delta = 0.0
                lo = 0.0
                hi = abs(raw_delta)
                sign = 1.0 if raw_delta >= 0 else -1.0

                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    test_heat = np.clip(
                        core.heat_duty + sign * mid,
                        core.surrogate.heat_min,
                        core.surrogate.heat_max
                    )
                    test_pred = core.surrogate.predict(test_heat)
                    if test_pred["XD_ss"] >= shield_threshold:
                        lo = mid
                        safe_delta = mid
                    else:
                        hi = mid

                clipped_action = np.array([sign * safe_delta / core.delta_heat_max],
                                          dtype=np.float32)
                ep_interventions += 1
            else:
                clipped_action = action

            obs, reward, terminated, truncated, info = env.step(clipped_action)
            ep_reward += reward
            xds.append(info.get("xd", 0.0))
            if not info.get("safe", True):
                ep_viol += 1
            if terminated or truncated:
                break

        shield_rewards.append(ep_reward)
        shield_violations.append(ep_viol)
        shield_xds.append(np.mean(xds))
        shield_interventions.append(ep_interventions)

    result = {
        "noshield_mean_reward": float(np.mean(noshield_rewards)),
        "noshield_violations": int(np.sum(noshield_violations)),
        "noshield_mean_xd": float(np.mean(noshield_xds)),
        "shield_mean_reward": float(np.mean(shield_rewards)),
        "shield_violations": int(np.sum(shield_violations)),
        "shield_mean_xd": float(np.mean(shield_xds)),
        "shield_interventions": float(np.mean(shield_interventions)),
    }

    print("  WITHOUT Shield:")
    print("    Reward:     %.2f" % result["noshield_mean_reward"])
    print("    Violations: %d" % result["noshield_violations"])
    print("    Mean XD:    %.4f" % result["noshield_mean_xd"])
    print("  WITH Shield:")
    print("    Reward:     %.2f" % result["shield_mean_reward"])
    print("    Violations: %d" % result["shield_violations"])
    print("    Mean XD:    %.4f" % result["shield_mean_xd"])
    print("    Avg interventions per episode: %.1f" %
          result["shield_interventions"])
    print("=" * 65)

    return result


def plot_safety_shield(shield_result):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Safety Shield (Action Projection) Comparison",
                 fontsize=14, fontweight="bold")

    labels = ["No Shield", "With Shield"]
    w = 0.4

    # Reward
    vals = [shield_result["noshield_mean_reward"],
            shield_result["shield_mean_reward"]]
    colors = ["coral", "steelblue"]
    bars = axes[0].bar(range(2), vals, color=colors, alpha=0.8, width=w)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.1f" % b.get_height(), ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(range(2))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward Comparison")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Violations
    vals = [shield_result["noshield_violations"],
            shield_result["shield_violations"]]
    bars = axes[1].bar(range(2), vals, color=colors, alpha=0.8, width=w)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%d" % b.get_height(), ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Total Violations")
    axes[1].set_title("Safety Violations")
    axes[1].grid(True, alpha=0.3, axis="y")

    # XD
    vals = [shield_result["noshield_mean_xd"],
            shield_result["shield_mean_xd"]]
    bars = axes[2].bar(range(2), vals, color=colors, alpha=0.8, width=w)
    axes[2].axhline(0.95, color="red", ls="--", lw=1.5, label="Spec")
    for b in bars:
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(range(2))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Mean XD")
    axes[2].set_title("Purity Comparison")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_safety_shield.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_safety_shield.png")


# ==================================================================
# SECTION 5: Empirical Stability (Lyapunov-like)
# ==================================================================

def run_stability_analysis(env, model, n_episodes=200,
                           total_steps=200, seed=42):
    """
    Analyze convergence behavior of the control actions.
    If |delta_heat| converges to zero over time, the controller
    exhibits Lyapunov-like stability (empirically).

    Also tracks |XD - XD_target| as a Lyapunov candidate.

    Returns:
        dict with convergence trajectories
    """
    print("\n" + "=" * 65)
    print("  EMPIRICAL STABILITY ANALYSIS (Lyapunov-like)")
    print("=" * 65)

    all_delta_heat = []
    all_xd_error = []
    all_xd = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        core = env.core
        prev_heat = core.heat_duty

        ep_deltas = []
        ep_errors = []
        ep_xds = []

        for t in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            current_heat = info["heat_duty"]
            delta = abs(current_heat - prev_heat)
            xd_error = abs(info["xd"] - 0.95)

            ep_deltas.append(delta)
            ep_errors.append(xd_error)
            ep_xds.append(info["xd"])

            prev_heat = current_heat

            if terminated or truncated:
                break

        all_delta_heat.append(ep_deltas)
        all_xd_error.append(ep_errors)
        all_xd.append(ep_xds)

    # Pad to same length and average
    max_len = max(len(d) for d in all_delta_heat)
    padded_deltas = np.zeros((n_episodes, max_len))
    padded_errors = np.zeros((n_episodes, max_len))
    padded_xds = np.zeros((n_episodes, max_len))
    counts = np.zeros(max_len)

    for ep in range(n_episodes):
        length = len(all_delta_heat[ep])
        padded_deltas[ep, :length] = all_delta_heat[ep]
        padded_errors[ep, :length] = all_xd_error[ep]
        padded_xds[ep, :length] = all_xd[ep]
        counts[:length] += 1

    mean_deltas = np.sum(padded_deltas, axis=0) / np.maximum(counts, 1)
    mean_errors = np.sum(padded_errors, axis=0) / np.maximum(counts, 1)
    mean_xds = np.sum(padded_xds, axis=0) / np.maximum(counts, 1)

    # Check convergence: is delta_heat decreasing over time?
    first_quarter = mean_deltas[:max_len // 4]
    last_quarter = mean_deltas[3 * max_len // 4:]

    first_mean = float(np.mean(first_quarter)) if len(first_quarter) > 0 else 0.0
    last_mean = float(np.mean(last_quarter)) if len(last_quarter) > 0 else 0.0
    converged = last_mean < first_mean * 0.5

    print("  Mean |delta_heat| first quarter:  %.4f" % first_mean)
    print("  Mean |delta_heat| last quarter:   %.4f" % last_mean)
    print("  Reduction ratio:                  %.2f" %
          (last_mean / first_mean if first_mean > 0 else 0.0))
    print("  Empirically converged:            %s" %
          ("YES" if converged else "NO"))
    print("=" * 65)

    return {
        "mean_deltas": mean_deltas,
        "mean_errors": mean_errors,
        "mean_xds": mean_xds,
        "first_quarter_mean": first_mean,
        "last_quarter_mean": last_mean,
        "converged": converged,
        "max_len": max_len,
    }


def plot_stability_analysis(stab_result):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Empirical Stability Analysis (Lyapunov-like Behavior)",
                 fontsize=14, fontweight="bold")

    steps = np.arange(stab_result["max_len"])
    deltas = stab_result["mean_deltas"]
    errors = stab_result["mean_errors"]
    xds = stab_result["mean_xds"]

    # Smooth for visualization
    win = max(5, len(deltas) // 20)
    if len(deltas) >= win:
        smooth_d = np.convolve(deltas, np.ones(win) / win, mode="valid")
        smooth_e = np.convolve(errors, np.ones(win) / win, mode="valid")
    else:
        smooth_d = deltas
        smooth_e = errors

    # Plot 1: |delta_heat| over time
    axes[0].plot(deltas, color="lightblue", alpha=0.4, lw=0.8, label="Raw")
    axes[0].plot(np.arange(len(smooth_d)) + win // 2, smooth_d,
                 color="blue", lw=2.5, label="Smoothed")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("|Delta HeatDuty| (kW)")
    axes[0].set_title("Control Action Convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: |XD - 0.95| over time (Lyapunov candidate)
    axes[1].plot(errors, color="lightgreen", alpha=0.4, lw=0.8, label="Raw")
    axes[1].plot(np.arange(len(smooth_e)) + win // 2, smooth_e,
                 color="green", lw=2.5, label="Smoothed")
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("|XD - 0.95|")
    axes[1].set_title("Purity Error (Lyapunov Candidate)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Phase portrait (delta_heat vs XD)
    axes[2].scatter(xds[:len(deltas)], deltas, c=steps[:len(deltas)],
                    cmap="viridis", s=10, alpha=0.6)
    axes[2].set_xlabel("XD")
    axes[2].set_ylabel("|Delta HeatDuty| (kW)")
    axes[2].set_title("Phase Portrait (color = time)")
    axes[2].axvline(0.95, color="red", ls="--", lw=1.5, alpha=0.7, label="Spec")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label("Step")

    plt.tight_layout()
    plt.savefig("fig_stability.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_stability.png")
"""
LEVEL 13: Robustness Analysis -- Stress Testing the SAC Policy

Sections:
    1. Step disturbances of varying magnitudes
    2. Multi-disturbance (sequential shock) scenarios
    3. Disturbance timing sensitivity
    4. Observation noise injection
    5. Grace step sensitivity analysis
    6. Process noise on HeatDuty (actuator uncertainty)
    7. Edge-of-feasibility stress test
    8. Sustained shock test
    9. Extreme shock test (breaking point search)
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================================================================
# INTERNAL HELPER
# ==================================================================

def _compute_recovery_time(xd_post, target, window=5):
    """
    Count steps until XD stays >= target for 'window' consecutive steps.
    Returns -1 if never recovered.
    """
    count = 0
    for i, val in enumerate(xd_post):
        if val >= target:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    return -1


# ==================================================================
# SECTION 1: Step Disturbance Analysis
# ==================================================================

def run_step_disturbance(env, model, shock_kw, shock_step=50,
                         total_steps=200, seed=42):
    obs, _ = env.reset(seed=seed)
    core = env.core

    data = {
        "xd": [], "heat_duty": [], "reward": [], "violation": [],
        "ttop": [], "tmid": [], "tbottom": [], "step": []
    }

    for t in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)

        if t == shock_step:
            core.heat_duty = np.clip(
                core.heat_duty + shock_kw,
                core.surrogate.heat_min,
                core.surrogate.heat_max
            )

        obs, reward, terminated, truncated, info = env.step(action)

        data["xd"].append(info["xd"])
        data["heat_duty"].append(info["heat_duty"])
        data["reward"].append(reward)
        data["violation"].append(0 if info.get("safe", True) else 1)
        data["ttop"].append(info.get("ttop", 0.0))
        data["tmid"].append(info.get("tmid", 0.0))
        data["tbottom"].append(info.get("tbottom", 0.0))
        data["step"].append(t)

        if terminated or truncated:
            break

    for k in data:
        data[k] = np.array(data[k])

    xd_arr = data["xd"]
    xd_target = 0.95

    post_shock = xd_arr[shock_step:] if shock_step < len(xd_arr) else xd_arr
    min_xd = float(np.min(post_shock)) if len(post_shock) > 0 else float(np.min(xd_arr))
    recovery_steps = _compute_recovery_time(post_shock, xd_target, window=5)

    post_heat = data["heat_duty"][shock_step:] if shock_step < len(data["heat_duty"]) else data["heat_duty"]
    max_heat = float(np.max(post_heat)) if len(post_heat) > 0 else 0.0

    total_violations = int(data["violation"].sum())

    pre_shock_xd = float(np.mean(xd_arr[max(0, shock_step - 10):shock_step])) if shock_step > 0 else xd_arr[0]
    overshoot = float(np.max(post_shock) - pre_shock_xd) if len(post_shock) > 0 else 0.0

    return {
        "shock_kw": shock_kw,
        "min_xd": min_xd,
        "recovery_steps": recovery_steps,
        "max_heat_post": max_heat,
        "violations": total_violations,
        "overshoot": max(0.0, overshoot),
        "episode_length": len(xd_arr),
        "trajectory": data,
    }


def run_all_step_disturbances(env, model, shocks=None, shock_step=50, seed=42):
    if shocks is None:
        shocks = [50.0, 100.0, 200.0, -100.0]

    results = []
    print("\n" + "=" * 65)
    print("  STEP DISTURBANCE ANALYSIS")
    print("=" * 65)

    for shock in shocks:
        m = run_step_disturbance(env, model, shock_kw=shock,
                                 shock_step=shock_step, seed=seed)
        results.append(m)
        sign = "+" if shock >= 0 else ""
        print("  Shock %s%7.0f kW | Min XD: %.4f | "
              "Violations: %d | Recovery: %d steps | "
              "Overshoot: %.4f" %
              (sign, shock, m["min_xd"],
               m["violations"], m["recovery_steps"],
               m["overshoot"]))

    print("=" * 65)

    print("\n+----------------+----------+------------+----------------+-----------+")
    print("| Shock (kW)     |  Min XD  | Violations | Recovery Steps | Overshoot |")
    print("+----------------+----------+------------+----------------+-----------+")
    for m in results:
        sign = "+" if m["shock_kw"] >= 0 else ""
        rec = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
        print("| %s%10.0f    | %.4f   | %10d | %14s | %.4f    |" %
              (sign, m["shock_kw"], m["min_xd"],
               m["violations"], rec, m["overshoot"]))
    print("+----------------+----------+------------+----------------+-----------+")

    return results


# ==================================================================
# SECTION 2: Multi-Disturbance Scenario
# ==================================================================

def run_multi_disturbance(env, model, disturbances=None,
                          total_steps=200, seed=42):
    if disturbances is None:
        disturbances = [(50, 100.0), (100, -150.0), (150, 80.0)]

    obs, _ = env.reset(seed=seed)
    core = env.core

    data = {
        "xd": [], "heat_duty": [], "reward": [], "violation": [],
        "ttop": [], "tmid": [], "tbottom": [], "step": []
    }

    shock_dict = {s: v for s, v in disturbances}

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
        data["ttop"].append(info.get("ttop", 0.0))
        data["tmid"].append(info.get("tmid", 0.0))
        data["tbottom"].append(info.get("tbottom", 0.0))
        data["step"].append(t)

        if terminated or truncated:
            break

    for k in data:
        data[k] = np.array(data[k])

    total_violations = int(data["violation"].sum())
    min_xd = float(np.min(data["xd"]))
    collapsed = total_violations > 10 or len(data["xd"]) < total_steps * 0.5

    print("\n" + "=" * 65)
    print("  MULTI-DISTURBANCE SCENARIO")
    print("=" * 65)
    for step, shock in disturbances:
        sign = "+" if shock >= 0 else ""
        print("  Step %3d: %s%.0f kW" % (step, sign, shock))
    print("  --------------------------------")
    print("  Episode length:   %d" % len(data["xd"]))
    print("  Min XD:           %.4f" % min_xd)
    print("  Total violations: %d" % total_violations)
    print("  Collapsed:        %s" % ("YES" if collapsed else "NO"))
    print("=" * 65)

    return {
        "disturbances": disturbances,
        "trajectory": data,
        "min_xd": min_xd,
        "total_violations": total_violations,
        "collapsed": collapsed,
        "episode_length": len(data["xd"]),
    }


# ==================================================================
# SECTION 3: Disturbance Timing Sensitivity
# ==================================================================

def run_timing_sensitivity(env, model, shock_kw=100.0,
                           timing_steps=None, seed=42):
    if timing_steps is None:
        timing_steps = [10, 20, 50, 100, 150]

    results = []
    print("\n" + "=" * 65)
    print("  TIMING SENSITIVITY (Shock = +%.0f kW)" % shock_kw)
    print("=" * 65)

    for t_shock in timing_steps:
        m = run_step_disturbance(env, model, shock_kw=shock_kw,
                                 shock_step=t_shock, seed=seed)
        m["shock_step"] = t_shock
        results.append(m)
        rec = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
        print("  Shock at step %3d | Min XD: %.4f | "
              "Recovery: %4s steps | Violations: %d" %
              (t_shock, m["min_xd"], rec, m["violations"]))

    print("=" * 65)
    return results


# ==================================================================
# SECTION 4: Observation Noise Robustness
# ==================================================================

def run_noise_robustness(env, model, noise_levels=None,
                         total_steps=200, n_episodes=200, seed=42):
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.002, 0.005, 0.01]

    results = []
    print("\n" + "=" * 65)
    print("  NOISE ROBUSTNESS ANALYSIS (Observation Noise)")
    print("=" * 65)

    for sigma in noise_levels:
        ep_rewards = []
        ep_violations = []
        ep_min_xds = []
        ep_lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_viol = 0
            xds = []

            for t in range(total_steps):
                noisy_obs = obs.copy()
                if sigma > 0:
                    noisy_obs[0] += np.random.normal(0, sigma)
                    noisy_obs[0] = np.clip(noisy_obs[0], 0.0, 1.0)

                action, _ = model.predict(noisy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += reward
                xds.append(info.get("xd", 0.0))
                if not info.get("safe", True):
                    ep_viol += 1

                if terminated or truncated:
                    break

            ep_rewards.append(ep_reward)
            ep_violations.append(ep_viol)
            ep_min_xds.append(np.min(xds))
            ep_lengths.append(len(xds))

        result = {
            "noise_sigma": sigma,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "mean_min_xd": float(np.mean(ep_min_xds)),
            "total_violations": int(np.sum(ep_violations)),
            "mean_episode_length": float(np.mean(ep_lengths)),
        }
        results.append(result)

        print("  sigma = %.4f | Reward: %8.2f +/- "
              "%.2f | Min XD: %.4f | "
              "Violations: %d" %
              (sigma, result["mean_reward"], result["std_reward"],
               result["mean_min_xd"], result["total_violations"]))

    print("=" * 65)
    return results


# ==================================================================
# SECTION 5: Grace Step Sensitivity
# ==================================================================

def run_grace_sensitivity(env, model, grace_values=None,
                          n_episodes=200, total_steps=200, seed=42):
    if grace_values is None:
        grace_values = [0, 2, 5, 10, 20]

    core = env.core
    original_grace = core.grace_steps

    results = []
    print("\n" + "=" * 65)
    print("  GRACE STEP SENSITIVITY")
    print("=" * 65)

    for g in grace_values:
        core.grace_steps = g

        ep_rewards = []
        ep_violations = []
        ep_lengths = []
        ep_min_xds = []

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

            ep_rewards.append(ep_reward)
            ep_violations.append(ep_viol)
            ep_lengths.append(len(xds))
            ep_min_xds.append(np.min(xds) if len(xds) > 0 else 0.0)

        result = {
            "grace_steps": g,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "mean_episode_length": float(np.mean(ep_lengths)),
            "mean_min_xd": float(np.mean(ep_min_xds)),
            "total_violations": int(np.sum(ep_violations)),
        }
        results.append(result)

        print("  grace=%2d | Reward: %8.2f +/- "
              "%.2f | Avg Length: %6.1f | "
              "Violations: %d" %
              (g, result["mean_reward"], result["std_reward"],
               result["mean_episode_length"],
               result["total_violations"]))

    core.grace_steps = original_grace
    print("  (Restored grace_steps = %d)" % original_grace)
    print("=" * 65)

    return results


# ==================================================================
# SECTION 6: Process Noise (HeatDuty Perturbation)
# ==================================================================

def run_process_noise_test(env, model, noise_kw_levels=None,
                           n_episodes=200, total_steps=200, seed=42):
    if noise_kw_levels is None:
        noise_kw_levels = [0.0, 1.0, 2.0, 5.0, 10.0]

    results = []
    print("\n" + "=" * 65)
    print("  PROCESS NOISE ANALYSIS (HeatDuty Perturbation)")
    print("=" * 65)

    for sigma_kw in noise_kw_levels:
        ep_rewards = []
        ep_violations = []
        ep_min_xds = []
        ep_lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            core = env.core
            ep_reward = 0.0
            ep_viol = 0
            xds = []

            rng = np.random.RandomState(seed + ep + int(sigma_kw * 1000))

            for t in range(total_steps):
                action, _ = model.predict(obs, deterministic=True)

                if sigma_kw > 0:
                    noise = rng.normal(0, sigma_kw)
                    core.heat_duty = np.clip(
                        core.heat_duty + noise,
                        core.surrogate.heat_min,
                        core.surrogate.heat_max,
                    )

                obs, reward, terminated, truncated, info = env.step(action)
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
            "noise_kw": sigma_kw,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "mean_min_xd": float(np.mean(ep_min_xds)),
            "total_violations": int(np.sum(ep_violations)),
            "mean_episode_length": float(np.mean(ep_lengths)),
        }
        results.append(result)

        print("  sigma = %5.1f kW | Reward: %8.2f +/- %5.2f | "
              "Min XD: %.4f | Violations: %d" %
              (sigma_kw, result["mean_reward"], result["std_reward"],
               result["mean_min_xd"], result["total_violations"]))

    print("=" * 65)
    return results


# ==================================================================
# SECTION 7: Edge-of-Feasibility Stress Test
# ==================================================================

def run_edge_stress_test(env, model, shocks=None,
                         shock_step=50, total_steps=200, seed=42):
    """
    Start the system near the purity constraint boundary (XD ~ 0.955)
    then apply disturbances. This forces the agent to operate where
    margin is minimal, revealing true transient behavior.
    """
    if shocks is None:
        shocks = [50.0, 100.0, 200.0, -50.0, -100.0]

    core = env.core
    surrogate = core.surrogate

    # Find heat_duty where XD_ss is closest to 0.955
    search_heats = np.linspace(surrogate.heat_min, surrogate.heat_max, 2000)
    best_heat = None
    best_diff = 999.0
    target_xd_start = 0.955

    for h in search_heats:
        pred = surrogate.predict(h)
        diff = abs(pred["XD_ss"] - target_xd_start)
        if diff < best_diff and pred["XD_ss"] >= 0.95:
            best_diff = diff
            best_heat = h

    if best_heat is None:
        for h in search_heats:
            pred = surrogate.predict(h)
            if pred["XD_ss"] >= 0.95:
                best_heat = h
                break

    if best_heat is None:
        print("[EDGE TEST] WARNING: Could not find heat near XD=0.955")
        print("[EDGE TEST] Using surrogate heat_min as fallback")
        best_heat = surrogate.heat_min

    start_pred = surrogate.predict(best_heat)
    print("\n" + "=" * 65)
    print("  EDGE-OF-FEASIBILITY STRESS TEST")
    print("=" * 65)
    print("  Starting point:")
    print("    HeatDuty = %.4f kW" % best_heat)
    print("    XD_ss    = %.4f (target was %.3f)" %
          (start_pred["XD_ss"], target_xd_start))
    print("    Margin above 0.95: %.4f" %
          (start_pred["XD_ss"] - 0.95))
    print("  This is the WORST case starting position.")
    print("=" * 65)

    results = []

    for shock in shocks:
        obs, _ = env.reset(seed=seed)

        # Override state to edge position
        core.heat_duty = best_heat
        pred = surrogate.predict(best_heat)
        core.xd = pred["XD_ss"]
        core.ttop = pred.get("TTOP_ss", 80.0)
        core.tmid = pred.get("TMID_ss", 100.0)
        core.tbottom = pred.get("TBOTTOM_ss", 120.0)
        core.prev_heat = best_heat
        core.step_count = 0
        core.violation_counter = 0
        core.dynamics.reset()

        data = {
            "xd": [], "heat_duty": [], "reward": [], "violation": [],
            "ttop": [], "tmid": [], "tbottom": [], "step": []
        }

        for t in range(total_steps):
            action, _ = model.predict(core._get_obs(), deterministic=True)

            if t == shock_step:
                core.heat_duty = np.clip(
                    core.heat_duty + shock,
                    surrogate.heat_min,
                    surrogate.heat_max
                )

            obs, reward, terminated, truncated, info = env.step(action)

            data["xd"].append(info["xd"])
            data["heat_duty"].append(info["heat_duty"])
            data["reward"].append(reward)
            data["violation"].append(0 if info.get("safe", True) else 1)
            data["ttop"].append(info.get("ttop", 0.0))
            data["tmid"].append(info.get("tmid", 0.0))
            data["tbottom"].append(info.get("tbottom", 0.0))
            data["step"].append(t)

            if terminated or truncated:
                break

        for k in data:
            data[k] = np.array(data[k])

        xd_arr = data["xd"]
        post_shock = xd_arr[shock_step:] if shock_step < len(xd_arr) else xd_arr
        min_xd = float(np.min(post_shock)) if len(post_shock) > 0 else float(np.min(xd_arr))
        recovery = _compute_recovery_time(post_shock, 0.95, window=5)
        total_violations = int(data["violation"].sum())

        pre_shock_xd = float(np.mean(xd_arr[max(0, shock_step - 10):shock_step])) if shock_step > 0 else xd_arr[0]
        overshoot = float(np.max(post_shock) - pre_shock_xd) if len(post_shock) > 0 else 0.0

        m = {
            "shock_kw": shock,
            "min_xd": min_xd,
            "recovery_steps": recovery,
            "violations": total_violations,
            "overshoot": max(0.0, overshoot),
            "episode_length": len(xd_arr),
            "start_xd": start_pred["XD_ss"],
            "start_heat": best_heat,
            "trajectory": data,
        }
        results.append(m)

        sign = "+" if shock >= 0 else ""
        rec_str = str(recovery) if recovery >= 0 else "N/A"
        print("  Shock %s%7.0f kW | Min XD: %.4f | "
              "Recovery: %4s | Violations: %d | "
              "Overshoot: %.4f" %
              (sign, shock, min_xd, rec_str,
               total_violations, m["overshoot"]))

    print("=" * 65)
    return results


# ==================================================================
# SECTION 8: Sustained Shock Test
# ==================================================================

def run_sustained_shock_test(env, model, shock_kw=100.0,
                             hold_steps=20, shock_start=50,
                             total_steps=200, seed=42):
    """
    Apply a disturbance and HOLD it for hold_steps before removing.
    This simulates a prolonged process upset.
    """
    obs, _ = env.reset(seed=seed)
    core = env.core
    shock_end = shock_start + hold_steps

    data = {
        "xd": [], "heat_duty": [], "reward": [], "violation": [],
        "ttop": [], "tmid": [], "tbottom": [], "step": []
    }

    for t in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)

        if shock_start <= t < shock_end:
            core.heat_duty = np.clip(
                core.heat_duty + shock_kw / hold_steps,
                core.surrogate.heat_min,
                core.surrogate.heat_max
            )

        obs, reward, terminated, truncated, info = env.step(action)

        data["xd"].append(info["xd"])
        data["heat_duty"].append(info["heat_duty"])
        data["reward"].append(reward)
        data["violation"].append(0 if info.get("safe", True) else 1)
        data["ttop"].append(info.get("ttop", 0.0))
        data["tmid"].append(info.get("tmid", 0.0))
        data["tbottom"].append(info.get("tbottom", 0.0))
        data["step"].append(t)

        if terminated or truncated:
            break

    for k in data:
        data[k] = np.array(data[k])

    xd_arr = data["xd"]
    post_shock = xd_arr[shock_end:] if shock_end < len(xd_arr) else xd_arr
    during_shock = xd_arr[shock_start:shock_end] if shock_end <= len(xd_arr) else xd_arr[shock_start:]

    min_xd_during = float(np.min(during_shock)) if len(during_shock) > 0 else 0.0
    min_xd_after = float(np.min(post_shock)) if len(post_shock) > 0 else 0.0
    recovery = _compute_recovery_time(post_shock, 0.95, window=5)
    total_violations = int(data["violation"].sum())

    return {
        "shock_kw": shock_kw,
        "hold_steps": hold_steps,
        "min_xd_during": min_xd_during,
        "min_xd_after": min_xd_after,
        "recovery_steps": recovery,
        "violations": total_violations,
        "episode_length": len(xd_arr),
        "trajectory": data,
    }


def run_all_sustained_shocks(env, model, shock_configs=None, seed=42):
    if shock_configs is None:
        shock_configs = [
            (100.0, 5),
            (100.0, 20),
            (100.0, 50),
            (200.0, 20),
            (-150.0, 20),
        ]

    results = []
    print("\n" + "=" * 65)
    print("  SUSTAINED SHOCK TEST")
    print("  (Disturbance held for multiple steps)")
    print("=" * 65)

    for shock_kw, hold in shock_configs:
        m = run_sustained_shock_test(
            env, model, shock_kw=shock_kw,
            hold_steps=hold, shock_start=50,
            total_steps=200, seed=seed
        )
        results.append(m)

        sign = "+" if shock_kw >= 0 else ""
        rec_str = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
        print("  Shock %s%.0f kW x %2d steps | "
              "Min XD during: %.4f | after: %.4f | "
              "Recovery: %4s | Violations: %d" %
              (sign, shock_kw, hold,
               m["min_xd_during"], m["min_xd_after"],
               rec_str, m["violations"]))

    print("=" * 65)
    return results


# ==================================================================
# SECTION 9: Extreme Shock Test
# ==================================================================

def run_extreme_shocks(env, model, shocks=None, shock_step=50,
                       total_steps=200, seed=42):
    if shocks is None:
        shocks = [100.0, 200.0, 400.0, 600.0, 800.0,
                  -100.0, -200.0, -400.0]

    results = []
    print("\n" + "=" * 65)
    print("  EXTREME SHOCK TEST (Finding the breaking point)")
    print("=" * 65)

    for shock in shocks:
        m = run_step_disturbance(env, model, shock_kw=shock,
                                 shock_step=shock_step,
                                 total_steps=total_steps, seed=seed)
        results.append(m)

        sign = "+" if shock >= 0 else ""
        rec_str = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
        broke = "BROKE" if m["violations"] > 0 or m["recovery_steps"] < 0 else "OK"
        print("  Shock %s%7.0f kW | Min XD: %.4f | "
              "Recovery: %4s | Violations: %d | %s" %
              (sign, shock, m["min_xd"],
               rec_str, m["violations"], broke))

    breaking_point = None
    for m in results:
        if m["violations"] > 0 or m["recovery_steps"] < 0:
            breaking_point = m["shock_kw"]
            break

    if breaking_point is not None:
        print("\n  BREAKING POINT: %.0f kW" % breaking_point)
    else:
        print("\n  NO BREAKING POINT FOUND in tested range")
        print("  Policy survived all shocks up to %.0f kW" %
              max(abs(s) for s in shocks))

    print("=" * 65)
    return results


# ==================================================================
# PLOTTING
# ==================================================================

def plot_step_disturbances(step_results, xd_target=0.95):
    n = len(step_results)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Step Disturbance Analysis -- SAC Response",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    for i, m in enumerate(step_results):
        traj = m["trajectory"]
        sign = "+" if m["shock_kw"] >= 0 else ""
        label = "%s%.0f kW" % (sign, m["shock_kw"])
        axes[0].plot(traj["step"], traj["xd"], color=colors[i], lw=1.8, label=label)
        axes[1].plot(traj["step"], traj["heat_duty"], color=colors[i], lw=1.8, label=label)

    axes[0].axhline(xd_target, color="black", ls=":", lw=1.5, label="Spec=0.95")
    axes[0].axvline(50, color="gray", ls="--", lw=1, alpha=0.5, label="Shock applied")
    axes[0].set_ylabel("XD (Purity)")
    axes[0].set_title("XD Response After Step Disturbance")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.85, 1.01])

    axes[1].axvline(50, color="gray", ls="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("HeatDuty (kW)")
    axes[1].set_title("HeatDuty Response After Step Disturbance")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig6_step_disturbance.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig6_step_disturbance.png")


def plot_multi_disturbance(multi_result, xd_target=0.95):
    traj = multi_result["trajectory"]
    disturbances = multi_result["disturbances"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Multi-Disturbance Scenario -- Sequential Shocks",
                 fontsize=14, fontweight="bold")

    axes[0].plot(traj["step"], traj["xd"], "b-", lw=2, label="SAC")
    axes[0].axhline(xd_target, color="black", ls=":", lw=1.5, label="Spec=0.95")
    for step, shock in disturbances:
        sign = "+" if shock >= 0 else ""
        axes[0].axvline(step, color="red", ls="--", lw=1.2, alpha=0.7)
        axes[0].annotate("%s%.0f kW" % (sign, shock), xy=(step, 0.86),
                         fontsize=8, color="red", ha="center", fontweight="bold")
    axes[0].set_ylabel("XD")
    axes[0].set_title("XD Response Under Multiple Shocks")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.83, 1.01])

    axes[1].plot(traj["step"], traj["heat_duty"], "b-", lw=2, label="HeatDuty")
    for step, shock in disturbances:
        axes[1].axvline(step, color="red", ls="--", lw=1.2, alpha=0.7)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("HeatDuty (kW)")
    axes[1].set_title("HeatDuty Control Under Multiple Shocks")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig7_multi_disturbance.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig7_multi_disturbance.png")


def plot_timing_sensitivity(timing_results):
    steps = [r["shock_step"] for r in timing_results]
    min_xds = [r["min_xd"] for r in timing_results]
    recoveries = [r["recovery_steps"] if r["recovery_steps"] >= 0 else 50
                  for r in timing_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Disturbance Timing Sensitivity",
                 fontsize=14, fontweight="bold")

    axes[0].bar(range(len(steps)), min_xds, color="steelblue", alpha=0.8)
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(["Step %d" % s for s in steps])
    axes[0].set_ylabel("Min XD After Shock")
    axes[0].set_title("Minimum XD by Shock Timing")
    axes[0].axhline(0.95, color="red", ls="--", lw=1.5, label="Spec")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(steps)), recoveries, color="coral", alpha=0.8)
    axes[1].set_xticks(range(len(steps)))
    axes[1].set_xticklabels(["Step %d" % s for s in steps])
    axes[1].set_ylabel("Recovery Steps")
    axes[1].set_title("Recovery Time by Shock Timing")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig8_timing_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig8_timing_sensitivity.png")


def plot_noise_robustness(noise_results):
    sigmas = [r["noise_sigma"] for r in noise_results]
    rewards = [r["mean_reward"] for r in noise_results]
    min_xds = [r["mean_min_xd"] for r in noise_results]
    violations = [r["total_violations"] for r in noise_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Observation Noise Robustness Analysis",
                 fontsize=14, fontweight="bold")

    axes[0].plot(sigmas, rewards, "bo-", lw=2, markersize=8)
    axes[0].set_xlabel("Noise sigma")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs Noise Level")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigmas, min_xds, "go-", lw=2, markersize=8)
    axes[1].axhline(0.95, color="red", ls="--", lw=1.5, label="Spec")
    axes[1].set_xlabel("Noise sigma")
    axes[1].set_ylabel("Mean Min XD")
    axes[1].set_title("Purity Floor vs Noise Level")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(sigmas)), violations, color="red", alpha=0.7)
    axes[2].set_xticks(range(len(sigmas)))
    axes[2].set_xticklabels(["%.3f" % s for s in sigmas])
    axes[2].set_xlabel("Noise sigma")
    axes[2].set_ylabel("Total Violations")
    axes[2].set_title("Safety Violations vs Noise Level")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig9_noise_robustness.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig9_noise_robustness.png")


def plot_grace_sensitivity(grace_results):
    graces = [r["grace_steps"] for r in grace_results]
    rewards = [r["mean_reward"] for r in grace_results]
    lengths = [r["mean_episode_length"] for r in grace_results]
    violations = [r["total_violations"] for r in grace_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Grace Step Sensitivity Analysis",
                 fontsize=14, fontweight="bold")

    axes[0].bar(range(len(graces)), rewards, color="steelblue", alpha=0.8)
    axes[0].set_xticks(range(len(graces)))
    axes[0].set_xticklabels(["g=%d" % g for g in graces])
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs Grace Steps")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(graces)), lengths, color="seagreen", alpha=0.8)
    axes[1].set_xticks(range(len(graces)))
    axes[1].set_xticklabels(["g=%d" % g for g in graces])
    axes[1].set_ylabel("Mean Episode Length")
    axes[1].set_title("Episode Survival vs Grace Steps")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(range(len(graces)), violations, color="coral", alpha=0.8)
    axes[2].set_xticks(range(len(graces)))
    axes[2].set_xticklabels(["g=%d" % g for g in graces])
    axes[2].set_ylabel("Total Violations")
    axes[2].set_title("Safety Violations vs Grace Steps")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig10_grace_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig10_grace_sensitivity.png")


def plot_process_noise(pn_results):
    sigmas = [r["noise_kw"] for r in pn_results]
    rewards = [r["mean_reward"] for r in pn_results]
    min_xds = [r["mean_min_xd"] for r in pn_results]
    violations = [r["total_violations"] for r in pn_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Process Noise Analysis (HeatDuty Perturbation)",
                 fontsize=14, fontweight="bold")

    axes[0].plot(sigmas, rewards, "bo-", lw=2, markersize=8)
    axes[0].set_xlabel("Process Noise Std Dev (kW)")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs Process Noise")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigmas, min_xds, "go-", lw=2, markersize=8)
    axes[1].axhline(0.95, color="red", ls="--", lw=1.5, label="Spec=0.95")
    axes[1].set_xlabel("Process Noise Std Dev (kW)")
    axes[1].set_ylabel("Mean Min XD")
    axes[1].set_title("Purity Floor vs Process Noise")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(sigmas)), violations, color="red", alpha=0.7)
    axes[2].set_xticks(range(len(sigmas)))
    axes[2].set_xticklabels(["%.1f" % s for s in sigmas])
    axes[2].set_xlabel("Process Noise Std Dev (kW)")
    axes[2].set_ylabel("Total Violations")
    axes[2].set_title("Safety Violations vs Process Noise")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_hardened3_process_noise.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_hardened3_process_noise.png")


def plot_edge_stress(edge_results, xd_target=0.95):
    n = len(edge_results)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Edge-of-Feasibility Stress Test (Start near XD=0.955)",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    for i, m in enumerate(edge_results):
        traj = m["trajectory"]
        sign = "+" if m["shock_kw"] >= 0 else ""
        label = "%s%.0f kW (min=%.4f)" % (sign, m["shock_kw"], m["min_xd"])
        axes[0].plot(traj["step"], traj["xd"], color=colors[i], lw=1.8, label=label)
        axes[1].plot(traj["step"], traj["heat_duty"], color=colors[i],
                     lw=1.8, label="%s%.0f kW" % (sign, m["shock_kw"]))

    axes[0].axhline(xd_target, color="black", ls=":", lw=1.5, label="Spec=0.95")
    axes[0].axvline(50, color="gray", ls="--", lw=1, alpha=0.5, label="Shock applied")
    axes[0].set_ylabel("XD (Purity)")
    axes[0].set_title("XD Response -- Starting Near Constraint Boundary")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(50, color="gray", ls="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("HeatDuty (kW)")
    axes[1].set_title("HeatDuty Control Response")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_edge_stress.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_edge_stress.png")


def plot_sustained_shocks(sustained_results, xd_target=0.95):
    n = len(sustained_results)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Sustained Shock Test (Disturbance held for N steps)",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    for i, m in enumerate(sustained_results):
        traj = m["trajectory"]
        sign = "+" if m["shock_kw"] >= 0 else ""
        label = "%s%.0f kW x %d steps" % (sign, m["shock_kw"], m["hold_steps"])
        axes[0].plot(traj["step"], traj["xd"], color=colors[i], lw=1.8, label=label)
        axes[1].plot(traj["step"], traj["heat_duty"], color=colors[i], lw=1.8, label=label)

    if len(sustained_results) > 0:
        axes[0].axvspan(50, 50 + sustained_results[0]["hold_steps"],
                        alpha=0.1, color="red", label="Shock window (first)")

    axes[0].axhline(xd_target, color="black", ls=":", lw=1.5, label="Spec=0.95")
    axes[0].set_ylabel("XD (Purity)")
    axes[0].set_title("XD Response Under Sustained Disturbance")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("HeatDuty (kW)")
    axes[1].set_title("HeatDuty Control Response")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_sustained_shock.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_sustained_shock.png")


def plot_extreme_shocks(extreme_results, xd_target=0.95):
    shocks = [m["shock_kw"] for m in extreme_results]
    min_xds = [m["min_xd"] for m in extreme_results]
    violations = [m["violations"] for m in extreme_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Extreme Shock Test -- Finding the Breaking Point",
                 fontsize=14, fontweight="bold")

    colors_bar = []
    for xd in min_xds:
        if xd >= xd_target:
            colors_bar.append("steelblue")
        else:
            colors_bar.append("red")

    labels = []
    for s in shocks:
        sign = "+" if s >= 0 else ""
        labels.append("%s%.0f" % (sign, s))

    axes[0].bar(range(len(shocks)), min_xds, color=colors_bar, alpha=0.8)
    axes[0].set_xticks(range(len(shocks)))
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_xlabel("Shock (kW)")
    axes[0].set_ylabel("Min XD")
    axes[0].set_title("Min XD After Extreme Shock")
    axes[0].axhline(xd_target, color="red", ls="--", lw=1.5, label="Spec=0.95")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    colors_v = []
    for v in violations:
        if v == 0:
            colors_v.append("steelblue")
        else:
            colors_v.append("red")

    axes[1].bar(range(len(shocks)), violations, color=colors_v, alpha=0.8)
    axes[1].set_xticks(range(len(shocks)))
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_xlabel("Shock (kW)")
    axes[1].set_ylabel("Violations")
    axes[1].set_title("Safety Violations After Extreme Shock")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_extreme_shocks.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_extreme_shocks.png")


# ==================================================================
# MASTER RUNNER -- Baseline Robustness (Sections 1-5)
# ==================================================================

def run_full_robustness_suite(env, model, seed=42):
    print("\n" + "=" * 65)
    print("  FULL ROBUSTNESS ANALYSIS SUITE")
    print("=" * 65)

    step_results = run_all_step_disturbances(
        env, model, shocks=[50.0, 100.0, 200.0, -100.0],
        shock_step=50, seed=seed
    )
    plot_step_disturbances(step_results)

    multi_result = run_multi_disturbance(
        env, model,
        disturbances=[(50, 100.0), (100, -150.0), (150, 80.0)],
        seed=seed
    )
    plot_multi_disturbance(multi_result)

    timing_results = run_timing_sensitivity(
        env, model, shock_kw=100.0,
        timing_steps=[10, 20, 50, 100, 150], seed=seed
    )
    plot_timing_sensitivity(timing_results)

    noise_results = run_noise_robustness(
        env, model, noise_levels=[0.0, 0.001, 0.002, 0.005, 0.01],
        n_episodes=200, seed=seed
    )
    plot_noise_robustness(noise_results)

    grace_results = run_grace_sensitivity(
        env, model, grace_values=[0, 2, 5, 10, 20],
        n_episodes=200, seed=seed
    )
    plot_grace_sensitivity(grace_results)

    print("\n" + "=" * 65)
    print("  ROBUSTNESS SUITE COMPLETE")
    print("  Generated figures: fig6 through fig10")
    print("=" * 65)

    return {
        "step_disturbances": step_results,
        "multi_disturbance": multi_result,
        "timing_sensitivity": timing_results,
        "noise_robustness": noise_results,
        "grace_sensitivity": grace_results,
    }
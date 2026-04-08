"""
LEVEL 10: Evaluation & Comparison

- Evaluates trained SAC policy vs PID baselines vs Random.
- Tracks spec violations (XD < 0.95) separately from hard safety violations
  (SafetyChecker including temperature constraints).
- Includes an OPEN-LOOP STEP RESPONSE TEST for clean dynamic validation
  (controller OFF, action fixed to zero).

Author intent:
    This file is for evaluation + validation utilities (not robustness stress tests).
"""

import numpy as np


# ======================================================================
#  Core Evaluation
# ======================================================================

def evaluate_policy(env, model=None, pid=None, n_episodes=200, seed=42, label="Policy"):
    """
    Evaluate a policy/controller on the environment.
    """
    total_rewards = []
    total_heat_duties = []
    total_spec_violations = []
    total_hard_violations = []
    episode_lengths = []
    xd_histories = []

    for ep in range(int(n_episodes)):
        obs, info0 = env.reset(seed=seed + ep)
        last_xd = float(info0.get("xd", 0.95))

        if pid is not None:
            pid.reset()

        ep_reward = 0.0
        ep_heat_duty = 0.0
        ep_spec_viol = 0
        ep_hard_viol = 0
        ep_xds = []
        step = 0

        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            elif pid is not None:
                action = pid.compute(last_xd)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            last_xd = float(info.get("xd", last_xd))

            ep_reward += float(reward)
            ep_heat_duty += abs(float(info.get("heat_duty", 0.0)))
            ep_xds.append(float(info.get("xd", 0.0)))

            # Spec violation: only XD >= 0.95
            if not bool(info.get("safe_spec", True)):
                ep_spec_viol += 1

            # Hard safety violation: SafetyChecker (XD + temperatures)
            if not bool(info.get("safe_hard", True)):
                ep_hard_viol += 1

            step += 1
            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        total_heat_duties.append(ep_heat_duty / max(step, 1))
        total_spec_violations.append(ep_spec_viol)
        total_hard_violations.append(ep_hard_viol)
        episode_lengths.append(step)
        xd_histories.append(float(np.mean(ep_xds)) if len(ep_xds) else float(last_xd))

    metrics = {
        "label": label,
        "mean_reward": float(np.mean(total_rewards)) if len(total_rewards) else 0.0,
        "std_reward": float(np.std(total_rewards)) if len(total_rewards) else 0.0,
        "mean_heat_duty": float(np.mean(total_heat_duties)) if len(total_heat_duties) else 0.0,
        "mean_energy": float(np.mean(total_heat_duties)) if len(total_heat_duties) else 0.0,  # backward compat
        "mean_xd": float(np.mean(xd_histories)) if len(xd_histories) else 0.0,
        "spec_violations": int(sum(total_spec_violations)),
        "hard_violations": int(sum(total_hard_violations)),
        "total_violations": int(sum(total_hard_violations)),  # backward compat
        "mean_episode_length": float(np.mean(episode_lengths)) if len(episode_lengths) else 0.0,
    }

    print(f"\n[EVAL] [{label}] over {n_episodes} episodes:")
    print(f"   Reward:              {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"   Avg HeatDuty:        {metrics['mean_heat_duty']:.2f} kW")
    print(f"   Avg XD:              {metrics['mean_xd']:.4f}")
    print(f"   Spec Violations:     {metrics['spec_violations']}  (XD < 0.95)")
    print(f"   Hard Safety Viol.:   {metrics['hard_violations']}  (SafetyChecker)")
    print(f"   Avg Length:          {metrics['mean_episode_length']:.1f}")

    return metrics


def compare(env, trained_model, n_episodes=200, seed=42):
    """
    Head-to-head comparison: Random vs PID vs SAC
    """
    print("=" * 60)
    print("[COMPARE] HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    from level15_pid_baseline import build_default_pid, build_economic_pid
    pid_soft = build_default_pid(env)
    pid_econ = build_economic_pid(env)

    random_metrics = evaluate_policy(env, model=None, pid=None, n_episodes=n_episodes, seed=seed, label="Random")
    pid_soft_metrics = evaluate_policy(env, model=None, pid=pid_soft, n_episodes=n_episodes, seed=seed, label="PID-SoftSP(0.985)")
    pid_econ_metrics = evaluate_policy(env, model=None, pid=pid_econ, n_episodes=n_episodes, seed=seed, label="PID-EconomicTrim")
    sac_metrics = evaluate_policy(env, model=trained_model, pid=None, n_episodes=n_episodes, seed=seed, label="SAC")

    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print(f"   Reward improvement (SAC - Random):   {sac_metrics['mean_reward'] - random_metrics['mean_reward']:.2f}")
    print(f"   Reward improvement (SAC - PID-Soft): {sac_metrics['mean_reward'] - pid_soft_metrics['mean_reward']:.2f}")
    print(f"   Reward improvement (SAC - PID-Econ): {sac_metrics['mean_reward'] - pid_econ_metrics['mean_reward']:.2f}")

    print(f"   HeatDuty reduction (Random - SAC):   {random_metrics['mean_heat_duty'] - sac_metrics['mean_heat_duty']:.2f} kW")
    print(f"   HeatDuty reduction (PID-Soft - SAC): {pid_soft_metrics['mean_heat_duty'] - sac_metrics['mean_heat_duty']:.2f} kW")
    print(f"   HeatDuty reduction (PID-Econ - SAC): {pid_econ_metrics['mean_heat_duty'] - sac_metrics['mean_heat_duty']:.2f} kW")

    print(f"   Purity improvement (SAC - Random):   {sac_metrics['mean_xd'] - random_metrics['mean_xd']:.4f}")
    print(f"   Purity improvement (SAC - PID-Soft): {sac_metrics['mean_xd'] - pid_soft_metrics['mean_xd']:.4f}")
    print(f"   Purity improvement (SAC - PID-Econ): {sac_metrics['mean_xd'] - pid_econ_metrics['mean_xd']:.4f}")

    print(f"   Spec violations:  SAC={sac_metrics['spec_violations']}  "
          f"PID-Soft={pid_soft_metrics['spec_violations']}  "
          f"PID-Econ={pid_econ_metrics['spec_violations']}  "
          f"Random={random_metrics['spec_violations']}")
    print(f"   Hard safety viol.:SAC={sac_metrics['hard_violations']}  "
          f"PID-Soft={pid_soft_metrics['hard_violations']}  "
          f"PID-Econ={pid_econ_metrics['hard_violations']}  "
          f"Random={random_metrics['hard_violations']}")
    print("=" * 60)

    combined = {
        "sac_mean_reward": sac_metrics["mean_reward"],
        "sac_std_reward": sac_metrics["std_reward"],
        "sac_mean_energy": sac_metrics["mean_heat_duty"],
        "sac_mean_heat_duty": sac_metrics["mean_heat_duty"],
        "sac_mean_xd": sac_metrics["mean_xd"],
        "sac_spec_violations": sac_metrics["spec_violations"],
        "sac_hard_violations": sac_metrics["hard_violations"],

        "pid_mean_reward": pid_soft_metrics["mean_reward"],
        "pid_std_reward": pid_soft_metrics["std_reward"],
        "pid_mean_energy": pid_soft_metrics["mean_heat_duty"],
        "pid_mean_heat_duty": pid_soft_metrics["mean_heat_duty"],
        "pid_mean_xd": pid_soft_metrics["mean_xd"],
        "pid_spec_violations": pid_soft_metrics["spec_violations"],
        "pid_hard_violations": pid_soft_metrics["hard_violations"],

        "pid_econ_mean_reward": pid_econ_metrics["mean_reward"],
        "pid_econ_std_reward": pid_econ_metrics["std_reward"],
        "pid_econ_mean_energy": pid_econ_metrics["mean_heat_duty"],
        "pid_econ_mean_heat_duty": pid_econ_metrics["mean_heat_duty"],
        "pid_econ_mean_xd": pid_econ_metrics["mean_xd"],
        "pid_econ_spec_violations": pid_econ_metrics["spec_violations"],
        "pid_econ_hard_violations": pid_econ_metrics["hard_violations"],

        "random_mean_reward": random_metrics["mean_reward"],
        "random_std_reward": random_metrics["std_reward"],
        "random_mean_energy": random_metrics["mean_heat_duty"],
        "random_mean_heat_duty": random_metrics["mean_heat_duty"],
        "random_mean_xd": random_metrics["mean_xd"],
        "random_spec_violations": random_metrics["spec_violations"],
        "random_hard_violations": random_metrics["hard_violations"],
    }

    return combined


# ======================================================================
#  Open-loop step test (dynamic validation)
# ======================================================================

def run_open_loop_step_test(env, step_kw=50.0, step_at=30, total_steps=120, seed=42):
    """
    Open-loop step response test (controller OFF):
    - applies ONE step to core.heat_duty at step_at
    - holds HeatDuty constant afterward (action=0)
    - logs the resulting HeatDuty (and whatever info env returns)

    NOTE:
    - env safety/termination is still active; may terminate early.
    """
    obs, info0 = env.reset(seed=seed)
    core = env.core

    data = {
        "step": [],
        "heat_duty": [],
    }

    base_heat = float(core.heat_duty)
    step_applied = False

    for t in range(int(total_steps)):
        if (t == int(step_at)) and (not step_applied):
            core.heat_duty = float(np.clip(
                core.heat_duty + float(step_kw),
                core.surrogate.heat_min,
                core.surrogate.heat_max
            ))
            step_applied = True

        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        data["step"].append(int(t))
        data["heat_duty"].append(float(info.get("heat_duty", 0.0)))

        if terminated or truncated:
            break

    data["step"] = np.array(data["step"], dtype=np.int32)
    data["heat_duty"] = np.array(data["heat_duty"], dtype=np.float64)

    return {
        "base_heat": base_heat,
        "step_kw": float(step_kw),
        "step_at": int(step_at),
        "trajectory": data,
    }


def plot_open_loop_heatduty_only(result, save_path="fig_open_loop_heatduty_only.png"):
    """
    Plot only HeatDuty vs time (clean step input plot).
    """
    import matplotlib.pyplot as plt

    traj = result["trajectory"]
    step_at = int(result["step_at"])
    step_kw = float(result["step_kw"])
    base_heat = float(result["base_heat"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(traj["step"], traj["heat_duty"], lw=2, color="tab:blue", label="HeatDuty")
    ax.axvline(step_at, color="gray", ls="--", lw=1.2, label="Step applied")
    ax.axhline(base_heat, color="black", ls=":", lw=1.0, alpha=0.7, label="Base HeatDuty")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("HeatDuty (kW)")
    ax.set_title(f"Open-Loop Step Input in Reboiler Heat Duty (ΔHeat = {step_kw:+.0f} kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[STEP TEST] Saved: {save_path}")
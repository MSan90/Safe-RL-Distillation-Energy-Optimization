"""
MAIN PIPELINE -- Safe RL for Distillation Column
Levels 0-14 + Plant-Model Mismatch Analysis
Data -> Surrogate -> RL -> Evaluation -> Robustness -> Advanced -> Mismatch
Dynamic Model: FOPDT (First Order Plus Dead Time)
    Baseline: tau_xd=20, dead_time=3

SURROGATE OPTIONS:
    "interp"  ->  InterpSurrogate (linear interpolation, baseline)
    "mlp"     ->  MLPSurrogate (multi-output MLP, advanced)

Both surrogates share the same interface: predict(Q) -> dict
The rest of the pipeline is completely unchanged.
"""

import os, sys, importlib
import numpy as np

# NEW: ensure plt is available for the step-test helper functions
import matplotlib.pyplot as plt

# ── Configuration: Choose surrogate type ──
SURROGATE_TYPE = "mlp"   # "interp" or "mlp"

# -- Working Directory --
os.chdir(r"C:\Users\Lenovo\Downloads\final")
sys.path.insert(0, os.getcwd())

# -- Import All Modules --
import level1_data_layer, level2_surrogate, level3_feasibility
import level4_dynamics, level5_safety, level6_env_core
import level7_gym_wrapper, level8_sanity_checks, level9_training
import level10_evaluation, level11_reporting
import level13_robustness
import level14_advanced
import gain_mismatch_wrapper
import level15_pid_baseline  # NEW

# -- Reload --
for mod in [level1_data_layer, level2_surrogate, level3_feasibility,
            level4_dynamics, level5_safety, level6_env_core,
            level7_gym_wrapper, level8_sanity_checks, level9_training,
            level10_evaluation, level11_reporting, level13_robustness,
            level14_advanced, gain_mismatch_wrapper, level15_pid_baseline]:
    importlib.reload(mod)

from level1_data_layer import load_data, clean_data, data_sanity_gate
from level2_surrogate import InterpSurrogate, MLPSurrogate
from level3_feasibility import analyze_feasibility
from level4_dynamics import DynamicsBank
from level5_safety import SafetyChecker
from level6_env_core import SimpleDeltaHeatEnv
from level7_gym_wrapper import GymWrapper
from level8_sanity_checks import run_all_sanity_checks
from level9_training import train_sac
from level10_evaluation import compare
from level11_reporting import generate_report
from level13_robustness import (
    run_full_robustness_suite,
    run_all_step_disturbances,
    run_multi_disturbance,
    run_noise_robustness,
    plot_step_disturbances,
    plot_multi_disturbance,
    plot_noise_robustness,
    run_process_noise_test,
    plot_process_noise,
    run_edge_stress_test,
    run_all_sustained_shocks,
    run_extreme_shocks,
    plot_edge_stress,
    plot_sustained_shocks,
    plot_extreme_shocks,
)
from level14_advanced import (
    run_deadtime_sweep, plot_deadtime_sweep,
    run_constraint_tightening, plot_constraint_tightening,
    run_adversarial_sequence, plot_adversarial_sequences,
    run_with_safety_shield, plot_safety_shield,
    run_stability_analysis, plot_stability_analysis,
)
from gain_mismatch_wrapper import GainMismatchSurrogate, create_mismatch_suite


# ==================================================================
#  HELPER: Build environment from config dict
# ==================================================================
def build_env(surrogate, config):
    dynamics = DynamicsBank(
        tau_xd=config.get("tau_xd", 20.0),
        tau_ttop=config.get("tau_ttop", 5.0),
        tau_tmid=config.get("tau_tmid", 5.0),
        tau_tbottom=config.get("tau_tbottom", 5.0),
        dt=config.get("dt", 1.0),
        dead_time_steps=config.get("dead_time_steps", 3),
    )
    safety = SafetyChecker(
        xd_min=config.get("xd_min", 0.95),
        ttop_range=config.get("ttop_range", (50.0, 150.0)),
        tmid_range=config.get("tmid_range", (60.0, 180.0)),
        tbottom_range=config.get("tbottom_range", (80.0, 220.0)),
    )
    core = SimpleDeltaHeatEnv(
        surrogate=surrogate,
        dynamics_bank=dynamics,
        safety_checker=safety,
        max_steps=config.get("max_steps", 200),
        delta_heat_max=config.get("delta_heat_max", 50.0),
        reward_mode="monotonic",
    )
    return GymWrapper(core)


# ==================================================================
#  NEW: FOPDT step-test utilities (HeatDuty step -> XD response)
# ==================================================================
def run_fopdt_step_test(env, step_index=30, step_action=1.0, total_steps=120, seed=42):
    """
    Step test:
      - action=0 before step_index
      - action=step_action at step_index (one-step pulse)
      - action=0 after

    Records XD and HeatDuty from the env 'info' dict.
    """
    obs, info = env.reset(seed=seed)

    data = {
        "time": [],
        "xd": [],
        "heat_duty": [],
        "ttop": [],
        "tmid": [],
        "tbottom": [],
    }

    # t=0
    data["time"].append(0)
    data["xd"].append(info.get("xd", np.nan))
    data["heat_duty"].append(info.get("heat_duty", np.nan))
    data["ttop"].append(info.get("ttop", 0.0))
    data["tmid"].append(info.get("tmid", 0.0))
    data["tbottom"].append(info.get("tbottom", 0.0))

    for t in range(1, total_steps + 1):
        if t == step_index:
            action = np.array([step_action], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        data["time"].append(t)
        data["xd"].append(info.get("xd", np.nan))
        data["heat_duty"].append(info.get("heat_duty", np.nan))
        data["ttop"].append(info.get("ttop", 0.0))
        data["tmid"].append(info.get("tmid", 0.0))
        data["tbottom"].append(info.get("tbottom", 0.0))

        if terminated or truncated:
            print(f"[STEP TEST] Episode ended early at t={t}")
            break

    for k in data:
        data[k] = np.asarray(data[k], dtype=float)

    return data


def estimate_deadtime_and_tau(xd_history, step_index, threshold=None):
    """
    Rough FOPDT identification from XD trajectory.

    dead_time (θ):
      first t >= step_index where |XD(t) - XD0| > threshold

    tau (τ):
      first t >= dead_index where XD reaches 63.2% of total change,
      reported as (tau_index - dead_index)
    """
    xd_history = np.asarray(xd_history, dtype=float)

    xd0 = float(xd_history[step_index - 1]) if step_index > 0 else float(xd_history[0])
    xdf = float(np.nanmean(xd_history[-5:])) if len(xd_history) >= 5 else float(xd_history[-1])

    total_change = xdf - xd0
    if abs(total_change) < 1e-12:
        return {
            "xd0": xd0,
            "xdf": xdf,
            "dead_time": None,
            "tau": None,
            "dead_index": None,
            "tau_index": None,
        }

    if threshold is None:
        threshold = max(1e-6, 0.02 * abs(total_change))

    dead_index = None
    for t in range(step_index, len(xd_history)):
        if abs(xd_history[t] - xd0) > threshold:
            dead_index = t
            break

    xd63 = xd0 + 0.632 * total_change

    tau_index = None
    if dead_index is not None:
        if total_change > 0:
            for t in range(dead_index, len(xd_history)):
                if xd_history[t] >= xd63:
                    tau_index = t
                    break
        else:
            for t in range(dead_index, len(xd_history)):
                if xd_history[t] <= xd63:
                    tau_index = t
                    break

    dead_time = None if dead_index is None else int(dead_index - step_index)
    tau = None if (dead_index is None or tau_index is None) else int(tau_index - dead_index)

    return {
        "xd0": xd0,
        "xdf": xdf,
        "dead_time": dead_time,
        "tau": tau,
        "dead_index": dead_index,
        "tau_index": tau_index,
    }


def plot_fopdt_step_test(step_data, step_index=30, save_name="fig_fopdt_step_response.png"):
    time = np.asarray(step_data["time"], dtype=float)
    xd = np.asarray(step_data["xd"], dtype=float)
    heat = np.asarray(step_data["heat_duty"], dtype=float)

    est = estimate_deadtime_and_tau(xd, step_index=step_index)

    # zoom based on data
    xd_min = float(np.nanmin(xd))
    xd_max = float(np.nanmax(xd))
    span = max(1e-12, xd_max - xd_min)
    margin = max(1e-5, 0.10 * span)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("FOPDT Step Response of Distillation Column",
                 fontsize=14, fontweight="bold")

    # --------------------------------------------------------------
    # Top plot: HeatDuty
    # --------------------------------------------------------------
    axes[0].plot(time, heat, lw=2, label="HeatDuty")
    axes[0].axvline(step_index, color="red", linestyle="--", label="Step Time (ΔQ)")
    axes[0].set_ylabel("HeatDuty (kW)")
    axes[0].set_title("HeatDuty Step Input")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --------------------------------------------------------------
    # Bottom plot: XD response
    # --------------------------------------------------------------
    axes[1].plot(time, xd, lw=2, label="XD Response")
    axes[1].axvline(step_index, color="red", linestyle="--", label="Step Time (ΔQ)")

    if est["dead_index"] is not None:
        axes[1].axvline(est["dead_index"], color="green", linestyle="--", label="Dead Time (θ)")

    if est["tau_index"] is not None:
        axes[1].axvline(est["tau_index"], color="purple", linestyle="--", label="Time Constant (τ)")

    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("XD")
    axes[1].set_title("XD Response")
    axes[1].set_ylim(xd_min - margin, xd_max + margin)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    delta_xd = xd_max - xd_min
    txt = (
        f"Initial XD = {xd_min:.6f}\n"
        f"Final XD = {xd_max:.6f}\n"
        f"ΔXD = {delta_xd:.6f}\n"
        f"Dead Time (θ) = {est['dead_time']} steps\n"
        f"Time Constant (τ) = {est['tau']} steps"
    )
    axes[1].text(
        0.98, 0.05, txt,
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"[STEP TEST] Saved: {save_name}")
    print(f"[STEP TEST] Initial XD = {xd_min:.6f}")
    print(f"[STEP TEST] Final XD   = {xd_max:.6f}")
    print(f"[STEP TEST] Delta XD   = {delta_xd:.6f}")
    print(f"[STEP TEST] Dead Time (θ) = {est['dead_time']} steps")
    print(f"[STEP TEST] Time Const (τ) = {est['tau']} steps")


# ==================================================================
#  LEVEL 0: Header
# ==================================================================
print("=" * 65)
print("  SAFE RL FOR DISTILLATION COLUMN")
print("  Engineering Spec: XD >= 0.95")
print("  Dynamic Model: FOPDT (tau_xd=20, dead_time=3)")
print("  Surrogate Type: %s" % SURROGATE_TYPE.upper())
print("  Objective: Minimize HeatDuty | Maximize Purity | Zero Violations")
print("=" * 65)


# ==================================================================
#  LEVEL 1: Data
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 1: Data Layer")
print("-" * 65)

df = load_data(r"C:\Users\Lenovo\Desktop\Aspen_CSV.csv", sep=",", decimal=".")
df.columns = df.columns.str.strip()
df = df.rename(columns={"QN CAL/SEC": "HeatDuty"})
df["HeatDuty"] = df["HeatDuty"] * 0.004184
df = clean_data(df)
data_sanity_gate(df)
print("[Level 1] Data ready: %d rows" % len(df))


# ==================================================================
#  LEVEL 2: Surrogate (selectable)
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 2: Surrogate Model (%s)" % SURROGATE_TYPE.upper())
print("-" * 65)

if SURROGATE_TYPE == "mlp":
    surrogate = MLPSurrogate(
        df,
        hidden_layers=(64, 64),
        activation="relu",
        alpha=1e-4,
        max_iter=2000,
        random_state=42,
        test_size=0.2,
        bound_margin_pct=0.05,
    )
elif SURROGATE_TYPE == "interp":
    surrogate = InterpSurrogate(df)
else:
    raise ValueError("Unknown SURROGATE_TYPE: %s. Use 'interp' or 'mlp'." % SURROGATE_TYPE)

print("[Level 2] Surrogate ready (type=%s)" % SURROGATE_TYPE)


# ==================================================================
#  LEVEL 2.5: Surrogate Comparison (MLP vs Interp)
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 2.5: Surrogate Comparison (MLP vs Interpolation)")
print("-" * 65)

surrogate_interp = InterpSurrogate(df)
surrogate_mlp = MLPSurrogate(
    df,
    hidden_layers=(64, 64),
    activation="relu",
    alpha=1e-4,
    max_iter=2000,
    random_state=42,
    test_size=0.2,
    bound_margin_pct=0.05,
)

n_compare = 200
compare_heats = np.linspace(surrogate.heat_min, surrogate.heat_max, n_compare)
interp_preds = [surrogate_interp.predict(h) for h in compare_heats]
mlp_preds = [surrogate_mlp.predict(h) for h in compare_heats]

output_keys = ["XD_ss", "TTOP_ss", "TMID_ss", "TBOTTOM_ss"]

print("\n  Surrogate Comparison (RMSE between Interp and MLP):")
print("  " + "-" * 50)
for key in output_keys:
    interp_vals = np.array([p[key] for p in interp_preds])
    mlp_vals = np.array([p[key] for p in mlp_preds])
    rmse = np.sqrt(np.mean((interp_vals - mlp_vals) ** 2))
    max_diff = np.max(np.abs(interp_vals - mlp_vals))
    print("   %-12s  RMSE=%.6f  MaxDiff=%.6f" % (key, rmse, max_diff))

# Plot comparison
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Surrogate Comparison: Interpolation vs MLP",
             fontsize=14, fontweight="bold")

plot_labels = ["XD_ss", "TTOP_ss", "TMID_ss", "TBOTTOM_ss"]
plot_titles = ["XD (Purity)", "TTOP (Top Temp)", "TMID (Mid Temp)", "TBOTTOM (Bottom Temp)"]

for idx, (key, title) in enumerate(zip(plot_labels, plot_titles)):
    ax = axes[idx // 2][idx % 2]
    interp_vals = [p[key] for p in interp_preds]
    mlp_vals = [p[key] for p in mlp_preds]

    ax.plot(compare_heats, interp_vals, "b-", lw=2, label="Interpolation")
    ax.plot(compare_heats, mlp_vals, "r--", lw=2, label="MLP")

    col_name = key.replace("_ss", "")
    if col_name in df.columns:
        ax.scatter(df["HeatDuty"].values, df[col_name].values,
                   c="green", s=15, alpha=0.5, zorder=5, label="Data")

    if key == "XD_ss":
        ax.axhline(0.95, color="black", ls=":", lw=1.5, label="Spec=0.95")

    ax.set_xlabel("HeatDuty (kW)")
    ax.set_ylabel(col_name)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig0_surrogate_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig0_surrogate_comparison.png")

# MLP Test R² report plot
if SURROGATE_TYPE == "mlp":
    mlp_metrics = surrogate.get_test_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MLP Surrogate — Test Set Evaluation",
                 fontsize=14, fontweight="bold")

    cols = ["XD", "TTOP", "TMID", "TBOTTOM"]

    # R² bar chart
    train_r2 = [mlp_metrics["train_r2"][c] for c in cols]
    test_r2 = [mlp_metrics["test_r2"][c] for c in cols]
    x = np.arange(len(cols))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, train_r2, w, color="steelblue", alpha=0.8, label="Train R²")
    b2 = axes[0].bar(x + w / 2, test_r2, w, color="coral", alpha=0.8, label="Test R²")
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cols)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² per Output (Train vs Test)")
    axes[0].set_ylim([min(min(train_r2), min(test_r2)) - 0.02, 1.005])
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.8)
    axes[0].axhline(0.90, color="red", ls=":", lw=1, label="Threshold=0.90")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # RMSE bar chart
    train_rmse = [mlp_metrics["train_rmse"][c] for c in cols]
    test_rmse = [mlp_metrics["test_rmse"][c] for c in cols]
    b1 = axes[1].bar(x - w / 2, train_rmse, w, color="steelblue", alpha=0.8, label="Train RMSE")
    b2 = axes[1].bar(x + w / 2, test_rmse, w, color="coral", alpha=0.8, label="Test RMSE")
    for b in list(b1) + list(b2):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cols)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE per Output (Train vs Test)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig0b_mlp_test_metrics.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig0b_mlp_test_metrics.png")

print("[Level 2.5] Surrogate comparison complete")


# ==================================================================
#  LEVEL 3: Feasibility
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 3: Feasibility Analysis")
print("-" * 65)

feasibility = analyze_feasibility(surrogate, n_samples=500, feasible_fraction=0.90)
print("[Level 3] xd_min (data-based) = %.4f" % feasibility["xd_min"])
print("[Level 3] Achievable: %.1f%%" % feasibility["achievable_pct"])


# ==================================================================
#  ENGINEERING SPEC
# ==================================================================
XD_MIN_ENGINEERING = 0.95
print("\n>>> ENGINEERING SPEC: XD >= %s <<<" % XD_MIN_ENGINEERING)


# ==================================================================
#  LEVELS 4-7: Baseline Environment (FOPDT: tau_xd=20, dead_time=3)
# ==================================================================
print("\n" + "-" * 65)
print("  LEVELS 4-7: Baseline Environment (FOPDT)")
print("-" * 65)

BASELINE_CONFIG = {
    "tau_xd": 20.0,
    "tau_ttop": 5.0,
    "tau_tmid": 5.0,
    "tau_tbottom": 5.0,
    "dt": 1.0,
    "dead_time_steps": 3,
    "xd_min": XD_MIN_ENGINEERING,
    "max_steps": 200,
    "delta_heat_max": 50.0,
}

env = build_env(surrogate, BASELINE_CONFIG)
print("[Baseline] FOPDT environment ready (surrogate=%s)" % SURROGATE_TYPE)
print("   tau_xd=20, dead_time=3")
print("   Observation space: %s" % env.observation_space)
print("   Action space:      %s" % env.action_space)


# ==================================================================
#  LEVEL 8: Sanity Checks
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 8: Sanity Checks")
print("-" * 65)

sanity_ok = run_all_sanity_checks(env, n_random_steps=50)
if not sanity_ok:
    print("SANITY CHECKS FAILED -- Aborting!")
    sys.exit(1)
print("[Level 8] All sanity checks passed")


# ==================================================================
#  NEW: FOPDT STEP TEST (separate from training)
# ==================================================================
print("\n" + "-" * 65)
print("  FOPDT STEP TEST")
print("-" * 65)

env_step = build_env(surrogate, BASELINE_CONFIG)

step_data = run_fopdt_step_test(
    env_step,
    step_index=30,
    step_action=1.0,   # with delta_heat_max=50 => about +50 kW pulse (normalized)
    total_steps=120,
    seed=42,
)

plot_fopdt_step_test(
    step_data,
    step_index=30,
    save_name="fig_fopdt_step_response.png",
)


# ==================================================================
#  LEVEL 9: Training
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 9: SAC Training")
print("-" * 65)

model, training_rewards = train_sac(env, total_timesteps=200_000, seed=42)
print("[Level 9] Training complete -- %d episodes" % len(training_rewards))


# ==================================================================
#  LEVEL 10: Evaluation
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 10: Evaluation")
print("-" * 65)

results = compare(env, model, n_episodes=200, seed=42)
print("[Level 10] Evaluation complete")


# ==================================================================
#  LEVEL 11: Report
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 11: Report Generation")
print("-" * 65)

generate_report(
    results,
    xd_min_engineering=XD_MIN_ENGINEERING,
    feasibility=feasibility,
)
print("[Level 11] Report saved: final_report.txt")


# ==================================================================
#  LEVEL 12: Plots
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 12: Plotting")
print("-" * 65)

plot_namespace = {
    "env": env,
    "model": model,
    "results": results,
    "training_rewards": training_rewards,
    "np": np,
    "__name__": "__main__",
}

print("[Level 12] Generating 11 charts...")
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass

exec(open("level12_plots.py", "r", encoding="utf-8", errors="replace").read(), plot_namespace)
print("[Level 12] All 11 charts saved")


# ==================================================================
#  LEVEL 13: Robustness Analysis (Baseline FOPDT)
# ==================================================================
print("\n" + "-" * 65)
print("  LEVEL 13: Robustness Analysis (Baseline FOPDT)")
print("-" * 65)

robustness_results = run_full_robustness_suite(env, model, seed=42)
print("[Level 13] Baseline robustness complete")


# ==================================================================
#  HARDENED SCENARIO TESTS
# ==================================================================

print("\n")
print("=" * 65)
print("  HARDENED SCENARIO TESTS")
print("  (Same trained policy, harder environments)")
print("=" * 65)

import shutil
hardened_summary = {}

# ------------------------------------------------------------------
#  HARDENED TEST 1: Extended Dead-Time (dead_time=6)
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  HARDENED TEST 1: Extended Dead-Time (dead_time_steps=6)")
print("-" * 65)

DEADTIME_CONFIG = BASELINE_CONFIG.copy()
DEADTIME_CONFIG["dead_time_steps"] = 6

env_dt = build_env(surrogate, DEADTIME_CONFIG)

dt_step = run_all_step_disturbances(
    env_dt, model, shocks=[50.0, 100.0, 200.0, -100.0],
    shock_step=50, seed=42
)
plot_step_disturbances(dt_step, xd_target=0.95)
try:
    shutil.move("fig6_step_disturbance.png", "fig_hardened1_deadtime6_step.png")
    print("   Saved: fig_hardened1_deadtime6_step.png")
except Exception:
    pass

dt_multi = run_multi_disturbance(
    env_dt, model,
    disturbances=[(50, 100.0), (100, -150.0), (150, 80.0)],
    seed=42
)
plot_multi_disturbance(dt_multi, xd_target=0.95)
try:
    shutil.move("fig7_multi_disturbance.png", "fig_hardened1_deadtime6_multi.png")
    print("   Saved: fig_hardened1_deadtime6_multi.png")
except Exception:
    pass

hardened_summary["deadtime6"] = {
    "config": "dead_time_steps=6, tau_xd=20",
    "step_results": [
        {"shock": m["shock_kw"], "min_xd": m["min_xd"],
         "violations": m["violations"], "recovery": m["recovery_steps"]}
        for m in dt_step
    ],
    "multi_collapsed": dt_multi["collapsed"],
    "multi_min_xd": dt_multi["min_xd"],
    "multi_violations": dt_multi["total_violations"],
}

print("[Hardened 1] Extended dead-time test complete")


# ------------------------------------------------------------------
#  HARDENED TEST 2: Very Slow Composition (tau_xd=40)
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  HARDENED TEST 2: Very Slow Composition (tau_xd=40)")
print("-" * 65)

SLOWTAU_CONFIG = BASELINE_CONFIG.copy()
SLOWTAU_CONFIG["tau_xd"] = 40.0

env_st = build_env(surrogate, SLOWTAU_CONFIG)

st_step = run_all_step_disturbances(
    env_st, model, shocks=[50.0, 100.0, 200.0, -100.0],
    shock_step=50, seed=42
)
plot_step_disturbances(st_step, xd_target=0.95)
try:
    shutil.move("fig6_step_disturbance.png", "fig_hardened2_slowtau40_step.png")
    print("   Saved: fig_hardened2_slowtau40_step.png")
except Exception:
    pass

st_multi = run_multi_disturbance(
    env_st, model,
    disturbances=[(50, 100.0), (100, -150.0), (150, 80.0)],
    seed=42
)
plot_multi_disturbance(st_multi, xd_target=0.95)
try:
    shutil.move("fig7_multi_disturbance.png", "fig_hardened2_slowtau40_multi.png")
    print("   Saved: fig_hardened2_slowtau40_multi.png")
except Exception:
    pass

hardened_summary["slow_tau40"] = {
    "config": "dead_time_steps=3, tau_xd=40",
    "step_results": [
        {"shock": m["shock_kw"], "min_xd": m["min_xd"],
         "violations": m["violations"], "recovery": m["recovery_steps"]}
        for m in st_step
    ],
    "multi_collapsed": st_multi["collapsed"],
    "multi_min_xd": st_multi["min_xd"],
    "multi_violations": st_multi["total_violations"],
}

print("[Hardened 2] Very slow tau test complete")


# ------------------------------------------------------------------
#  HARDENED TEST 3: Process Noise
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  HARDENED TEST 3: Process Noise on HeatDuty")
print("-" * 65)

pn_results = run_process_noise_test(
    env, model,
    noise_kw_levels=[0.0, 1.0, 2.0, 5.0, 10.0],
    n_episodes=200,
    total_steps=200,
    seed=42,
)
plot_process_noise(pn_results)

hardened_summary["process_noise"] = []
for r in pn_results:
    hardened_summary["process_noise"].append({
        "noise_kw": r["noise_kw"],
        "mean_reward": r["mean_reward"],
        "mean_min_xd": r["mean_min_xd"],
        "violations": r["total_violations"],
    })

print("[Hardened 3] Process noise test complete")


# ------------------------------------------------------------------
#  HARDENED TEST: Combined (dead_time=6 + tau_xd=40)
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  HARDENED TEST: Combined (dead_time=6 + tau_xd=40)")
print("-" * 65)

COMBINED_CONFIG = BASELINE_CONFIG.copy()
COMBINED_CONFIG["dead_time_steps"] = 6
COMBINED_CONFIG["tau_xd"] = 40.0

env_cb = build_env(surrogate, COMBINED_CONFIG)

cb_step = run_all_step_disturbances(
    env_cb, model, shocks=[50.0, 100.0, 200.0, -100.0],
    shock_step=50, seed=42
)
plot_step_disturbances(cb_step, xd_target=0.95)
try:
    shutil.move("fig6_step_disturbance.png", "fig_hardened_combined_step.png")
    print("   Saved: fig_hardened_combined_step.png")
except Exception:
    pass

cb_multi = run_multi_disturbance(
    env_cb, model,
    disturbances=[(50, 100.0), (100, -150.0), (150, 80.0)],
    seed=42
)
plot_multi_disturbance(cb_multi, xd_target=0.95)
try:
    shutil.move("fig7_multi_disturbance.png", "fig_hardened_combined_multi.png")
    print("   Saved: fig_hardened_combined_multi.png")
except Exception:
    pass

hardened_summary["combined"] = {
    "config": "dead_time_steps=6, tau_xd=40",
    "step_results": [
        {"shock": m["shock_kw"], "min_xd": m["min_xd"],
         "violations": m["violations"], "recovery": m["recovery_steps"]}
        for m in cb_step
    ],
    "multi_collapsed": cb_multi["collapsed"],
    "multi_min_xd": cb_multi["min_xd"],
    "multi_violations": cb_multi["total_violations"],
}

print("[Hardened Combined] Test complete")


# ------------------------------------------------------------------
#  ADVANCED TEST: Edge-of-Feasibility Stress
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  ADVANCED TEST: Edge-of-Feasibility Stress")
print("-" * 65)

edge_results = run_edge_stress_test(
    env, model,
    shocks=[50.0, 100.0, 200.0, -50.0, -100.0],
    shock_step=50, total_steps=200, seed=42
)
plot_edge_stress(edge_results, xd_target=0.95)

hardened_summary["edge_stress"] = [
    {"shock": m["shock_kw"], "min_xd": m["min_xd"],
     "violations": m["violations"], "recovery": m["recovery_steps"],
     "overshoot": m["overshoot"], "start_xd": m["start_xd"]}
    for m in edge_results
]

print("[Advanced] Edge stress test complete")


# ------------------------------------------------------------------
#  ADVANCED TEST: Sustained Shock
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  ADVANCED TEST: Sustained Shock (hold for N steps)")
print("-" * 65)

sustained_results = run_all_sustained_shocks(
    env, model,
    shock_configs=[
        (100.0, 5),
        (100.0, 20),
        (100.0, 50),
        (200.0, 20),
        (-150.0, 20),
    ],
    seed=42
)
plot_sustained_shocks(sustained_results, xd_target=0.95)

hardened_summary["sustained_shock"] = [
    {"shock": m["shock_kw"], "hold": m["hold_steps"],
     "min_xd_during": m["min_xd_during"],
     "min_xd_after": m["min_xd_after"],
     "violations": m["violations"], "recovery": m["recovery_steps"]}
    for m in sustained_results
]

print("[Advanced] Sustained shock test complete")


# ------------------------------------------------------------------
#  ADVANCED TEST: Extreme Shocks
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  ADVANCED TEST: Extreme Shocks (breaking point search)")
print("-" * 65)

extreme_results = run_extreme_shocks(
    env, model,
    shocks=[100.0, 200.0, 400.0, 600.0, 800.0,
            -100.0, -200.0, -400.0],
    shock_step=50, total_steps=200, seed=42
)
plot_extreme_shocks(extreme_results, xd_target=0.95)

hardened_summary["extreme_shock"] = [
    {"shock": m["shock_kw"], "min_xd": m["min_xd"],
     "violations": m["violations"], "recovery": m["recovery_steps"]}
    for m in extreme_results
]

print("[Advanced] Extreme shock test complete")


# ==================================================================
#  LEVEL 14: ADVANCED ANALYSIS
# ==================================================================

print("\n")
print("=" * 65)
print("  LEVEL 14: ADVANCED ANALYSIS")
print("=" * 65)

advanced_results = {}


# ------------------------------------------------------------------
#  14.1 Dead-Time Sweep
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  14.1 Dead-Time Tolerance Sweep")
print("-" * 65)

dt_sweep = run_deadtime_sweep(
    build_env, surrogate, BASELINE_CONFIG, model,
    deadtime_values=[0, 1, 3, 5, 8, 12, 16, 20],
    n_episodes=200, seed=42
)
plot_deadtime_sweep(dt_sweep)
advanced_results["deadtime_sweep"] = dt_sweep


# ------------------------------------------------------------------
#  14.2 Constraint Tightening
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  14.2 Constraint Tightening")
print("-" * 65)

ct_results = run_constraint_tightening(
    build_env, surrogate, BASELINE_CONFIG, model,
    xd_specs=[0.950, 0.960, 0.970, 0.980, 0.985, 0.990, 0.995],
    n_episodes=200, seed=42
)
plot_constraint_tightening(ct_results)
advanced_results["constraint_tightening"] = ct_results


# ------------------------------------------------------------------
#  14.3 Adversarial Shock Sequences
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  14.3 Worst-Case Adversarial Sequences")
print("-" * 65)

adv_results = run_adversarial_sequence(env, model, seed=42)
plot_adversarial_sequences(adv_results)
advanced_results["adversarial"] = adv_results


# ------------------------------------------------------------------
#  14.4 Safety Shield
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  14.4 Safety Shield (Action Projection)")
print("-" * 65)

shield_result = run_with_safety_shield(env, model, n_episodes=200, seed=42)
plot_safety_shield(shield_result)
advanced_results["safety_shield"] = shield_result


# ------------------------------------------------------------------
#  14.5 Empirical Stability
# ------------------------------------------------------------------
print("\n" + "-" * 65)
print("  14.5 Empirical Stability Analysis")
print("-" * 65)

stab_result = run_stability_analysis(env, model, n_episodes=200, seed=42)
plot_stability_analysis(stab_result)
advanced_results["stability"] = stab_result


# ==================================================================
#  PLANT-MODEL MISMATCH ANALYSIS
# ==================================================================

print("\n")
print("=" * 65)
print("  PLANT-MODEL MISMATCH ANALYSIS")
print("  (Controller trained on nominal, tested on plant with gain)")
print("=" * 65)

print("\n[MISMATCH] Creating mismatch scenarios...")
mismatch_suite = create_mismatch_suite(surrogate)

mismatch_results = {}

for scenario_name, mismatched_surrogate in mismatch_suite.items():
    print("\n" + "-" * 65)
    print(f"  Testing scenario: {scenario_name.upper()}")
    print("-" * 65)

    env_mismatch = build_env(mismatched_surrogate, BASELINE_CONFIG)

    from level10_evaluation import evaluate_policy

    metrics_mismatch = evaluate_policy(
        env_mismatch, model=model, n_episodes=200,
        seed=42, label=f"SAC-{scenario_name}"
    )

    mismatch_results[scenario_name] = metrics_mismatch

    step_mismatch = run_all_step_disturbances(
        env_mismatch, model, shocks=[50.0, 100.0, 200.0, -100.0],
        shock_step=50, seed=42
    )

    mismatch_results[scenario_name]["step_disturbances"] = step_mismatch

    print(f"\n[{scenario_name.upper()}] Summary:")
    print(f"   Mean Reward:      {metrics_mismatch['mean_reward']:.2f}")
    print(f"   Mean XD:          {metrics_mismatch['mean_xd']:.4f}")
    print(f"   Spec Violations:  {metrics_mismatch['spec_violations']}")
    print(f"   Hard Violations:  {metrics_mismatch['hard_violations']}")

    plot_step_disturbances(step_mismatch, xd_target=0.95)
    try:
        shutil.move("fig6_step_disturbance.png",
                    f"fig_mismatch_{scenario_name}_step.png")
        print(f"   Saved: fig_mismatch_{scenario_name}_step.png")
    except Exception as e:
        print(f"   Warning: Could not save plot: {e}")

print("\n" + "=" * 65)
print("  MISMATCH ANALYSIS COMPLETE")
print("=" * 65)

print("\n[MISMATCH] Generating comparison plot...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Plant-Model Mismatch — Controller Performance Degradation",
                 fontsize=14, fontweight="bold")

    scenarios = ["nominal", "mild", "moderate", "severe"]
    labels_pretty = ["Nominal\n(gain=1.0)", "Mild\n(gain=1.05)",
                     "Moderate\n(gain=1.10)", "Severe\n(gain=1.20)"]

    rewards = [mismatch_results[s]["mean_reward"] for s in scenarios]
    xds = [mismatch_results[s]["mean_xd"] for s in scenarios]
    spec_viols = [mismatch_results[s]["spec_violations"] for s in scenarios]
    hard_viols = [mismatch_results[s]["hard_violations"] for s in scenarios]

    x_pos = np.arange(len(scenarios))
    colors = ["green", "yellow", "orange", "red"]

    bars = axes[0, 0].bar(x_pos, rewards, color=colors, alpha=0.7, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2.0, height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels_pretty, fontsize=8)
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title("Reward vs Mismatch Level")
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    bars = axes[0, 1].bar(x_pos, xds, color=colors, alpha=0.7, edgecolor="black")
    axes[0, 1].axhline(0.95, color="red", linestyle="--", linewidth=2, label="Spec=0.95")
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(labels_pretty, fontsize=8)
    axes[0, 1].set_ylabel("Mean XD")
    axes[0, 1].set_title("Purity vs Mismatch Level")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    bars = axes[1, 0].bar(x_pos, spec_viols, color=colors, alpha=0.7, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2.0, height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels_pretty, fontsize=8)
    axes[1, 0].set_ylabel("Spec Violations (XD<0.95)")
    axes[1, 0].set_title("Spec Violations vs Mismatch")
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    bars = axes[1, 1].bar(x_pos, hard_viols, color=colors, alpha=0.7, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2.0, height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels_pretty, fontsize=8)
    axes[1, 1].set_ylabel("Hard Safety Violations")
    axes[1, 1].set_title("Safety Violations vs Mismatch")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("fig_mismatch_summary.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("   Saved: fig_mismatch_summary.png")

except Exception as e:
    print(f"   Warning: Could not generate mismatch comparison plot: {e}")

print("\n[MISMATCH] Generating report...")

with open("plant_model_mismatch_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  PLANT-MODEL MISMATCH ANALYSIS\n")
    f.write("  Gain-based purity mismatch\n")
    f.write("  Controller trained on nominal model (gain=1.0)\n")
    f.write("  Evaluated on plant with systematic impurity bias\n")
    f.write("=" * 60 + "\n\n")

    f.write("MISMATCH DEFINITION:\n")
    f.write("  Impurity_plant = Impurity_nominal × gain\n")
    f.write("  XD_plant = 1 - Impurity_plant\n")
    f.write("  gain > 1.0 → plant has MORE impurity than model predicts\n\n")

    f.write("SCENARIOS TESTED:\n")
    f.write("  Nominal:  gain=1.00 (no mismatch)\n")
    f.write("  Mild:     gain=1.05 (5% worse purity)\n")
    f.write("  Moderate: gain=1.10 (10% worse purity)\n")
    f.write("  Severe:   gain=1.20 (20% worse purity)\n\n")

    f.write("=" * 60 + "\n")
    f.write("STEADY-STATE EVALUATION (200 episodes)\n")
    f.write("=" * 60 + "\n\n")

    scenarios = ["nominal", "mild", "moderate", "severe"]

    f.write("%-12s %-12s %-10s %-12s %-12s\n" %
            ("Scenario", "Reward", "Mean XD", "Spec Viol.", "Hard Viol."))
    f.write("-" * 60 + "\n")

    for scenario in scenarios:
        m = mismatch_results[scenario]
        f.write("%-12s %-12.2f %-10.4f %-12d %-12d\n" %
                (scenario.capitalize(), m["mean_reward"], m["mean_xd"],
                 m["spec_violations"], m["hard_violations"]))

    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("DISTURBANCE RESPONSE UNDER MISMATCH\n")
    f.write("=" * 60 + "\n\n")

    for scenario in scenarios:
        f.write(f"\n{scenario.upper()} SCENARIO:\n")
        f.write("-" * 45 + "\n")
        f.write("%-14s %-10s %-12s %-12s\n" %
                ("Shock (kW)", "Min XD", "Violations", "Recovery"))
        f.write("-" * 45 + "\n")

        for m in mismatch_results[scenario]["step_disturbances"]:
            sign = "+" if m["shock_kw"] >= 0 else ""
            rec = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
            f.write("%-14s %-10.4f %-12d %-12s\n" %
                    (f"{sign}{m['shock_kw']:.0f}",
                     m["min_xd"], m["violations"], rec))

    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("=" * 60 + "\n\n")

    nominal_reward = mismatch_results["nominal"]["mean_reward"]
    severe_reward = mismatch_results["severe"]["mean_reward"]
    reward_degradation = ((nominal_reward - severe_reward) / abs(nominal_reward)) * 100 if nominal_reward != 0 else 0

    nominal_xd = mismatch_results["nominal"]["mean_xd"]
    severe_xd = mismatch_results["severe"]["mean_xd"]

    f.write(f"1. Reward degradation (nominal→severe): {reward_degradation:.1f}%\n")
    f.write(f"2. XD drop (nominal→severe): {nominal_xd:.4f} → {severe_xd:.4f}\n")
    f.write(f"3. Spec violations (severe): {mismatch_results['severe']['spec_violations']}\n")

    if mismatch_results["severe"]["hard_violations"] == 0:
        f.write("\n✓ Controller maintained hard safety even under 20% mismatch\n")
    else:
        f.write("\n✗ Hard safety violations occurred under severe mismatch\n")

    if mismatch_results["moderate"]["spec_violations"] == 0:
        f.write("✓ Controller robust to 10% mismatch (moderate scenario)\n")
    else:
        f.write("⚠ Spec violations appear at 10% mismatch level\n")

    f.write("\n" + "=" * 60 + "\n")

print("   Saved: plant_model_mismatch_report.txt")


# ==================================================================
#  SAVE ALL REPORTS
# ==================================================================
print("\n" + "-" * 65)
print("  Saving All Reports")
print("-" * 65)

# -- Hardened Report --
with open("hardened_scenarios_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  HARDENED SCENARIO TESTS -- SUMMARY\n")
    f.write("  Baseline: FOPDT (tau_xd=20, dead_time=3)\n")
    f.write("  Surrogate: %s\n" % SURROGATE_TYPE.upper())
    f.write("  Policy NOT retrained for hardened tests.\n")
    f.write("=" * 60 + "\n\n")

    def write_step_table(f, label, config_str, step_data):
        f.write("\n%s\n" % label)
        f.write("Config: %s\n" % config_str)
        f.write("-" * 55 + "\n")
        f.write("%-14s %-10s %-12s %-12s\n" %
                ("Shock (kW)", "Min XD", "Violations", "Recovery"))
        f.write("-" * 55 + "\n")
        for s in step_data:
            sign = "+" if s["shock"] >= 0 else ""
            rec = str(s["recovery"]) if s["recovery"] >= 0 else "N/A"
            f.write("%-14s %-10.4f %-12d %-12s\n" %
                    ("%s%.0f" % (sign, s["shock"]),
                     s["min_xd"], s["violations"], rec))
        f.write("\n")

    def write_multi_result(f, data):
        f.write("  Multi-shock collapsed: %s\n" %
                ("YES" if data["multi_collapsed"] else "NO"))
        f.write("  Multi-shock min XD:    %.4f\n" % data["multi_min_xd"])
        f.write("  Multi-shock violations: %d\n" % data["multi_violations"])
        f.write("\n")

    f.write("BASELINE (tau_xd=20, dead_time=3)\n")
    f.write("-" * 55 + "\n")
    f.write("  SAC Reward:     %.2f\n" % results["sac_mean_reward"])
    f.write("  SAC XD:         %.4f\n" % results["sac_mean_xd"])
    f.write("  SAC Violations: %d\n" % results["sac_violations"])
    f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("TEST 1: EXTENDED DEAD-TIME (dead_time=6)\n")
    f.write("=" * 60 + "\n")
    write_step_table(f, "Step Disturbances:",
                     hardened_summary["deadtime6"]["config"],
                     hardened_summary["deadtime6"]["step_results"])
    write_multi_result(f, hardened_summary["deadtime6"])

    f.write("=" * 60 + "\n")
    f.write("TEST 2: VERY SLOW COMPOSITION (tau_xd=40)\n")
    f.write("=" * 60 + "\n")
    write_step_table(f, "Step Disturbances:",
                     hardened_summary["slow_tau40"]["config"],
                     hardened_summary["slow_tau40"]["step_results"])
    write_multi_result(f, hardened_summary["slow_tau40"])

    f.write("=" * 60 + "\n")
    f.write("TEST 3: PROCESS NOISE\n")
    f.write("=" * 60 + "\n")
    f.write("-" * 55 + "\n")
    f.write("%-12s %-12s %-10s %-12s\n" %
            ("Noise (kW)", "Reward", "Min XD", "Violations"))
    f.write("-" * 55 + "\n")
    for r in hardened_summary["process_noise"]:
        f.write("%-12.1f %-12.2f %-10.4f %-12d\n" %
                (r["noise_kw"], r["mean_reward"],
                 r["mean_min_xd"], r["violations"]))
    f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("TEST: COMBINED (dead_time=6 + tau_xd=40)\n")
    f.write("=" * 60 + "\n")
    write_step_table(f, "Step Disturbances:",
                     hardened_summary["combined"]["config"],
                     hardened_summary["combined"]["step_results"])
    write_multi_result(f, hardened_summary["combined"])

    f.write("=" * 60 + "\n")
    f.write("TEST: EDGE-OF-FEASIBILITY STRESS\n")
    f.write("=" * 60 + "\n")
    f.write("Starting XD near 0.955 (minimal margin)\n")
    f.write("-" * 55 + "\n")
    f.write("%-14s %-10s %-10s %-12s %-12s\n" %
            ("Shock (kW)", "Min XD", "Overshoot", "Recovery", "Violations"))
    f.write("-" * 55 + "\n")
    for s in hardened_summary.get("edge_stress", []):
        sign = "+" if s["shock"] >= 0 else ""
        rec = str(s["recovery"]) if s["recovery"] >= 0 else "N/A"
        f.write("%-14s %-10.4f %-10.4f %-12s %-12d\n" %
                ("%s%.0f" % (sign, s["shock"]),
                 s["min_xd"], s["overshoot"], rec, s["violations"]))
    f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("TEST: SUSTAINED SHOCK\n")
    f.write("=" * 60 + "\n")
    f.write("-" * 60 + "\n")
    f.write("%-10s %-6s %-12s %-12s %-10s %-10s\n" %
            ("Shock kW", "Hold", "Min During", "Min After", "Recovery", "Viol"))
    f.write("-" * 60 + "\n")
    for s in hardened_summary.get("sustained_shock", []):
        sign = "+" if s["shock"] >= 0 else ""
        rec = str(s["recovery"]) if s["recovery"] >= 0 else "N/A"
        f.write("%-10s %-6d %-12.4f %-12.4f %-10s %-10d\n" %
                ("%s%.0f" % (sign, s["shock"]),
                 s["hold"], s["min_xd_during"],
                 s["min_xd_after"], rec, s["violations"]))
    f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("TEST: EXTREME SHOCKS\n")
    f.write("=" * 60 + "\n")
    f.write("-" * 55 + "\n")
    f.write("%-14s %-10s %-12s %-12s\n" %
            ("Shock (kW)", "Min XD", "Recovery", "Violations"))
    f.write("-" * 55 + "\n")
    for s in hardened_summary.get("extreme_shock", []):
        sign = "+" if s["shock"] >= 0 else ""
        rec = str(s["recovery"]) if s["recovery"] >= 0 else "N/A"
        broke = " << BROKE" if s["violations"] > 0 else ""
        f.write("%-14s %-10.4f %-12s %-12d%s\n" %
                ("%s%.0f" % (sign, s["shock"]),
                 s["min_xd"], rec, s["violations"], broke))
    f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("CONCLUSION\n")
    f.write("=" * 60 + "\n")
    all_collapsed = [
        hardened_summary["deadtime6"]["multi_collapsed"],
        hardened_summary["slow_tau40"]["multi_collapsed"],
        hardened_summary["combined"]["multi_collapsed"],
    ]
    if not any(all_collapsed):
        f.write("  The SAC policy trained on FOPDT baseline (tau=20, dt=3)\n")
        f.write("  with %s surrogate maintained stability across all\n" % SURROGATE_TYPE.upper())
        f.write("  hardened scenarios.\n")
    else:
        collapsed_names = []
        if hardened_summary["deadtime6"]["multi_collapsed"]:
            collapsed_names.append("Extended Dead-Time")
        if hardened_summary["slow_tau40"]["multi_collapsed"]:
            collapsed_names.append("Very Slow Tau")
        if hardened_summary["combined"]["multi_collapsed"]:
            collapsed_names.append("Combined")
        f.write("  Policy collapsed under: %s\n" % ", ".join(collapsed_names))
        f.write("  Retraining under harder dynamics may help.\n")
    f.write("\n" + "=" * 60 + "\n")

print("[Report] Saved: hardened_scenarios_report.txt")


# -- Robustness Summary --
with open("robustness_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 55 + "\n")
    f.write("  ROBUSTNESS ANALYSIS SUMMARY\n")
    f.write("  Baseline: FOPDT (tau_xd=20, dead_time=3)\n")
    f.write("  Surrogate: %s\n" % SURROGATE_TYPE.upper())
    f.write("=" * 55 + "\n\n")

    f.write("1. STEP DISTURBANCE TESTS\n")
    f.write("-" * 45 + "\n")
    f.write("%-14s %-10s %-12s %-12s %-10s\n" %
            ("Shock (kW)", "Min XD", "Violations", "Recovery", "Overshoot"))
    f.write("-" * 45 + "\n")
    for m in robustness_results["step_disturbances"]:
        sign = "+" if m["shock_kw"] >= 0 else ""
        rec = str(m["recovery_steps"]) if m["recovery_steps"] >= 0 else "N/A"
        f.write("%-14s %-10.4f %-12d %-12s %-10.4f\n" %
                ("%s%.0f" % (sign, m["shock_kw"]),
                 m["min_xd"], m["violations"], rec, m["overshoot"]))

    f.write("\n\n2. MULTI-DISTURBANCE SCENARIO\n")
    f.write("-" * 45 + "\n")
    mr = robustness_results["multi_disturbance"]
    f.write("  Disturbances:\n")
    for step, shock in mr["disturbances"]:
        sign = "+" if shock >= 0 else ""
        f.write("    Step %d: %s%.0f kW\n" % (step, sign, shock))
    f.write("  Min XD:           %.4f\n" % mr["min_xd"])
    f.write("  Total Violations: %d\n" % mr["total_violations"])
    f.write("  Episode Length:   %d\n" % mr["episode_length"])
    f.write("  Collapsed:        %s\n" % ("YES" if mr["collapsed"] else "NO"))

    f.write("\n\n3. DISTURBANCE TIMING SENSITIVITY\n")
    f.write("-" * 45 + "\n")
    f.write("%-14s %-10s %-12s %-12s\n" %
            ("Shock Step", "Min XD", "Recovery", "Violations"))
    f.write("-" * 45 + "\n")
    for r in robustness_results["timing_sensitivity"]:
        rec = str(r["recovery_steps"]) if r["recovery_steps"] >= 0 else "N/A"
        f.write("Step %-9d %-10.4f %-12s %-12d\n" %
                (r["shock_step"], r["min_xd"], rec, r["violations"]))

    f.write("\n\n4. NOISE ROBUSTNESS (Observation Noise)\n")
    f.write("-" * 45 + "\n")
    f.write("%-10s %-12s %-10s %-12s\n" %
            ("Noise s", "Reward", "Min XD", "Violations"))
    f.write("-" * 45 + "\n")
    for r in robustness_results["noise_robustness"]:
        f.write("%-10.4f %-12.2f %-10.4f %-12d\n" %
                (r["noise_sigma"], r["mean_reward"],
                 r["mean_min_xd"], r["total_violations"]))

    f.write("\n\n5. GRACE STEP SENSITIVITY\n")
    f.write("-" * 45 + "\n")
    f.write("%-8s %-12s %-12s %-12s\n" %
            ("Grace", "Reward", "Avg Length", "Violations"))
    f.write("-" * 45 + "\n")
    for r in robustness_results["grace_sensitivity"]:
        f.write("%-8d %-12.2f %-12.1f %-12d\n" %
                (r["grace_steps"], r["mean_reward"],
                 r["mean_episode_length"], r["total_violations"]))

    f.write("\n" + "=" * 55 + "\n")

print("[Report] Saved: robustness_summary.txt")


# -- Advanced Analysis Report --
with open("advanced_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  ADVANCED ANALYSIS REPORT (Level 14)\n")
    f.write("  Safe RL for Distillation Column\n")
    f.write("  Baseline: FOPDT (tau_xd=20, dead_time=3)\n")
    f.write("  Surrogate: %s\n" % SURROGATE_TYPE.upper())
    f.write("=" * 60 + "\n\n")

    if SURROGATE_TYPE == "mlp":
        f.write("SURROGATE MODEL EVALUATION (MLP)\n")
        f.write("-" * 50 + "\n")
        mlp_m = surrogate.get_test_metrics()
        f.write("%-10s %-12s %-12s %-12s %-12s\n" %
                ("Output", "Train RMSE", "Train R²", "Test RMSE", "Test R²"))
        f.write("-" * 50 + "\n")
        for col in ["XD", "TTOP", "TMID", "TBOTTOM"]:
            f.write("%-10s %-12.6f %-12.4f %-12.6f %-12.4f\n" %
                    (col,
                     mlp_m["train_rmse"][col], mlp_m["train_r2"][col],
                     mlp_m["test_rmse"][col], mlp_m["test_r2"][col]))
        f.write("\n")

    f.write("14.1 DEAD-TIME TOLERANCE SWEEP\n")
    f.write("-" * 50 + "\n")
    f.write("%-12s %-12s %-10s %-12s %-10s\n" %
            ("Dead-Time", "Reward", "Min XD", "Violations", "Length"))
    f.write("-" * 50 + "\n")
    for r in dt_sweep:
        f.write("%-12d %-12.2f %-10.4f %-12d %-10.0f\n" %
                (r["dead_time"], r["mean_reward"],
                 r["mean_min_xd"], r["total_violations"],
                 r["mean_length"]))
    f.write("\n")

    f.write("14.2 CONSTRAINT TIGHTENING\n")
    f.write("-" * 50 + "\n")
    f.write("%-10s %-12s %-10s %-12s %-12s\n" %
            ("XD Spec", "Reward", "Mean XD", "Energy", "Violations"))
    f.write("-" * 50 + "\n")
    for r in ct_results:
        f.write("%-10.3f %-12.2f %-10.4f %-12.2f %-12d\n" %
                (r["xd_spec"], r["mean_reward"],
                 r["mean_xd"], r["mean_energy"],
                 r["total_violations"]))
    f.write("\n")

    f.write("14.3 ADVERSARIAL SHOCK SEQUENCES\n")
    f.write("-" * 50 + "\n")
    f.write("%-18s %-10s %-12s %-10s\n" %
            ("Sequence", "Min XD", "Violations", "Survived"))
    f.write("-" * 50 + "\n")
    for r in adv_results:
        f.write("%-18s %-10.4f %-12d %-10s\n" %
                (r["name"], r["min_xd"], r["violations"],
                 "YES" if r["survived"] else "NO"))
    f.write("\n")

    f.write("14.4 SAFETY SHIELD COMPARISON\n")
    f.write("-" * 50 + "\n")
    f.write("  Without shield:\n")
    f.write("    Reward:     %.2f\n" % shield_result["noshield_mean_reward"])
    f.write("    Violations: %d\n" % shield_result["noshield_violations"])
    f.write("    Mean XD:    %.4f\n" % shield_result["noshield_mean_xd"])
    f.write("  With shield:\n")
    f.write("    Reward:     %.2f\n" % shield_result["shield_mean_reward"])
    f.write("    Violations: %d\n" % shield_result["shield_violations"])
    f.write("    Mean XD:    %.4f\n" % shield_result["shield_mean_xd"])
    f.write("    Avg interventions: %.1f per episode\n" %
            shield_result["shield_interventions"])
    f.write("\n")

    f.write("14.5 EMPIRICAL STABILITY\n")
    f.write("-" * 50 + "\n")
    f.write("  |delta_heat| first quarter:  %.4f\n" %
            stab_result["first_quarter_mean"])
    f.write("  |delta_heat| last quarter:   %.4f\n" %
            stab_result["last_quarter_mean"])
    f.write("  Converged:                   %s\n" %
            ("YES" if stab_result["converged"] else "NO"))
    f.write("\n")

    f.write("=" * 60 + "\n")

print("[Level 14] Saved: advanced_analysis_report.txt")


# ==================================================================
#  FINAL SUMMARY
# ==================================================================
print("\n")
print("=" * 65)
print("  PIPELINE COMPLETE -- ALL LEVELS (0-14) + MISMATCH")
print("  Dynamic Model: FOPDT (tau_xd=20, dead_time=3)")
print("  Surrogate Type: %s" % SURROGATE_TYPE.upper())
print("=" * 65)
print("")
print("  Generated Reports:")
print("    final_report.txt")
print("    robustness_summary.txt")
print("    hardened_scenarios_report.txt")
print("    advanced_analysis_report.txt")
print("    plant_model_mismatch_report.txt")
print("")
print("  Generated Figures:")
print("    fig0_surrogate_comparison     (Interp vs MLP)")
if SURROGATE_TYPE == "mlp":
    print("    fig0b_mlp_test_metrics        (MLP Train/Test R² & RMSE)")
print("    fig1-fig5                     (Baseline)")
print("    fig6-fig10                    (Robustness)")
print("    fig_hardened1_deadtime6_*     (Hardened Dead-Time)")
print("    fig_hardened2_slowtau40_*     (Hardened Slow Tau)")
print("    fig_hardened_combined_*       (Hardened Combined)")
print("    fig_hardened3_process_noise   (Process Noise)")
print("    fig_edge_stress               (Edge Stress)")
print("    fig_sustained_shock           (Sustained Shock)")
print("    fig_extreme_shocks            (Extreme Shocks)")
print("    fig_deadtime_sweep            (Level 14.1)")
print("    fig_constraint_tightening     (Level 14.2)")
print("    fig_adversarial               (Level 14.3)")
print("    fig_safety_shield             (Level 14.4)")
print("    fig_stability                 (Level 14.5)")
print("    fig_mismatch_*_step.png       (Mismatch per scenario)")
print("    fig_mismatch_summary.png      (Mismatch 4-panel)")
print("    fig_fopdt_step_response.png   (NEW: Step test)")
print("")
print("  Baseline Results:")
print("    SAC Reward:     %.2f (PID: %.2f | Random: %.2f)" %
      (results["sac_mean_reward"], results.get("pid_mean_reward", float("nan")), results["random_mean_reward"]))
print("    SAC XD:         %.4f (PID: %.4f | Random: %.4f)" %
      (results["sac_mean_xd"], results.get("pid_mean_xd", float("nan")), results["random_mean_xd"]))
print("    SAC Violations: %d (PID: %d | Random: %d)" %
      (results["sac_violations"], results.get("pid_violations", -1), results["random_violations"]))

if SURROGATE_TYPE == "mlp":
    print("")
    print("  MLP Surrogate Test-Set R²:")
    mlp_m = surrogate.get_test_metrics()
    for col in ["XD", "TTOP", "TMID", "TBOTTOM"]:
        print("    %-10s  R²=%.4f  RMSE=%.6f" %
              (col, mlp_m["test_r2"][col], mlp_m["test_rmse"][col]))
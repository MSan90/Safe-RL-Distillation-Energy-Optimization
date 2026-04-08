"""
Level 12: Plotting — All 11 Charts (5 Figures)

Professional version:
  - Chart 3: Episode-level binary violation (Violated yes/no) + Rate annotation
  - Figure 4: Violation Rate bar (not boxplot) + Mean violation steps
  - Figure 5: Temperature limits read from env.core.safety (not hard-coded)
  - All violation labels explicitly say "Hard Safety" (XD + T constraints)

UPDATED:
  - Adds PID baseline: SAC vs PID vs Random
"""
import numpy as np
import matplotlib.pyplot as plt
from level15_pid_baseline import build_default_pid


def collect_ep(env, model=None, pid=None, seed=42):
    obs, info0 = env.reset(seed=seed)
    core = getattr(env, "core", None)
    prev_heat = core.heat_duty if core else 0.0

    last_xd = info0.get("xd", 0.95)
    if pid is not None:
        pid.reset()

    data = {"xd": [], "heat_duty": [], "reward": [], "violation": [], "action": [],
            "ttop": [], "tmid": [], "tbottom": []}
    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif pid is not None:
            action = pid.compute(last_xd)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        last_xd = info.get("xd", last_xd)

        data["xd"].append(info["xd"])
        data["heat_duty"].append(info["heat_duty"])
        data["reward"].append(reward)
        data["violation"].append(0 if info.get("safe", True) else 1)
        data["action"].append(info["heat_duty"] - prev_heat)
        data["ttop"].append(info.get("ttop", 0.0))
        data["tmid"].append(info.get("tmid", 0.0))
        data["tbottom"].append(info.get("tbottom", 0.0))
        prev_heat = info["heat_duty"]
        if terminated or truncated:
            break
    for k in data:
        data[k] = np.array(data[k])
    return data


print("[PLOTS] Collecting episode data...")
pid = build_default_pid(env)

sac_data = collect_ep(env, model=model, pid=None, seed=42)
pid_data = collect_ep(env, model=None, pid=pid, seed=42)
rnd_data = collect_ep(env, model=None, pid=None, seed=42)

xd_min = 0.95
w = 0.27  # narrower bars since we have 3 series

# ══════════════════════════════════════
# FIGURE 1: 5 main charts
# ══════════════════════════════════════
print("[1/5] Figure 1: Main 5 charts...")
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Safe RL -- SAC vs PID vs Random", fontsize=16, fontweight="bold", y=0.98)
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 1, 3)

# Chart 1: XD vs Time
ax1.plot(rnd_data["xd"], "r--", lw=1.5, alpha=0.7, label="Random")
ax1.plot(pid_data["xd"], "g-", lw=1.8, alpha=0.8, label="PID")
ax1.plot(sac_data["xd"], "b-", lw=2, label="SAC")
ax1.axhline(xd_min, color="black", ls=":", lw=1.5, label="Spec=0.95")
ax1.set_xlabel("Step")
ax1.set_ylabel("XD")
ax1.set_title("1. Purity (XD)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.88, 1.01])

# Chart 2: HeatDuty vs Time
ax2.plot(rnd_data["heat_duty"], "r--", lw=1.5, alpha=0.7, label="Random")
ax2.plot(pid_data["heat_duty"], "g-", lw=1.8, alpha=0.8, label="PID")
ax2.plot(sac_data["heat_duty"], "b-", lw=2, label="SAC")
ax2.set_xlabel("Step")
ax2.set_ylabel("HeatDuty (kW)")
ax2.set_title("2. Energy")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Chart 3: Hard Safety Violations (30 episodes — episode-level binary)
n_eps = 30
sac_violated = []
pid_violated = []
rnd_violated = []
sac_viol_steps = []
pid_viol_steps = []
rnd_viol_steps = []

for ep in range(n_eps):
    s = collect_ep(env, model=model, pid=None, seed=200 + ep)
    p = collect_ep(env, model=None, pid=build_default_pid(env), seed=200 + ep)
    r = collect_ep(env, model=None, pid=None, seed=200 + ep)

    sac_violated.append(int(s["violation"].sum() > 0))
    pid_violated.append(int(p["violation"].sum() > 0))
    rnd_violated.append(int(r["violation"].sum() > 0))

    sac_viol_steps.append(s["violation"].sum())
    pid_viol_steps.append(p["violation"].sum())
    rnd_viol_steps.append(r["violation"].sum())

x3 = np.arange(n_eps)
ax3.bar(x3 - w, rnd_violated, w, color="red", alpha=0.7, label="Random")
ax3.bar(x3, pid_violated, w, color="green", alpha=0.7, label="PID")
ax3.bar(x3 + w, sac_violated, w, color="blue", alpha=0.7, label="SAC")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Violated (0=Safe, 1=Unsafe)")
ax3.set_title("3. Hard Safety Violations — XD + T constraints (%d Eps)" % n_eps)
ticks3 = np.arange(0, n_eps, max(1, n_eps // 10))
ax3.set_xticks(ticks3)
ax3.set_xticklabels(["Ep%d" % (i + 1) for i in ticks3], fontsize=7)
ax3.set_ylim([0, 1.3])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["Safe", "Violated"])

sac_rate = 100.0 * sum(sac_violated) / n_eps
pid_rate = 100.0 * sum(pid_violated) / n_eps
rnd_rate = 100.0 * sum(rnd_violated) / n_eps

ax3.text(0.98, 0.95,
         "Violation Rate:\nRandom: %.0f%%\nPID: %.0f%%\nSAC: %.0f%%" % (rnd_rate, pid_rate, sac_rate),
         transform=ax3.transAxes, fontsize=8, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# Chart 4: Summary Bar
labels = ["Reward", "XD", "Energy"]
sv = [results["sac_mean_reward"], results["sac_mean_xd"], results["sac_mean_energy"]]
pv = [results.get("pid_mean_reward", 0.0), results.get("pid_mean_xd", 0.0), results.get("pid_mean_energy", 0.0)]
rv = [results["random_mean_reward"], results["random_mean_xd"], results["random_mean_energy"]]

x4 = np.arange(3)
b_r = ax4.bar(x4 - w, rv, w, color="red", alpha=0.7, label="Random")
b_p = ax4.bar(x4, pv, w, color="green", alpha=0.7, label="PID")
b_s = ax4.bar(x4 + w, sv, w, color="blue", alpha=0.7, label="SAC")

for bar in list(b_r) + list(b_p) + list(b_s):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             "%.2f" % bar.get_height(), ha="center", va="bottom", fontsize=7)

ax4.set_ylabel("Value")
ax4.set_title("4. Summary")
ax4.set_xticks(x4)
ax4.set_xticklabels(labels)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

# Chart 5: Learning Curve
ep_r = training_rewards
win = max(5, len(ep_r) // 10)
if len(ep_r) >= win:
    sm = np.convolve(ep_r, np.ones(win) / win, mode="valid")
else:
    sm = np.array(ep_r)
ax5.plot(ep_r, color="lightblue", alpha=0.4, lw=0.8, label="Raw")
ax5.plot(np.arange(win - 1, win - 1 + len(sm)), sm,
         color="blue", lw=2.5, label="Avg(w=%d)" % win)
ax5.axhline(0, color="gray", ls="--", lw=0.8)
ax5.set_xlabel("Episode")
ax5.set_ylabel("Reward")
ax5.set_title("5. Learning Curve (SAC)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig1_main_5charts.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig1_main_5charts.png")

# ══════════════════════════════════════
# FIGURE 2: Surrogate
# ══════════════════════════════════════
print("[2/5] Figure 2: Surrogate...")
sur = env.core.surrogate
heats = np.linspace(sur.heat_min, sur.heat_max, 200)
xd_p = [sur.predict(h)["XD_ss"] for h in heats]
tt_p = [sur.predict(h)["TTOP_ss"] for h in heats]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Surrogate Model", fontsize=14, fontweight="bold")
axes[0].plot(heats, xd_p, "b-", lw=2)
axes[0].axhline(xd_min, color="red", ls="--", lw=1.5, label="Spec=0.95")
axes[0].set_xlabel("HeatDuty (kW)")
axes[0].set_ylabel("XD")
axes[0].set_title("6. HeatDuty -> XD")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].plot(heats, tt_p, "g-", lw=2)
axes[1].set_xlabel("HeatDuty (kW)")
axes[1].set_ylabel("TTOP (C)")
axes[1].set_title("7. HeatDuty -> TTOP")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_surrogate.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig2_surrogate.png")

# ══════════════════════════════════════
# FIGURE 3: Trade-off + Action
# ══════════════════════════════════════
print("[3/5] Figure 3: Trade-off + Action...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trade-off and Control Effort", fontsize=14, fontweight="bold")
axes[0].scatter(rnd_data["heat_duty"], rnd_data["xd"],
                c="red", alpha=0.4, s=15, label="Random")
axes[0].scatter(pid_data["heat_duty"], pid_data["xd"],
                c="green", alpha=0.5, s=18, label="PID")
axes[0].scatter(sac_data["heat_duty"], sac_data["xd"],
                c="blue", alpha=0.6, s=20, label="SAC")
axes[0].axhline(xd_min, color="black", ls=":", lw=1.5, label="Spec=0.95")
axes[0].set_xlabel("HeatDuty (kW)")
axes[0].set_ylabel("XD")
axes[0].set_title("8. Trade-off")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].plot(rnd_data["action"], "r--", lw=1, alpha=0.6, label="Random")
axes[1].plot(pid_data["action"], "g-", lw=1.2, alpha=0.7, label="PID")
axes[1].plot(sac_data["action"], "b-", lw=1.5, label="SAC")
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Delta Heat (kW)")
axes[1].set_title("9. Control Effort")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_tradeoff_action.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig3_tradeoff_action.png")

# ══════════════════════════════════════
# FIGURE 4: Boxplot 200 episodes
# ══════════════════════════════════════
print("[4/5] Figure 4: Statistical comparison (200 episodes)...")
n_box = 200
sac_m = [collect_ep(env, model=model, pid=None, seed=300 + i) for i in range(n_box)]
pid_m = [collect_ep(env, model=None, pid=build_default_pid(env), seed=300 + i) for i in range(n_box)]
rnd_m = [collect_ep(env, model=None, pid=None, seed=300 + i) for i in range(n_box)]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Statistical Comparison (%d Episodes)" % n_box, fontsize=14, fontweight="bold")

# Panels 1-3: Boxplots (Mean XD, Mean HeatDuty, Episode Length)
titles_box = ["Mean XD", "Mean HeatDuty (kW)", "Episode Length"]

sa_box = [[np.mean(e["xd"]) for e in sac_m],
          [np.mean(e["heat_duty"]) for e in sac_m],
          [len(e["xd"]) for e in sac_m]]
pa_box = [[np.mean(e["xd"]) for e in pid_m],
          [np.mean(e["heat_duty"]) for e in pid_m],
          [len(e["xd"]) for e in pid_m]]
ra_box = [[np.mean(e["xd"]) for e in rnd_m],
          [np.mean(e["heat_duty"]) for e in rnd_m],
          [len(e["xd"]) for e in rnd_m]]

for i in range(3):
    ax = axes[i]
    bp = ax.boxplot([ra_box[i], pa_box[i], sa_box[i]], tick_labels=["Random", "PID", "SAC"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    bp["boxes"][2].set_facecolor("lightblue")
    ax.set_title(titles_box[i])
    ax.grid(True, alpha=0.3, axis="y")
    if i == 0:
        ax.axhline(xd_min, color="black", ls=":", lw=1.5)

# Panel 4: Hard Safety Violation Rate bar + Mean violation steps
ax_viol = axes[3]
sac_viol_binary = [int(e["violation"].sum() > 0) for e in sac_m]
pid_viol_binary = [int(e["violation"].sum() > 0) for e in pid_m]
rnd_viol_binary = [int(e["violation"].sum() > 0) for e in rnd_m]

sac_viol_rate = 100.0 * sum(sac_viol_binary) / n_box
pid_viol_rate = 100.0 * sum(pid_viol_binary) / n_box
rnd_viol_rate = 100.0 * sum(rnd_viol_binary) / n_box

sac_mean_vsteps = np.mean([e["violation"].sum() for e in sac_m])
pid_mean_vsteps = np.mean([e["violation"].sum() for e in pid_m])
rnd_mean_vsteps = np.mean([e["violation"].sum() for e in rnd_m])

bars_v = ax_viol.bar(["Random", "PID", "SAC"],
                     [rnd_viol_rate, pid_viol_rate, sac_viol_rate],
                     color=["lightcoral", "lightgreen", "lightblue"],
                     edgecolor=["red", "green", "blue"], linewidth=1.5,
                     alpha=0.8, width=0.6)
for bv in bars_v:
    ax_viol.text(bv.get_x() + bv.get_width() / 2, bv.get_height() + 1,
                 "%.1f%%" % bv.get_height(),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_viol.set_ylabel("Violation Rate (%)")
ax_viol.set_title("Hard Safety Violation Rate")
ax_viol.set_ylim([0, max(rnd_viol_rate, pid_viol_rate, sac_viol_rate) * 1.3 + 5])
ax_viol.grid(True, alpha=0.3, axis="y")
ax_viol.text(0.98, 0.95,
             "Mean viol. steps/ep:\nRandom: %.1f\nPID: %.1f\nSAC: %.1f" %
             (rnd_mean_vsteps, pid_mean_vsteps, sac_mean_vsteps),
             transform=ax_viol.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig("fig4_boxplot_200eps.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig4_boxplot_200eps.png")

# ══════════════════════════════════════
# FIGURE 5: Cumulative Reward + Temperature
# ══════════════════════════════════════
print("[5/5] Figure 5: Cumulative Reward + Temperature...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Advanced Analysis", fontsize=14, fontweight="bold")

sac_cum = np.cumsum(sac_data["reward"])
pid_cum = np.cumsum(pid_data["reward"])
rnd_cum = np.cumsum(rnd_data["reward"])

axes[0].plot(rnd_cum, "r--", lw=1.5, alpha=0.7, label="Random")
axes[0].plot(pid_cum, "g-", lw=1.5, alpha=0.8, label="PID")
axes[0].plot(sac_cum, "b-", lw=2, label="SAC")
axes[0].axhline(0, color="gray", ls="--", lw=0.8)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Cumulative Reward")
axes[0].set_title("10. Cumulative Reward")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# ── Read temperature limits from SafetyChecker (not hard-coded) ──
safety = env.core.safety
ttop_min, ttop_max = safety.ttop_range
tmid_min, tmid_max = safety.tmid_range
tbot_min, tbot_max = safety.tbottom_range

axes[1].plot(sac_data["ttop"], "b-", lw=1.5, label="TTOP (SAC)")
axes[1].plot(sac_data["tmid"], "g-", lw=1.5, label="TMID (SAC)")
axes[1].plot(sac_data["tbottom"], color="orange", lw=1.5, label="TBOTTOM (SAC)")

# TTOP limits
axes[1].axhline(ttop_min, color="blue", ls="--", lw=1, alpha=0.5)
axes[1].axhline(ttop_max, color="blue", ls="--", lw=1, alpha=0.5,
                label="TTOP [%.0f, %.0f]" % (ttop_min, ttop_max))

# TMID limits
axes[1].axhline(tmid_min, color="green", ls="--", lw=1, alpha=0.5)
axes[1].axhline(tmid_max, color="green", ls="--", lw=1, alpha=0.5,
                label="TMID [%.0f, %.0f]" % (tmid_min, tmid_max))

# TBOTTOM limits
axes[1].axhline(tbot_min, color="orange", ls="--", lw=1, alpha=0.5)
axes[1].axhline(tbot_max, color="darkred", ls="--", lw=1, alpha=0.5,
                label="TBOTTOM [%.0f, %.0f]" % (tbot_min, tbot_max))

axes[1].set_xlabel("Step")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("11. Hard Safety Temperature Constraints (SAC)")
axes[1].legend(fontsize=7, loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_advanced.png", dpi=200, bbox_inches="tight")
plt.show()
print("   Saved: fig5_advanced.png")

print("")
print(">>> ALL 11 CHARTS DONE <<<")
print("   fig1_main_5charts.png")
print("   fig2_surrogate.png")
print("   fig3_tradeoff_action.png")
print("   fig4_boxplot_200eps.png")
print("   fig5_advanced.png")
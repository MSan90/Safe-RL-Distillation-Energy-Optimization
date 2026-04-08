"""
Level 6: Environment Core -- Safe RL for Distillation Column
Engineering spec: XD >= 0.95

Design:
  1. Soft termination: grace_steps before episode ends
  2. Reward:
      - Hard penalty below XD_hard (0.95)
      - Purity reward ramps up until XD_soft (0.985), then saturates (no extra reward near 1.0)
      - Energy penalty meaningful
      - Small smoothness penalty
  3. Safe reset: episodes start in feasible HeatDuty region (XD_ss >= XD_hard)
  4. FOPDT dynamics with proper dead-time initialization
"""
import numpy as np


class SimpleDeltaHeatEnv:
    def __init__(
        self,
        surrogate,
        dynamics_bank,
        safety_checker,
        max_steps=200,
        delta_heat_max=50.0,
        reward_mode="monotonic",
    ):
        self.surrogate = surrogate
        self.dynamics = dynamics_bank
        self.safety = safety_checker
        self.max_steps = int(max_steps)
        self.delta_heat_max = float(delta_heat_max)
        self.reward_mode = str(reward_mode)

        # --- Hard spec + economic margin targets ---
        self.xd_hard = 0.95          # hard/spec constraint (termination + safe reset)
        self.xd_soft = 0.985         # operating margin target (reward saturates here)

        # Backward-compat: other code may read xd_target
        self.xd_target = self.xd_hard

        self.grace_steps = 5
        self.violation_counter = 0

        self.obs_dim = 5

        self.heat_duty = None
        self.xd = None
        self.ttop = None
        self.tmid = None
        self.tbottom = None
        self.step_count = 0
        self.prev_heat = None

        # Track safety state properly
        self.last_safe_hard = True       # Full SafetyChecker result
        self.last_hard_violations = []   # List of violation descriptions

        self._find_safe_heat_range()

    def _find_safe_heat_range(self):
        heats = np.linspace(self.surrogate.heat_min, self.surrogate.heat_max, 500)
        safe_heats = []
        for h in heats:
            pred = self.surrogate.predict(h)
            if pred["XD_ss"] >= self.xd_hard:
                safe_heats.append(h)

        if len(safe_heats) > 0:
            self.safe_heat_min = float(min(safe_heats))
            self.safe_heat_max = float(max(safe_heats))
        else:
            self.safe_heat_min = float(self.surrogate.heat_min)
            self.safe_heat_max = float(self.surrogate.heat_max)
            print("[WARNING] No HeatDuty region achieves XD >= %.3f!" % self.xd_hard)

        print("[ENV] Safe HeatDuty range: [%.4f, %.4f]" % (self.safe_heat_min, self.safe_heat_max))

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.heat_duty = float(np.random.uniform(self.safe_heat_min, self.safe_heat_max))
        pred = self.surrogate.predict(self.heat_duty)

        self.xd = float(pred["XD_ss"])
        self.ttop = float(pred.get("TTOP_ss", 80.0))
        self.tmid = float(pred.get("TMID_ss", 100.0))
        self.tbottom = float(pred.get("TBOTTOM_ss", 120.0))

        self.step_count = 0
        self.violation_counter = 0
        self.prev_heat = float(self.heat_duty)

        # Reset safety tracking
        self.last_safe_hard = True
        self.last_hard_violations = []

        # Reset dynamics with initial steady-state values
        self.dynamics.reset(
            initial_values={
                "XD": self.xd,
                "TTOP": self.ttop,
                "TMID": self.tmid,
                "TBOTTOM": self.tbottom,
            }
        )

        return self._get_obs(), self._get_info()

    def step(self, action):
        # action expected shape: (1,) normalized in [-1,1]
        delta_heat = float(action[0]) * self.delta_heat_max
        new_heat = np.clip(
            self.heat_duty + delta_heat,
            self.surrogate.heat_min,
            self.surrogate.heat_max,
        )

        # Get steady-state targets from surrogate
        pred = self.surrogate.predict(new_heat)

        # Apply FOPDT dynamics (lag + dead-time)
        current = {
            "XD": self.xd,
            "TTOP": self.ttop,
            "TMID": self.tmid,
            "TBOTTOM": self.tbottom,
        }
        ss_targets = {
            "XD_ss": float(pred["XD_ss"]),
            "TTOP_ss": float(pred.get("TTOP_ss", 80.0)),
            "TMID_ss": float(pred.get("TMID_ss", 100.0)),
            "TBOTTOM_ss": float(pred.get("TBOTTOM_ss", 120.0)),
        }
        updated = self.dynamics.step_all(current, ss_targets)

        self.xd = float(updated["XD"])
        self.ttop = float(updated["TTOP"])
        self.tmid = float(updated["TMID"])
        self.tbottom = float(updated["TBOTTOM"])

        self.prev_heat = float(self.heat_duty)
        self.heat_duty = float(new_heat)
        self.step_count += 1

        # Safety check — store FULL result
        safety_result = self.safety.check(
            xd=self.xd, ttop=self.ttop, tmid=self.tmid, tbottom=self.tbottom
        )
        self.last_safe_hard = bool(safety_result["safe"])
        self.last_hard_violations = list(safety_result.get("violations", []))

        # Reward
        reward = self._compute_reward(safety_result)

        # Soft termination (spec violation: XD < xd_hard)
        if self.xd < self.xd_hard:
            self.violation_counter += 1
        else:
            self.violation_counter = 0

        terminated = False
        if self.violation_counter >= self.grace_steps:
            terminated = True
        if not safety_result["safe"]:
            terminated = True

        truncated = (self.step_count >= self.max_steps)

        info = self._get_info()
        info["violations"] = int(self.violation_counter)

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _compute_reward(self, safety_result):
        xd = float(self.xd)

        # --- Normalize heat like before (0..1) ---
        heat_range = float(self.surrogate.heat_max - self.surrogate.heat_min)
        heat_norm = (self.heat_duty - self.surrogate.heat_min) / (heat_range + 1e-9)
        heat_norm = float(np.clip(heat_norm, 0.0, 1.0))

        delta_heat = float(abs(self.heat_duty - self.prev_heat))

        if self.reward_mode == "monotonic":
            # 1) Purity: hard penalty below 0.95; saturate above 0.985
            if xd < self.xd_hard:
                r_purity = -800.0 * (self.xd_hard - xd) ** 1.5
            else:
                frac = (xd - self.xd_hard) / (self.xd_soft - self.xd_hard + 1e-9)
                frac = float(np.clip(frac, 0.0, 1.0))
                r_purity = 10.0 * frac

            # 2) Energy penalty: meaningful
            r_energy = -4.0 * heat_norm

            # 3) Smoothness
            r_smooth = -0.05 * (delta_heat / (self.delta_heat_max + 1e-9))

            return float(r_purity + r_energy + r_smooth)

        # Fallback (same for now)
        if xd < self.xd_hard:
            r_purity = -800.0 * (self.xd_hard - xd) ** 1.5
        else:
            frac = (xd - self.xd_hard) / (self.xd_soft - self.xd_hard + 1e-9)
            frac = float(np.clip(frac, 0.0, 1.0))
            r_purity = 10.0 * frac

        r_energy = -4.0 * heat_norm
        r_smooth = -0.05 * (delta_heat / (self.delta_heat_max + 1e-9))
        return float(r_purity + r_energy + r_smooth)

    def _get_obs(self):
        heat_range = float(self.surrogate.heat_max - self.surrogate.heat_min)
        heat_norm = (self.heat_duty - self.surrogate.heat_min) / (heat_range + 1e-9)
        heat_norm = float(np.clip(heat_norm, 0.0, 1.0))

        return np.array(
            [
                float(self.xd),
                float(heat_norm),
                float(self.ttop) / 200.0,
                float(self.tmid) / 200.0,
                float(self.tbottom) / 200.0,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "xd": float(self.xd),
            "heat_duty": float(self.heat_duty),
            "ttop": float(self.ttop),
            "tmid": float(self.tmid),
            "tbottom": float(self.tbottom),
            "step": int(self.step_count),
            # Three separate safety flags
            "safe": bool(self.last_safe_hard),                 # SafetyChecker (XD + temps)
            "safe_spec": bool(self.xd >= self.xd_hard),        # Only XD >= 0.95
            "safe_hard": bool(self.last_safe_hard),            # SafetyChecker (XD + temps)
            "hard_violations": list(self.last_hard_violations),
        }
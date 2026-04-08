"""
LEVEL 9: RL Training with SAC (Soft Actor-Critic)
Returns model + training rewards for learning curve.
"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class SafetyRewardCallback(BaseCallback):
    """Logs episode rewards + safety violations during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.violation_count = 0
        self.episode_count = 0
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        r = self.locals.get("rewards")
        d = self.locals.get("dones")
        r0 = float(r[0] if np.ndim(r) else r)
        d0 = bool(d[0] if np.ndim(d) else d)

        self._current_reward += r0

        infos = self.locals.get("infos", [])
        for info in infos:
            if not info.get("safe", True):
                self.violation_count += 1

        if d0:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
            self.episode_count += 1
            if self.verbose > 0 and self.episode_count % 100 == 0:
                print("[TRAIN] Episode %d | Violations: %d" %
                      (self.episode_count, self.violation_count))
        return True


def train_sac(env, total_timesteps=100_000, seed=42):
    """
    Train SAC and return model + training episode rewards.

    Returns:
        model:            trained SAC model
        episode_rewards:  list of episode rewards (for learning curve)
    """
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100_000,
        learning_starts=1000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
    )

    callback = SafetyRewardCallback(verbose=1)

    print("[TRAIN] Starting SAC training for %d steps..." % total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    print("[TRAIN] Done. Violations: %d | Episodes: %d" %
          (callback.violation_count, callback.episode_count))

    return model, callback.episode_rewards
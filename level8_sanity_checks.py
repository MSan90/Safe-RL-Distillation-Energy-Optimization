"""
Level 8: Sanity Checks
Quick checks to make sure the environment works before training.
"""
import numpy as np


def run_all_sanity_checks(env, n_random_steps=50):
    """Run basic sanity checks on the Gym environment."""

    errors = []
    print("[Sanity] Running checks...")

    # ── Check 1: Spaces exist ──
    if env.observation_space is None:
        errors.append("observation_space is None")
    if env.action_space is None:
        errors.append("action_space is None")

    # ── Check 2: Reset produces valid observation ──
    obs, info = env.reset(seed=42)
    obs_dim = env.observation_space.shape[0]

    if obs is None or len(obs) != obs_dim:
        errors.append(f"reset() returned invalid obs: {obs}")
    elif not np.all(np.isfinite(obs)):
        errors.append(f"reset() returned non-finite obs: {obs}")
    else:
        print(f"[Sanity] reset() OK — obs shape: {obs.shape}")

    # ── Check 3: Step with zero action ──
    zero_action = np.array([0.0], dtype=np.float32)
    obs2, reward, terminated, truncated, info2 = env.step(zero_action)

    if obs2 is None or len(obs2) != obs_dim:
        errors.append(f"step() returned invalid obs: {obs2}")
    elif not np.all(np.isfinite(obs2)):
        errors.append(f"step() returned non-finite obs: {obs2}")
    else:
        print(f"[Sanity] step(0) OK — reward: {reward:.4f}")

    # ── Check 4: Random rollout ──
    obs, _ = env.reset(seed=123)
    total_reward = 0.0
    for i in range(n_random_steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"[Sanity] Random rollout ({n_random_steps} steps) OK — total reward: {total_reward:.2f}")

    # ── Check 5: Info contains expected keys ──
    expected_keys = ["xd", "heat_duty", "safe"]
    for key in expected_keys:
        if key not in info:
            errors.append(f"info missing key: '{key}'")

    print(f"[Sanity] Info keys: {list(info.keys())}")

    # ── Results ──
    if errors:
        print("\n[Sanity] ❌ ERRORS FOUND:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("\n[Sanity] ✅ ALL CHECKS PASSED")

    return len(errors) == 0
"""Validate TrafficEnv state extraction against direct TraCI reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.traffic_env import TrafficEnv


def main() -> None:
    env = TrafficEnv(normalize_observations=False)
    try:
        obs, _ = env.reset()
        raw = env._get_raw_obs()  # Validation-only internal check.
        if not np.allclose(obs, raw, atol=1e-6):
            raise AssertionError("reset observation mismatch with raw TraCI state")
        print("reset validation passed; obs shape:", obs.shape)

        for i in range(10):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            raw = env._get_raw_obs()
            if not np.allclose(obs, raw, atol=1e-6):
                raise AssertionError(
                    f"step {i + 1} mismatch between env observation and raw TraCI state"
                )
        print("step validation passed for 10 random actions.")
    finally:
        env.close()
        print("env.close() done.")


if __name__ == "__main__":
    main()

"""Smoke test for TrafficEnv + SUMO / TraCI wiring."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.traffic_env import TrafficEnv


def main() -> None:
    env = TrafficEnv()
    try:
        obs, info = env.reset()
        print("reset() -> obs shape:", obs.shape, "info:", info)
        print("action meanings:", env.action_meanings())
        print("first 5 observation features:", env.observation_feature_names()[:5])

        for i in range(5):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            print(
                f"step {i + 1}: action={int(a)} reward={reward!r} "
                f"terminated={terminated} truncated={truncated} "
                f"obs[:4]={obs[:4].tolist()} ..."
            )
    finally:
        env.close()
        print("env.close() done.")


if __name__ == "__main__":
    main()

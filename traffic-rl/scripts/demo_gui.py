"""
demo_gui.py — Run a trained agent with the SUMO GUI open.

Opens sumo-gui so you can watch vehicles move through the intersection,
while the terminal prints what the agent sees and decides each step.

Usage:
    python scripts/demo_gui.py                    # PPO best model (default)
    python scripts/demo_gui.py --agent ppo
    python scripts/demo_gui.py --agent dqn
    python scripts/demo_gui.py --agent fixed
    python scripts/demo_gui.py --agent actuated
    python scripts/demo_gui.py --delay 200        # slow down GUI (ms per step)
    python scripts/demo_gui.py --steps 300        # number of steps to run
    python scripts/demo_gui.py --no-gui           # headless with terminal output only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from env.traffic_env import TrafficEnv


# ── Phase display helpers ──────────────────────────────────────────────────────

PHASE_LABELS = {
    0: "GREEN  EW",
    1: "YELLOW EW",
    2: "GREEN  NS",
    3: "YELLOW NS",
}


def phase_bar(phase: int) -> str:
    label = PHASE_LABELS.get(phase, f"PHASE {phase}")
    if "GREEN" in label:
        return f"[G] {label}"
    return f"[Y] {label}"


# ── Agent loaders ──────────────────────────────────────────────────────────────

def load_ppo(checkpoint: Path, state_dim: int) -> PPOAgent:
    with open(PROJECT_ROOT / "configs" / "ppo_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cpu")
    agent = PPOAgent(
        PPOConfig(
            state_dim=state_dim,
            action_dim=2,
            hidden_dim=int(cfg["hidden_dim"]),
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            adv_estimate_lambda=float(cfg["adv_estimate_lambda"]),
            clip_epsilon=float(cfg["clip_epsilon"]),
            update_epochs=int(cfg["update_epochs"]),
            device=device,
        )
    )
    agent.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    agent.model.eval()
    return agent


def load_dqn(checkpoint: Path, state_dim: int) -> DQNAgent:
    with open(PROJECT_ROOT / "configs" / "dqn_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cpu")
    agent = DQNAgent(
        DQNConfig(
            state_dim=state_dim,
            action_dim=2,
            hidden_dim=int(cfg["hidden_dim"]),
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            buffer_capacity=int(cfg["buffer_capacity"]),
            batch_size=int(cfg["batch_size"]),
            target_update_freq=int(cfg["target_update_freq"]),
            device=device,
        )
    )
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    agent.q_net.load_state_dict(data["q_net_state_dict"])
    agent.q_net.eval()
    return agent


# ── Agent decision helpers ─────────────────────────────────────────────────────

def ppo_decision(agent: PPOAgent, state: np.ndarray):
    """Returns (action, action_probs as list, value_estimate)."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.model(state_t)
        probs_t = torch.softmax(logits, dim=-1).squeeze(0)
    action = int(probs_t.argmax().item())
    return action, probs_t.tolist(), float(value.item())


def dqn_decision(agent: DQNAgent, state: np.ndarray):
    """Returns (action, q_values as list, None)."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_t = agent.q_net(state_t).squeeze(0)
    action = int(q_t.argmax().item())
    return action, q_t.tolist(), None


# ── Terminal display ───────────────────────────────────────────────────────────

def print_step(
    step: int,
    obs: np.ndarray,
    action: int,
    agent_type: str,
    aux,       # probs for PPO, q-values for DQN
    value,     # critic value for PPO, None for DQN
    info: dict,
    feature_names: list[str],
) -> None:
    phase = int(obs[0] * 3) if obs[0] <= 1.0 else int(obs[0])  # handle normalized or raw
    # Use info for exact phase if available
    phase = info.get("current_phase", phase)
    in_yellow = info.get("in_yellow", False)
    time_in_phase = info.get("time_in_phase", 0.0)

    action_name = "SWITCH" if action == 1 else "KEEP  "

    print(f"\n{'─'*70}")
    print(f"  Step {step:4d}  |  Phase: {phase_bar(phase)}  |  {time_in_phase:5.1f}s in phase")

    # Agent decision line
    if agent_type == "ppo" and aux is not None:
        p_keep, p_switch = aux[0], aux[1]
        val_str = f"  V={value:+.2f}" if value is not None else ""
        print(f"  Action: {action_name}  |  P(keep)={p_keep:.2f}  P(switch)={p_switch:.2f}{val_str}")
    elif agent_type == "dqn" and aux is not None:
        q_keep, q_switch = aux[0], aux[1]
        print(f"  Action: {action_name}  |  Q(keep)={q_keep:+.2f}  Q(switch)={q_switch:+.2f}")
    elif agent_type == "fixed":
        print(f"  Action: {action_name}  (fixed-time controller)")
    elif agent_type == "actuated":
        print(f"  Action: {action_name}  (actuated controller)")

    # Lane state (skip index 0 = phase)
    lane_features = feature_names[1:]  # remove 'phase_index'
    raw_obs = obs[1:]
    print("  Lanes:")
    for i in range(0, len(lane_features), 3):
        lane_base = lane_features[i].replace("_queue_length", "")
        q  = raw_obs[i]
        wt = raw_obs[i + 1]
        d  = raw_obs[i + 2]
        print(f"    {lane_base:20s}  queue={q:.2f}  wait={wt:.2f}s  density={d:.2f}")

    # Reward
    cars = info.get("cars_through", 0)
    wait = info.get("total_wait", 0.0)
    penalty = info.get("constraint_penalty", 0.0)
    reward = info.get("final_reward", 0.0)
    penalty_str = f"  penalty={penalty:+.1f}" if penalty != 0.0 else ""
    print(f"  Reward: cars_through={cars}  total_wait={wait:.1f}s{penalty_str}  → {reward:+.2f}")


# ── Fixed-time and actuated controllers ───────────────────────────────────────

class FixedTimeController:
    """Mirrors the fixed-time baseline logic: cycle through phases on a schedule."""
    def __init__(self, phase_durations: list[int]):
        self.phase_durations = phase_durations
        self._step = 0
        self._phase = 0
        self._steps_in_phase = 0

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        current_phase = info.get("current_phase", 0)
        time_in_phase = info.get("time_in_phase", 0.0)
        target_duration = self.phase_durations[current_phase]
        if time_in_phase >= target_duration:
            return 1  # switch
        return 0  # keep


class ActuatedController:
    def __init__(self, min_green: float = 10.0, max_green: float = 45.0,
                 demand_gap: float = 2.0, low_demand_threshold: float = 1.0):
        self.min_green = min_green
        self.max_green = max_green
        self.demand_gap = demand_gap
        self.low_threshold = low_demand_threshold

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        current_phase = info.get("current_phase", 0)
        time_in_phase = info.get("time_in_phase", 0.0)
        in_yellow = info.get("in_yellow", False)

        if in_yellow:
            return 0

        # Queue sums from raw obs (indices 1,4,7,10 are queue lengths)
        queues = obs[1::3]  # every 3rd starting at 1

        if time_in_phase < self.min_green:
            return 0
        if time_in_phase >= self.max_green:
            return 1

        # Determine which lanes are "active" for this green phase
        # Phase 0 = EW green (lanes 0,1), Phase 2 = NS green (lanes 2,3)
        if current_phase == 0:
            active_demand = float(np.sum(queues[:2]))
        else:
            active_demand = float(np.sum(queues[2:]))

        if active_demand < self.low_threshold and time_in_phase >= self.demand_gap:
            return 1
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained agent with SUMO GUI")
    parser.add_argument("--agent", choices=["ppo", "dqn", "fixed", "actuated"], default="ppo",
                        help="Which controller to use (default: ppo)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint (default: results/checkpoints/<agent>_best.pt)")
    parser.add_argument("--delay", type=int, default=100,
                        help="GUI delay in ms per simulation step (default: 100)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Max steps to run (default: 300)")
    parser.add_argument("--seed", type=int, default=4100,
                        help="SUMO random seed (default: 4100)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (no SUMO window), terminal output only")
    args = parser.parse_args()

    # ── Build environment ──────────────────────────────────────────────────────
    sumo_binary = "sumo" if args.no_gui else "sumo-gui"
    extra_flags: list[str] = []
    if not args.no_gui:
        extra_flags += ["--start", "--quit-on-end", f"--delay", str(args.delay)]

    env = TrafficEnv(
        sumo_binary=sumo_binary,
        normalize_observations=True,
        extra_sumo_flags=extra_flags,
    )

    obs, _ = env.reset(seed=args.seed)
    state_dim = env.observation_space.shape[0]
    feature_names = env.observation_feature_names()

    # ── Load controller ────────────────────────────────────────────────────────
    agent_type = args.agent
    rl_agent = None
    fixed_ctrl = None

    if agent_type in ("ppo", "dqn"):
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = PROJECT_ROOT / "results" / "checkpoints" / f"{agent_type}_best.pt"
        if not ckpt.exists():
            # Fall back to final checkpoint
            ckpt = PROJECT_ROOT / "results" / "checkpoints" / f"{agent_type}_final.pt"
        if not ckpt.exists():
            print(f"ERROR: No checkpoint found for '{agent_type}'. Train the agent first.")
            env.close()
            sys.exit(1)
        print(f"Loading {agent_type.upper()} checkpoint: {ckpt}")
        if agent_type == "ppo":
            rl_agent = load_ppo(ckpt, state_dim)
        else:
            rl_agent = load_dqn(ckpt, state_dim)
    elif agent_type == "fixed":
        fixed_ctrl = FixedTimeController(phase_durations=[42, 3, 42, 3])
        print("Using fixed-time controller (42s green, 3s yellow)")
    elif agent_type == "actuated":
        fixed_ctrl = ActuatedController()
        print("Using actuated controller (demand-responsive)")

    mode_str = "headless" if args.no_gui else f"sumo-gui (delay={args.delay}ms)"
    print(f"\nRunning {agent_type.upper()} agent | {mode_str} | max {args.steps} steps")
    print("Press Ctrl+C to stop early.\n")

    # ── Run loop ───────────────────────────────────────────────────────────────
    total_reward = 0.0
    total_cars = 0
    info: dict = {}

    try:
        for step in range(1, args.steps + 1):
            # Choose action
            aux = None
            value = None

            if agent_type == "ppo" and rl_agent is not None:
                action, aux, value = ppo_decision(rl_agent, obs)
            elif agent_type == "dqn" and rl_agent is not None:
                action, aux, value = dqn_decision(rl_agent, obs)
            elif fixed_ctrl is not None:
                action = fixed_ctrl.select_action(obs, info)
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_cars += int(info.get("cars_through", 0))

            print_step(step, obs, action, agent_type, aux, value, info, feature_names)

            if terminated or truncated:
                print("\n[Simulation ended early — all vehicles have departed]")
                break
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    finally:
        env.close()

    print(f"\n{'='*70}")
    print(f"  Episode summary: {step} steps | total_reward={total_reward:.2f} | cars_through={total_cars}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

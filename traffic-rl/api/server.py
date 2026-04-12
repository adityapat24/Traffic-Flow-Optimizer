"""
FastAPI WebSocket server — streams live simulation state to the dashboard.

Run from the traffic-rl/ directory:
    uvicorn api.server:app --reload --port 8000

Then open the frontend dashboard and use the "Live Sim" tab.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from env.traffic_env import TrafficEnv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only one SUMO simulation at a time — prevents port conflicts on quick restart
_sim_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _best_checkpoint(agent: str) -> Path | None:
    for name in [f"{agent}_best.pt", f"{agent}_final.pt"]:
        p = PROJECT_ROOT / "results" / "checkpoints" / name
        if p.exists():
            return p
    return None


def _load_ppo(state_dim: int) -> PPOAgent:
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
    ckpt = _best_checkpoint("ppo")
    if ckpt:
        agent.model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
    agent.model.eval()
    return agent


def _load_dqn(state_dim: int) -> DQNAgent:
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
    ckpt = _best_checkpoint("dqn")
    if ckpt:
        data = torch.load(ckpt, map_location=device, weights_only=True)
        agent.q_net.load_state_dict(data["q_net_state_dict"])
    agent.q_net.eval()
    return agent


# ── REST endpoint ──────────────────────────────────────────────────────────────

@app.get("/api/agents")
def get_agents() -> dict:
    return {
        "agents": [
            {"id": "ppo",      "label": "PPO (RL)",           "available": _best_checkpoint("ppo") is not None},
            {"id": "dqn",      "label": "DQN (RL)",           "available": _best_checkpoint("dqn") is not None},
            {"id": "fixed",    "label": "Fixed-Time Baseline", "available": True},
            {"id": "actuated", "label": "Actuated Baseline",   "available": True},
        ]
    }


@app.get("/api/results/{agent_type}")
async def get_results(
    agent_type: str,
    n_episodes: int = 5,
    steps: int = 300,
) -> dict:
    """Run *n_episodes* evaluation episodes for *agent_type* and return metrics."""
    valid = {"ppo", "dqn", "fixed", "actuated"}
    if agent_type not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{agent_type}'. Must be one of {sorted(valid)}.")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_eval, agent_type, n_episodes, steps)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return result


# ── Shared action selection ────────────────────────────────────────────────────

def _select_action(
    agent_type: str,
    obs: np.ndarray,
    info: dict[str, Any],
    rl_agent: PPOAgent | DQNAgent | None,
) -> int:
    if agent_type == "ppo" and isinstance(rl_agent, PPOAgent):
        t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = rl_agent.model(t)
            probs_t = torch.softmax(logits, dim=-1).squeeze(0)
        return int(probs_t.argmax().item())

    if agent_type == "dqn" and isinstance(rl_agent, DQNAgent):
        t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_t = rl_agent.q_net(t).squeeze(0)
        return int(q_t.argmax().item())

    if agent_type == "fixed":
        durations = [42, 3, 42, 3]
        tp = float(info.get("time_in_phase", 0.0))
        cp = int(info.get("current_phase", 0))
        return 1 if tp >= durations[cp] else 0

    if agent_type == "actuated":
        tp = float(info.get("time_in_phase", 0.0))
        cp = int(info.get("current_phase", 0))
        iy = bool(info.get("in_yellow", False))
        queues = obs[1::3]
        if iy or tp < 10:
            return 0
        if tp >= 45:
            return 1
        active = float(np.sum(queues[:2])) if cp == 0 else float(np.sum(queues[2:]))
        return 1 if active < 1.0 else 0

    return 0


# ── Multi-episode evaluation (used by REST endpoint) ──────────────────────────

def _run_eval(agent_type: str, n_episodes: int, steps: int) -> dict[str, Any]:
    """Run *n_episodes* episodes and return per-episode aggregated metrics."""
    if not _sim_lock.acquire(timeout=120):
        raise RuntimeError("Simulation lock unavailable — another simulation is running.")

    episodes: list[int] = []
    avg_waits: list[float] = []
    throughputs: list[int] = []
    queue_lengths: list[float] = []
    mses: list[float] = []

    try:
        # Load RL agent once; reuse across episodes.
        rl_agent: PPOAgent | DQNAgent | None = None
        # We need state_dim to load the agent — grab it from a quick env peek.
        probe_env = TrafficEnv(normalize_observations=True)
        try:
            probe_obs, _ = probe_env.reset(seed=4100)
            state_dim = probe_env.observation_space.shape[0]
            n_lanes = (len(probe_obs) - 1) // 3
        finally:
            probe_env.close()

        if agent_type == "ppo":
            rl_agent = _load_ppo(state_dim)
        elif agent_type == "dqn":
            rl_agent = _load_dqn(state_dim)

        for ep_idx in range(n_episodes):
            seed = 4100 + ep_idx
            env = TrafficEnv(normalize_observations=True)
            try:
                obs, _ = env.reset(seed=seed)
                info: dict[str, Any] = {}

                ep_total_wait = 0.0
                ep_total_cars = 0
                ep_total_queue = 0.0
                ep_wait_sq = 0.0
                ep_steps = 0

                for _ in range(steps):
                    action = _select_action(agent_type, obs, info, rl_agent)
                    obs, _reward, terminated, truncated, info = env.step(action)

                    step_wait = float(info.get("total_wait", 0.0))
                    step_cars = int(info.get("cars_through", 0))
                    # Sum of normalized queue values across all controlled lanes
                    step_queue = float(sum(obs[1 + 3 * i] for i in range(n_lanes)))

                    ep_total_wait += step_wait
                    ep_total_cars += step_cars
                    ep_total_queue += step_queue
                    ep_wait_sq += step_wait ** 2
                    ep_steps += 1

                    if terminated or truncated:
                        break

            finally:
                env.close()

            denom = max(ep_steps, 1)
            episodes.append(ep_idx + 1)
            avg_waits.append(round(ep_total_wait / denom, 1))
            throughputs.append(ep_total_cars)
            queue_lengths.append(round(ep_total_queue / denom, 3))
            mses.append(round(ep_wait_sq / denom, 1))

    finally:
        _sim_lock.release()

    return {
        "episodes": episodes,
        "avg_wait": avg_waits,
        "throughput": throughputs,
        "queue_length": queue_lengths,
        "mse": mses,
    }


# ── Simulation thread ──────────────────────────────────────────────────────────

def _simulate(
    agent_type: str,
    steps: int,
    seed: int,
    stop_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> None:
    if not _sim_lock.acquire(timeout=10):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "error", "message": "Another simulation is still shutting down. Please wait a moment and try again."},
        )
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return

    env = TrafficEnv(normalize_observations=True)
    try:
        obs, _ = env.reset(seed=seed)
        lane_ids: list[str] = list(env._lane_ids)
        state_dim = env.observation_space.shape[0]

        rl_agent: PPOAgent | DQNAgent | None = None
        if agent_type == "ppo":
            rl_agent = _load_ppo(state_dim)
        elif agent_type == "dqn":
            rl_agent = _load_dqn(state_dim)

        total_reward = 0.0
        total_cars = 0
        info: dict[str, Any] = {}

        for step in range(1, steps + 1):
            if stop_event.is_set():
                break

            # ── Choose action (with extra detail for WebSocket streaming) ──
            probs: list[float] | None = None
            q_values: list[float] | None = None
            value: float | None = None

            if agent_type == "ppo" and isinstance(rl_agent, PPOAgent):
                t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, val = rl_agent.model(t)
                    probs_t = torch.softmax(logits, dim=-1).squeeze(0)
                action = int(probs_t.argmax().item())
                probs = probs_t.tolist()
                value = float(val.item())
            elif agent_type == "dqn" and isinstance(rl_agent, DQNAgent):
                t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_t = rl_agent.q_net(t).squeeze(0)
                action = int(q_t.argmax().item())
                q_values = q_t.tolist()
            else:
                action = _select_action(agent_type, obs, info, rl_agent)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_cars += int(info.get("cars_through", 0))

            # ── Build lane snapshot ────────────────────────────────────────
            lanes = []
            for i, lid in enumerate(lane_ids):
                base = 1 + 3 * i
                lanes.append({
                    "id": lid,
                    "queue":   float(obs[base]),
                    "wait":    float(obs[base + 1]),
                    "density": float(obs[base + 2]),
                })

            msg: dict[str, Any] = {
                "type":                 "step",
                "step":                 step,
                "phase":                int(info.get("current_phase", 0)),
                "in_yellow":            bool(info.get("in_yellow", False)),
                "time_in_phase":        float(info.get("time_in_phase", 0.0)),
                "action":               int(action),
                "action_name":          "SWITCH" if action == 1 else "KEEP",
                "probs":                probs,
                "q_values":             q_values,
                "value":                value,
                "lanes":                lanes,
                "reward":               float(reward),
                "cars_through":         int(info.get("cars_through", 0)),
                "total_wait":           float(info.get("total_wait", 0.0)),
                "constraint_penalty":   float(info.get("constraint_penalty", 0.0)),
                "episode_total_reward": total_reward,
                "episode_total_cars":   total_cars,
            }
            loop.call_soon_threadsafe(queue.put_nowait, msg)

            if terminated or truncated:
                break

    except Exception as exc:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "error", "message": str(exc)},
        )
    finally:
        env.close()
        _sim_lock.release()
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel → done


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/simulate")
async def ws_simulate(
    websocket: WebSocket,
    agent: str = "ppo",
    seed: int = 4100,
    steps: int = 300,
) -> None:
    await websocket.accept()
    loop = asyncio.get_event_loop()
    msg_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=_simulate,
        args=(agent, steps, seed, stop_event, loop, msg_queue),
        daemon=True,
    )
    thread.start()

    try:
        while True:
            item = await msg_queue.get()
            if item is None:
                await websocket.send_json({"type": "done"})
                break
            await websocket.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=5)

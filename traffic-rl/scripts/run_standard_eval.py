from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Protocol
import xml.etree.ElementTree as ET

import numpy as np
import torch
import traci

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from env.traffic_env import TrafficEnv
from experiment_utils import DEFAULT_EXPERIMENT_CONFIG, collect_run_metadata

DEFAULT_SCENARIOS = [PROJECT_ROOT / "sumo" / "sim.sumocfg"]
DEFAULT_SEEDS = [4100, 4101, 4102, 4103, 4104]
DEFAULT_HORIZON = 1000
DEFAULT_CONTROLLERS = ("fixed_time", "actuated", "dqn", "ppo")
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

@dataclass
class EpisodeMetrics:
    controller: str
    scenario: str
    seed: int
    steps: int
    mean_waiting_time: float
    mean_queue_length: float
    throughput: float
    mean_travel_time: float
    travel_time_mse: float
    completed_trips: int


class Controller(Protocol):
    name: str

    def reset(self, env: TrafficEnv) -> None:
        ...

    def act(self, obs: np.ndarray, env: TrafficEnv) -> int:
        ...


@dataclass(frozen=True)
class TravelSegment:
    length: float
    speed: float

    def free_flow_seconds(self, vehicle_max_speed: float | None = None) -> float:
        effective_speed = self.speed
        if vehicle_max_speed is not None:
            effective_speed = min(effective_speed, vehicle_max_speed)
        return self.length / max(effective_speed, 1e-6)


@dataclass(frozen=True)
class TravelTimePredictor:
    edge_segments: dict[str, TravelSegment]
    connection_segments: dict[tuple[str, str], TravelSegment]

    def predict_route_time(
        self,
        route_edges: list[str],
        vehicle_max_speed: float | None = None,
    ) -> float:
        total_seconds = 0.0

        for edge_id in route_edges:
            segment = self.edge_segments.get(edge_id)
            if segment is not None:
                total_seconds += segment.free_flow_seconds(vehicle_max_speed)

        for from_edge, to_edge in zip(route_edges, route_edges[1:]):
            segment = self.connection_segments.get((from_edge, to_edge))
            if segment is not None:
                total_seconds += segment.free_flow_seconds(vehicle_max_speed)

        return total_seconds


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_scenarios(raw: str) -> list[Path]:
    scenarios = [resolve_path(item) for item in parse_csv_list(raw)]
    if not scenarios:
        raise ValueError("Expected at least one scenario path.")
    for scenario in scenarios:
        if not scenario.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario}")
    return scenarios


def resolve_sumo_input_path(sumocfg_path: Path, value: str) -> Path:
    return (sumocfg_path.parent / value).resolve()


def load_travel_time_predictor(sumocfg_path: Path) -> TravelTimePredictor:
    sumocfg_root = ET.parse(sumocfg_path).getroot()
    input_node = sumocfg_root.find("input")
    if input_node is None:
        raise RuntimeError(f"SUMO config '{sumocfg_path}' is missing an <input> section.")

    net_file_node = input_node.find("net-file")
    if net_file_node is None:
        raise RuntimeError(f"SUMO config '{sumocfg_path}' is missing a <net-file> entry.")

    net_file_value = net_file_node.attrib.get("value", "").strip()
    if not net_file_value:
        raise RuntimeError(f"SUMO config '{sumocfg_path}' has an empty net-file value.")

    net_path = resolve_sumo_input_path(sumocfg_path, net_file_value)
    if not net_path.exists():
        raise FileNotFoundError(f"SUMO network file not found: {net_path}")

    net_root = ET.parse(net_path).getroot()

    lane_segments: dict[str, TravelSegment] = {}
    edge_segments: dict[str, TravelSegment] = {}

    for edge_node in net_root.findall("edge"):
        edge_id = edge_node.attrib.get("id")
        if not edge_id:
            continue

        segments_for_edge: list[TravelSegment] = []
        for lane_node in edge_node.findall("lane"):
            length = float(lane_node.attrib["length"])
            speed = float(lane_node.attrib["speed"])
            segment = TravelSegment(length=length, speed=speed)
            lane_id = lane_node.attrib.get("id")
            if lane_id:
                lane_segments[lane_id] = segment
            segments_for_edge.append(segment)

        if edge_node.attrib.get("function") == "internal" or not segments_for_edge:
            continue

        edge_segments[edge_id] = min(
            segments_for_edge,
            key=lambda segment: segment.free_flow_seconds(),
        )

    connection_segments: dict[tuple[str, str], TravelSegment] = {}
    for connection_node in net_root.findall("connection"):
        from_edge = connection_node.attrib.get("from")
        to_edge = connection_node.attrib.get("to")
        via_lane_id = connection_node.attrib.get("via")
        if not from_edge or not to_edge or not via_lane_id:
            continue

        via_segment = lane_segments.get(via_lane_id)
        if via_segment is not None:
            connection_segments[(from_edge, to_edge)] = via_segment

    return TravelTimePredictor(
        edge_segments=edge_segments,
        connection_segments=connection_segments,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def current_mean_waiting_time() -> float:
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        return 0.0
    waits = [float(traci.vehicle.getWaitingTime(vehicle_id)) for vehicle_id in vehicle_ids]
    return float(mean(waits))


def current_total_queue_length(lane_ids: list[str]) -> float:
    return float(sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids))


def register_vehicle_events(
    departure_times: dict[str, float],
    predicted_travel_times: dict[str, float],
    travel_times: list[float],
    squared_errors: list[float],
    predictor: TravelTimePredictor,
) -> int:
    sim_time = float(traci.simulation.getTime())
    depart_time = max(sim_time - 1.0, 0.0)

    for vehicle_id in traci.simulation.getDepartedIDList():
        departure_times.setdefault(vehicle_id, depart_time)
        predicted_travel_times[vehicle_id] = predictor.predict_route_time(
            route_edges=list(traci.vehicle.getRoute(vehicle_id)),
            vehicle_max_speed=float(traci.vehicle.getMaxSpeed(vehicle_id)),
        )

    arrived_ids = list(traci.simulation.getArrivedIDList())
    for vehicle_id in arrived_ids:
        start_time = departure_times.pop(vehicle_id, None)
        if start_time is not None:
            actual_travel_time = sim_time - start_time
            travel_times.append(actual_travel_time)
            predicted_travel_time = predicted_travel_times.pop(vehicle_id, None)
            if predicted_travel_time is not None:
                squared_errors.append((actual_travel_time - predicted_travel_time) ** 2)

    return len(arrived_ids)


def build_green_phase_lane_map(tl_id: str, green_phase_ids: list[int]) -> dict[int, set[str]]:
    logics = traci.trafficlight.getAllProgramLogics(tl_id)
    if not logics:
        raise RuntimeError(f"No program logic found for traffic light '{tl_id}'.")

    phases = logics[0].phases
    links_by_index = traci.trafficlight.getControlledLinks(tl_id)
    phase_lanes: dict[int, set[str]] = {}

    for phase_idx in green_phase_ids:
        if phase_idx >= len(phases):
            raise RuntimeError(f"Phase index {phase_idx} is out of range for '{tl_id}'.")

        lanes: set[str] = set()
        for signal_idx, signal_state in enumerate(phases[phase_idx].state):
            if signal_idx >= len(links_by_index):
                continue
            if signal_state not in ("G", "g"):
                continue

            for link in links_by_index[signal_idx]:
                if link and link[0]:
                    lanes.add(link[0])

        phase_lanes[phase_idx] = lanes

    return phase_lanes


def phase_demand(phase_idx: int, phase_lanes: dict[int, set[str]]) -> float:
    lanes = phase_lanes.get(phase_idx, set())
    if not lanes:
        return 0.0

    halted = sum(float(traci.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lanes)
    approaching = sum(float(traci.lane.getLastStepVehicleNumber(lane_id)) for lane_id in lanes)
    return halted + 0.5 * approaching


class FixedTimeController:
    name = "fixed_time"

    def __init__(self, green_durations: list[int]) -> None:
        if not green_durations:
            raise ValueError("Fixed-time controller requires at least one green duration.")
        if any(duration <= 0 for duration in green_durations):
            raise ValueError("Fixed-time green durations must be positive.")
        self.green_durations = green_durations

    def reset(self, env: TrafficEnv) -> None:
        if len(self.green_durations) != len(env.green_phases):
            raise ValueError(
                "Number of fixed-time green durations must match env.green_phases. "
                f"Got {len(self.green_durations)} durations for {len(env.green_phases)} green phases."
            )

    def act(self, obs: np.ndarray, env: TrafficEnv) -> int:
        if env.in_yellow:
            return 0

        current_time = float(traci.simulation.getTime())
        time_in_phase = current_time - float(env.phase_start_time)
        green_idx = env.green_phases.index(env.current_phase)
        return int(time_in_phase >= float(self.green_durations[green_idx]))


class ActuatedController:
    name = "actuated"

    def __init__(
        self,
        *,
        min_green: int,
        max_green: int,
        demand_gap: float,
        low_demand_threshold: float,
    ) -> None:
        if min_green <= 0 or max_green <= 0:
            raise ValueError("Actuated min/max green must be > 0.")
        if min_green > max_green:
            raise ValueError("Actuated min_green must be <= max_green.")

        self.min_green = min_green
        self.max_green = max_green
        self.demand_gap = demand_gap
        self.low_demand_threshold = low_demand_threshold
        self.phase_lanes: dict[int, set[str]] = {}

    def reset(self, env: TrafficEnv) -> None:
        tl_ids = traci.trafficlight.getIDList()
        if not tl_ids:
            raise RuntimeError("No traffic lights found in SUMO scenario.")
        self.phase_lanes = build_green_phase_lane_map(tl_ids[0], env.green_phases)

    def act(self, obs: np.ndarray, env: TrafficEnv) -> int:
        if env.in_yellow:
            return 0

        current_time = float(traci.simulation.getTime())
        time_in_phase = current_time - float(env.phase_start_time)
        if time_in_phase < float(self.min_green):
            return 0

        if time_in_phase >= float(self.max_green):
            return 1

        current_demand = phase_demand(env.current_phase, self.phase_lanes)
        competing_demand = max(
            (
                phase_demand(phase_idx, self.phase_lanes)
                for phase_idx in env.green_phases
                if phase_idx != env.current_phase
            ),
            default=0.0,
        )

        low_current = current_demand <= self.low_demand_threshold
        strong_competitor = competing_demand >= current_demand + self.demand_gap
        return int(low_current or strong_competitor)


class DQNPolicyController:
    name = "dqn"

    def __init__(self, *, config_path: Path, checkpoint_path: Path) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config = load_json(config_path)
        self.device = torch.device(
            "cuda" if bool(self.config.get("use_cuda", False)) and torch.cuda.is_available() else "cpu"
        )
        self.agent: DQNAgent | None = None
        self.state_dim: int | None = None
        self.action_dim: int | None = None

    def reset(self, env: TrafficEnv) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {self.checkpoint_path}. "
                "Train DQN first or pass --dqn-checkpoint."
            )

        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)

        if self.agent is None:
            agent_cfg = DQNConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=int(self.config["hidden_dim"]),
                learning_rate=float(self.config["learning_rate"]),
                gamma=float(self.config["gamma"]),
                buffer_capacity=int(self.config["buffer_capacity"]),
                batch_size=int(self.config["batch_size"]),
                target_update_freq=int(self.config["target_update_freq"]),
                device=self.device,
            )
            self.agent = DQNAgent(agent_cfg)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            target_state = checkpoint.get("target_net_state_dict", checkpoint["q_net_state_dict"])
            self.agent.target_net.load_state_dict(target_state)
            self.agent.q_net.eval()
            self.agent.target_net.eval()
            self.state_dim = state_dim
            self.action_dim = action_dim
        else:
            if state_dim != self.state_dim or action_dim != self.action_dim:
                raise ValueError(
                    "DQN checkpoint dimensions do not match the evaluation scenario set."
                )

    def act(self, obs: np.ndarray, env: TrafficEnv) -> int:
        assert self.agent is not None
        return int(self.agent.select_action(obs, epsilon=0.0))


class PPOPolicyController:
    name = "ppo"

    def __init__(self, *, config_path: Path, checkpoint_path: Path) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config = load_json(config_path)
        self.device = torch.device(
            "cuda" if bool(self.config.get("use_cuda", False)) and torch.cuda.is_available() else "cpu"
        )
        self.agent: PPOAgent | None = None
        self.state_dim: int | None = None
        self.action_dim: int | None = None

    def reset(self, env: TrafficEnv) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: {self.checkpoint_path}. "
                "Train PPO first or pass --ppo-checkpoint."
            )

        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)

        if self.agent is None:
            agent_cfg = PPOConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=int(self.config["hidden_dim"]),
                learning_rate=float(self.config["learning_rate"]),
                gamma=float(self.config["gamma"]),
                adv_estimate_lambda=float(self.config["adv_estimate_lambda"]),
                clip_epsilon=float(self.config["clip_epsilon"]),
                update_epochs=int(self.config["update_epochs"]),
                device=self.device,
            )
            self.agent = PPOAgent(agent_cfg)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.agent.model.load_state_dict(state_dict)
            self.agent.model.eval()
            self.state_dim = state_dim
            self.action_dim = action_dim
        else:
            if state_dim != self.state_dim or action_dim != self.action_dim:
                raise ValueError(
                    "PPO checkpoint dimensions do not match the evaluation scenario set."
                )

    def act(self, obs: np.ndarray, env: TrafficEnv) -> int:
        assert self.agent is not None
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.agent.model(obs_t)
        return int(torch.argmax(logits, dim=1).item())


def build_controllers(args: argparse.Namespace) -> list[Controller]:
    requested = [name.strip() for name in args.controllers.split(",") if name.strip()]
    if not requested:
        raise ValueError("Expected at least one controller in --controllers.")

    valid = set(DEFAULT_CONTROLLERS)
    invalid = [name for name in requested if name not in valid]
    if invalid:
        raise ValueError(f"Unknown controller(s): {', '.join(invalid)}")

    controllers: list[Controller] = []
    for name in requested:
        if name == "fixed_time":
            controllers.append(FixedTimeController(green_durations=parse_int_csv(args.fixed_green_steps)))
        elif name == "actuated":
            controllers.append(
                ActuatedController(
                    min_green=args.actuated_min_green,
                    max_green=args.actuated_max_green,
                    demand_gap=args.actuated_demand_gap,
                    low_demand_threshold=args.actuated_low_demand_threshold,
                )
            )
        elif name == "dqn":
            controllers.append(
                DQNPolicyController(
                    config_path=resolve_path(args.dqn_config),
                    checkpoint_path=resolve_path(args.dqn_checkpoint),
                )
            )
        elif name == "ppo":
            controllers.append(
                PPOPolicyController(
                    config_path=resolve_path(args.ppo_config),
                    checkpoint_path=resolve_path(args.ppo_checkpoint),
                )
            )
    return controllers


def run_episode(
    *,
    env: TrafficEnv,
    controller: Controller,
    scenario_path: Path,
    seed: int,
    horizon: int,
    travel_time_predictor: TravelTimePredictor,
) -> EpisodeMetrics:
    set_global_seed(seed)
    obs, _ = env.reset(seed=seed)
    controller.reset(env)

    tl_ids = traci.trafficlight.getIDList()
    if not tl_ids:
        raise RuntimeError("No traffic lights found in SUMO scenario.")
    lane_ids = sorted(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_ids[0])))

    waiting_trace: list[float] = []
    queue_trace: list[float] = []
    travel_times: list[float] = []
    squared_errors: list[float] = []
    departure_times: dict[str, float] = {}
    predicted_travel_times: dict[str, float] = {}
    completed_trips = 0
    steps = 0

    while steps < horizon:
        action = controller.act(obs, env)
        obs, _reward, terminated, truncated, _info = env.step(action)

        steps += 1
        completed_trips += register_vehicle_events(
            departure_times=departure_times,
            predicted_travel_times=predicted_travel_times,
            travel_times=travel_times,
            squared_errors=squared_errors,
            predictor=travel_time_predictor,
        )
        waiting_trace.append(current_mean_waiting_time())
        queue_trace.append(current_total_queue_length(lane_ids))

        if terminated or truncated:
            break

    return EpisodeMetrics(
        controller=controller.name,
        scenario=display_path(scenario_path),
        seed=seed,
        steps=steps,
        mean_waiting_time=float(mean(waiting_trace)) if waiting_trace else 0.0,
        mean_queue_length=float(mean(queue_trace)) if queue_trace else 0.0,
        throughput=float(completed_trips) / max(steps, 1),
        mean_travel_time=float(mean(travel_times)) if travel_times else 0.0,
        travel_time_mse=float(mean(squared_errors)) if squared_errors else 0.0,
        completed_trips=completed_trips,
    )


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "variance": 0.0, "ci95": 0.0}

    avg = float(mean(values))
    if len(values) == 1:
        return {"mean": avg, "std": 0.0, "variance": 0.0, "ci95": 0.0}

    std = float(stdev(values))
    return {
        "mean": avg,
        "std": std,
        "variance": std * std,
        "ci95": 1.96 * std / math.sqrt(len(values)),
    }


def summarize_controller(episodes: list[EpisodeMetrics]) -> dict:
    if not episodes:
        raise ValueError("Cannot summarize an empty episode list.")

    by_seed: dict[int, list[EpisodeMetrics]] = {}
    for episode in episodes:
        by_seed.setdefault(episode.seed, []).append(episode)

    def seed_means(field: str) -> list[float]:
        return [
            float(mean(getattr(ep, field) for ep in seed_episodes))
            for _, seed_episodes in sorted(by_seed.items())
        ]

    metrics = {
        "mean_waiting_time": summarize_metric(seed_means("mean_waiting_time")),
        "mean_queue_length": summarize_metric(seed_means("mean_queue_length")),
        "throughput": summarize_metric(seed_means("throughput")),
        "mean_travel_time": summarize_metric(seed_means("mean_travel_time")),
        "travel_time_mse": summarize_metric(seed_means("travel_time_mse")),
    }

    return {
        "controller": episodes[0].controller,
        "episodes": len(episodes),
        "seed_count": len(by_seed),
        "scenario_count": len({episode.scenario for episode in episodes}),
        "total_completed_trips": int(sum(ep.completed_trips for ep in episodes)),
        "metrics": metrics,
    }


def format_mean_ci(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.3f} +/- {stats['ci95']:.3f}"


def print_comparison_table(summaries: list[dict]) -> None:
    headers = [
        "controller",
        "episodes",
        "wait (mean+/-ci95)",
        "queue (mean+/-ci95)",
        "throughput (mean+/-ci95)",
        "travel (mean+/-ci95)",
        "travel_mse (mean+/-ci95)",
    ]

    rows = []
    for summary in summaries:
        rows.append(
            [
                summary["controller"],
                str(summary["episodes"]),
                format_mean_ci(summary["metrics"]["mean_waiting_time"]),
                format_mean_ci(summary["metrics"]["mean_queue_length"]),
                format_mean_ci(summary["metrics"]["throughput"]),
                format_mean_ci(summary["metrics"]["mean_travel_time"]),
                format_mean_ci(summary["metrics"]["travel_time_mse"]),
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def flatten_summary(summary: dict) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "controller": summary["controller"],
        "episodes": summary["episodes"],
        "seed_count": summary["seed_count"],
        "scenario_count": summary["scenario_count"],
        "total_completed_trips": summary["total_completed_trips"],
    }

    for metric_name, stats in summary["metrics"].items():
        row[f"{metric_name}_mean"] = stats["mean"]
        row[f"{metric_name}_std"] = stats["std"]
        row[f"{metric_name}_variance"] = stats["variance"]
        row[f"{metric_name}_ci95"] = stats["ci95"]

    return row


def save_outputs(
    *,
    args: argparse.Namespace,
    episodes: list[EpisodeMetrics],
    summaries: list[dict],
    output_prefix: Path,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metadata": collect_run_metadata(
            config_path=args.experiment_config if args.experiment_config.exists() else None,
            resolved_config=args._experiment_cfg if hasattr(args, "_experiment_cfg") else {},
            script_name="run_standard_eval.py",
        ),
        "config": {
            "controllers": [summary["controller"] for summary in summaries],
            "scenario_set": [display_path(resolve_path(path)) for path in args.sumocfgs.split(",") if path.strip()],
            "seeds": parse_int_csv(args.seeds),
            "episodes_per_controller": len(parse_int_csv(args.seeds)) * len(parse_scenarios(args.sumocfgs)),
            "horizon": args.horizon,
            "sumo_binary": args.sumo_binary,
            "fixed_green_steps": parse_int_csv(args.fixed_green_steps),
            "actuated_min_green": args.actuated_min_green,
            "actuated_max_green": args.actuated_max_green,
            "actuated_demand_gap": args.actuated_demand_gap,
            "actuated_low_demand_threshold": args.actuated_low_demand_threshold,
        },
        "metric_definitions": {
            "mean_waiting_time": "Per-step mean waiting time across active vehicles, averaged over the episode.",
            "mean_queue_length": "Per-step total halting vehicles across the controlled intersection, averaged over the episode.",
            "throughput": "Completed trips per simulation step.",
            "mean_travel_time": "Average trip duration for vehicles that completed within the episode horizon.",
            "travel_time_mse": "Per-episode mean squared error between each completed vehicle's predicted free-flow travel time and its actual travel time.",
            "uncertainty": "95% confidence intervals computed across seed-level aggregates.",
        },
        "travel_time_prediction_methodology": {
            "predicted_travel_time": (
                "For each departed vehicle, predicted travel time is computed as the "
                "sum of free-flow times across every route edge plus the internal "
                "junction connector between each consecutive edge pair. Each segment "
                "free-flow time is segment_length / min(segment_speed_limit, vehicle_max_speed_at_departure)."
            ),
            "actual_travel_time": (
                "For each completed vehicle, actual travel time is arrival_sim_time - departure_sim_time."
            ),
            "travel_time_mse": (
                "Per-episode travel_time_mse = mean((actual_travel_time - predicted_travel_time)^2) "
                "over vehicles that both departed and arrived within the episode horizon."
            ),
        },
        "summary": summaries,
        "episodes": [asdict(episode) for episode in episodes],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [flatten_summary(summary) for summary in summaries]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one standard evaluation process for RL controllers and baselines."
    )
    parser.add_argument(
        "--controllers",
        default=",".join(DEFAULT_CONTROLLERS),
        help="Comma-separated subset of: fixed_time,actuated,dqn,ppo",
    )
    parser.add_argument(
        "--sumocfgs",
        default=",".join(display_path(path) for path in DEFAULT_SCENARIOS),
        help="Comma-separated SUMO config paths defining the fixed scenario set.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated episode seeds shared by every controller.",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--sumo-binary", default="sumo")

    parser.add_argument(
        "--fixed-green-steps",
        default="42,42",
        help="Green durations for the fixed-time baseline, ordered by env.green_phases.",
    )

    parser.add_argument("--actuated-min-green", type=int, default=10)
    parser.add_argument("--actuated-max-green", type=int, default=45)
    parser.add_argument("--actuated-demand-gap", type=float, default=2.0)
    parser.add_argument("--actuated-low-demand-threshold", type=float, default=1.0)

    parser.add_argument("--dqn-config", default="configs/dqn_config.json")
    parser.add_argument("--dqn-checkpoint", default="results/checkpoints/dqn_best.pt")

    parser.add_argument("--ppo-config", default="configs/ppo_config.json")
    parser.add_argument("--ppo-checkpoint", default="results/checkpoints/ppo_best.pt")

    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output path prefix. Defaults to results/evaluation/controller_eval_<timestamp>.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Centralized experiment config JSON file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_cfg: dict = {}
    if args.experiment_config and args.experiment_config.exists():
        experiment_cfg = load_json(args.experiment_config)
        env_cfg = experiment_cfg.get("environment", {})
        eval_cfg = experiment_cfg.get("evaluation", {})
        repro_cfg = experiment_cfg.get("reproducibility", {})
        agents_cfg = experiment_cfg.get("agents", {})
        base_cfg = experiment_cfg.get("baselines", {})
        fixed_cfg = base_cfg.get("fixed_time", {})
        act_cfg = base_cfg.get("actuated", {})

        if env_cfg.get("sumocfg_path"):
            args.sumocfgs = str(env_cfg["sumocfg_path"])
        args.sumo_binary = env_cfg.get("sumo_binary", args.sumo_binary)
        args.horizon = int(eval_cfg.get("horizon", args.horizon))
        if repro_cfg.get("seeds"):
            args.seeds = ",".join(str(x) for x in repro_cfg["seeds"])
        if fixed_cfg.get("phase_durations"):
            phases = [int(x) for x in fixed_cfg["phase_durations"]]
            args.fixed_green_steps = ",".join(str(phases[i]) for i in range(0, len(phases), 2))
        args.actuated_min_green = int(act_cfg.get("min_green", args.actuated_min_green))
        args.actuated_max_green = int(act_cfg.get("max_green", args.actuated_max_green))
        args.actuated_demand_gap = float(act_cfg.get("demand_gap", args.actuated_demand_gap))
        args.actuated_low_demand_threshold = float(
            act_cfg.get("low_demand_threshold", args.actuated_low_demand_threshold)
        )
        args.dqn_config = agents_cfg.get("dqn_config_path", args.dqn_config)
        args.ppo_config = agents_cfg.get("ppo_config_path", args.ppo_config)
    args._experiment_cfg = experiment_cfg

    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0")

    scenarios = parse_scenarios(args.sumocfgs)
    seeds = parse_int_csv(args.seeds)
    controllers = build_controllers(args)

    all_episodes: list[EpisodeMetrics] = []
    summaries: list[dict] = []

    for controller in controllers:
        controller_episodes: list[EpisodeMetrics] = []

        for scenario_path in scenarios:
            travel_time_predictor = load_travel_time_predictor(scenario_path)
            env = TrafficEnv(
                sumocfg_path=scenario_path,
                sumo_binary=args.sumo_binary,
                normalize_observations=True,
            )
            try:
                for seed in seeds:
                    episode = run_episode(
                        env=env,
                        controller=controller,
                        scenario_path=scenario_path,
                        seed=seed,
                        horizon=args.horizon,
                        travel_time_predictor=travel_time_predictor,
                    )
                    controller_episodes.append(episode)
                    print(
                        f"[{episode.controller}] scenario={episode.scenario} seed={episode.seed} "
                        f"steps={episode.steps} wait={episode.mean_waiting_time:.3f} "
                        f"queue={episode.mean_queue_length:.3f} throughput={episode.throughput:.3f} "
                        f"travel={episode.mean_travel_time:.3f} travel_mse={episode.travel_time_mse:.3f}"
                    )
            finally:
                env.close()

        summaries.append(summarize_controller(controller_episodes))
        all_episodes.extend(controller_episodes)

    print("\n=== Controller Comparison ===")
    print_comparison_table(summaries)

    if args.output_prefix:
        output_prefix = resolve_path(args.output_prefix)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_prefix = RESULTS_DIR / f"controller_eval_{stamp}"

    json_path, csv_path = save_outputs(
        args=args,
        episodes=all_episodes,
        summaries=summaries,
        output_prefix=output_prefix,
    )

    # ===== Save comparison plots (Ticket 13) =====
    from results_utils import save_bar_plot

    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = [flatten_summary(summary) for summary in summaries]
    controllers = [row["controller"] for row in rows]

    def make_plot(metric_key, title, filename):
        values = [row[f"{metric_key}_mean"] for row in rows]
        errors = [row.get(f"{metric_key}_ci95", 0.0) for row in rows]

        save_bar_plot(
            controllers,
            values,
            yerr=errors,
            ylabel=title,
            title=f"{title} by Controller",
            path=plots_dir / filename,
        )

    make_plot("mean_waiting_time", "Mean Waiting Time", "eval_wait.png")
    make_plot("mean_queue_length", "Mean Queue Length", "eval_queue.png")
    make_plot("throughput", "Throughput", "eval_throughput.png")
    make_plot("mean_travel_time", "Mean Travel Time", "eval_travel_time.png")
    make_plot("travel_time_mse", "Travel Time MSE", "eval_mse.png")

    print("\nSaved evaluation artifacts:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()

"""Run a rule-based actuated traffic-signal baseline and save metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import traci
from experiment_utils import (
    DEFAULT_EXPERIMENT_CONFIG,
    collect_run_metadata,
    episode_seeds,
    load_json,
    resolve_path,
    set_global_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMOCFG = PROJECT_ROOT / "sumo" / "sim.sumocfg"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines" / "actuated"


@dataclass
class EpisodeMetrics:
    episode_idx: int
    seed: int
    steps: int
    avg_queue_length: float
    avg_waiting_time: float
    total_arrivals: int
    throughput_per_step: float


def _phase_incoming_lanes(tl_id: str) -> list[set[str]]:
    logics = traci.trafficlight.getAllProgramLogics(tl_id)
    if not logics:
        raise RuntimeError(f"No program logic found for traffic light '{tl_id}'.")
    phases = logics[0].phases
    links_by_index = traci.trafficlight.getControlledLinks(tl_id)

    phase_lanes: list[set[str]] = []
    for phase in phases:
        lanes: set[str] = set()
        for signal_idx, signal_state in enumerate(phase.state):
            if signal_idx >= len(links_by_index):
                continue
            if signal_state not in ("G", "g"):
                continue
            for link in links_by_index[signal_idx]:
                if link and link[0]:
                    lanes.add(link[0])
        phase_lanes.append(lanes)
    return phase_lanes


def _phase_demand(phase_idx: int, phase_lanes: list[set[str]]) -> float:
    lanes = phase_lanes[phase_idx]
    if not lanes:
        return 0.0
    halted = sum(float(traci.lane.getLastStepHaltingNumber(l)) for l in lanes)
    approaching = sum(float(traci.lane.getLastStepVehicleNumber(l)) for l in lanes)
    return halted + 0.5 * approaching


def run_one_episode(
    *,
    sumo_binary: str,
    sumocfg_path: Path,
    max_steps: int,
    seed: int,
    min_green: int,
    max_green: int,
    demand_gap: float,
    low_demand_threshold: float,
) -> EpisodeMetrics:
    traci.start(
        [
            sumo_binary,
            "-c",
            str(sumocfg_path),
            "--seed",
            str(seed),
        ]
    )
    try:
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in SUMO scenario.")
        tl_id = tls_ids[0]
        lane_ids = sorted(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
        if not lane_ids:
            raise RuntimeError(f"No controlled lanes found for traffic light '{tl_id}'.")

        phase_lanes = _phase_incoming_lanes(tl_id)
        time_in_phase = 0
        queue_trace: list[float] = []
        waiting_trace: list[float] = []
        total_arrivals = 0
        actual_steps = 0

        for step in range(max_steps):
            phase_idx = traci.trafficlight.getPhase(tl_id)
            phase_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            is_yellow = "y" in phase_state or "Y" in phase_state
            should_switch = False

            if not is_yellow:
                cur_demand = _phase_demand(phase_idx, phase_lanes)
                alt_demands = [
                    _phase_demand(i, phase_lanes)
                    for i in range(len(phase_lanes))
                    if i != phase_idx
                ]
                best_alt = max(alt_demands) if alt_demands else 0.0

                if time_in_phase >= max_green:
                    should_switch = True
                elif time_in_phase >= min_green:
                    low_current = cur_demand <= low_demand_threshold
                    strong_competitor = best_alt >= (cur_demand + demand_gap)
                    should_switch = low_current or strong_competitor

            if should_switch:
                traci.trafficlight.setPhase(tl_id, (phase_idx + 1) % len(phase_lanes))
                time_in_phase = 0

            traci.simulationStep()
            time_in_phase += 1
            actual_steps = step + 1

            total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_ids)
            total_wait = sum(traci.lane.getWaitingTime(l) for l in lane_ids)
            queue_trace.append(float(total_queue))
            waiting_trace.append(float(total_wait))
            total_arrivals += int(traci.simulation.getArrivedNumber())

            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        avg_queue = mean(queue_trace) if queue_trace else 0.0
        avg_wait = mean(waiting_trace) if waiting_trace else 0.0
        throughput = float(total_arrivals) / max(actual_steps, 1)
        return EpisodeMetrics(
            episode_idx=-1,
            seed=seed,
            steps=actual_steps,
            avg_queue_length=avg_queue,
            avg_waiting_time=avg_wait,
            total_arrivals=total_arrivals,
            throughput_per_step=throughput,
        )
    finally:
        if traci.isLoaded():
            traci.close()


def save_results(
    *,
    episodes: list[EpisodeMetrics],
    output_prefix: Path,
    args: argparse.Namespace,
    metadata: dict,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    aggregates = {
        "episodes": len(episodes),
        "mean_avg_queue_length": mean([e.avg_queue_length for e in episodes]),
        "mean_avg_waiting_time": mean([e.avg_waiting_time for e in episodes]),
        "mean_total_arrivals": mean([e.total_arrivals for e in episodes]),
        "mean_throughput_per_step": mean([e.throughput_per_step for e in episodes]),
        "total_arrivals": sum(e.total_arrivals for e in episodes),
    }

    payload = {
        "baseline": "actuated",
        "generated_at_utc": metadata["generated_at_utc"],
        "config": {
            "sumocfg_path": str(args.sumocfg),
            "sumo_binary": args.sumo_binary,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "min_green": args.min_green,
            "max_green": args.max_green,
            "demand_gap": args.demand_gap,
            "low_demand_threshold": args.low_demand_threshold,
            "seeds": args.seeds,
        },
        "metadata": metadata,
        "aggregates": aggregates,
        "episodes_data": [asdict(e) for e in episodes],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(episodes[0]).keys()))
        writer.writeheader()
        for ep in episodes:
            writer.writerow(asdict(ep))

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an actuated baseline and save comparable metrics."
    )
    parser.add_argument("--sumocfg", type=Path, default=DEFAULT_SUMOCFG)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--min-green", type=int, default=10)
    parser.add_argument("--max-green", type=int, default=45)
    parser.add_argument("--demand-gap", type=float, default=2.0)
    parser.add_argument("--low-demand-threshold", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=4100)
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output path prefix. Defaults to results/baselines/actuated/actuated_<timestamp>.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Centralized experiment config JSON file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiment_cfg: dict = {}
    if args.experiment_config and args.experiment_config.exists():
        experiment_cfg = load_json(args.experiment_config)
        env_cfg = experiment_cfg.get("environment", {})
        eval_cfg = experiment_cfg.get("evaluation", {})
        repro_cfg = experiment_cfg.get("reproducibility", {})
        act_cfg = experiment_cfg.get("baselines", {}).get("actuated", {})

        if args.sumocfg == DEFAULT_SUMOCFG and env_cfg.get("sumocfg_path"):
            args.sumocfg = resolve_path(env_cfg.get("sumocfg_path"))
        if args.sumo_binary == "sumo":
            args.sumo_binary = env_cfg.get("sumo_binary", args.sumo_binary)
        if args.episodes == 5:
            args.episodes = int(eval_cfg.get("episodes", args.episodes))
        if args.max_steps == 1000:
            args.max_steps = int(eval_cfg.get("horizon", args.max_steps))
        if not args.seeds.strip() and "seeds" in repro_cfg:
            args.seeds = ",".join(str(x) for x in repro_cfg["seeds"])
        if args.base_seed == 4100:
            args.base_seed = int(repro_cfg.get("base_seed", args.base_seed))
        if args.min_green == 10:
            args.min_green = int(act_cfg.get("min_green", args.min_green))
        if args.max_green == 45:
            args.max_green = int(act_cfg.get("max_green", args.max_green))
        if args.demand_gap == 2.0:
            args.demand_gap = float(act_cfg.get("demand_gap", args.demand_gap))
        if args.low_demand_threshold == 1.0:
            args.low_demand_threshold = float(
                act_cfg.get("low_demand_threshold", args.low_demand_threshold)
            )

    args.sumocfg = resolve_path(args.sumocfg)
    if not args.sumocfg.exists():
        raise FileNotFoundError(f"SUMO config not found: {args.sumocfg}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.min_green <= 0 or args.max_green <= 0:
        raise ValueError("--min-green and --max-green must be > 0")
    if args.min_green > args.max_green:
        raise ValueError("--min-green must be <= --max-green")

    seeds = episode_seeds(
        episodes=args.episodes,
        base_seed=args.base_seed,
        explicit_csv=args.seeds,
    )
    args.seeds = seeds

    episodes: list[EpisodeMetrics] = []
    for episode_idx, seed in enumerate(seeds):
        set_global_seed(seed)
        metrics = run_one_episode(
            sumo_binary=args.sumo_binary,
            sumocfg_path=args.sumocfg,
            max_steps=args.max_steps,
            seed=seed,
            min_green=args.min_green,
            max_green=args.max_green,
            demand_gap=args.demand_gap,
            low_demand_threshold=args.low_demand_threshold,
        )
        metrics.episode_idx = episode_idx
        episodes.append(metrics)
        print(
            f"[episode {episode_idx}] seed={seed} steps={metrics.steps} "
            f"avg_wait={metrics.avg_waiting_time:.2f} avg_queue={metrics.avg_queue_length:.2f} "
            f"arrivals={metrics.total_arrivals} throughput={metrics.throughput_per_step:.4f}"
        )

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        prefix = RESULTS_DIR / f"actuated_{stamp}"

    metadata = collect_run_metadata(
        config_path=args.experiment_config if args.experiment_config.exists() else None,
        resolved_config=experiment_cfg,
        script_name="run_actuated_baseline.py",
    )
    json_path, csv_path = save_results(
        episodes=episodes,
        output_prefix=prefix,
        args=args,
        metadata=metadata,
    )
    print("\nSaved actuated baseline metrics:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()

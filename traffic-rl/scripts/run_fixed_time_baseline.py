"""Run a fixed-time traffic-signal baseline and save reproducible metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import traci


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMOCFG = PROJECT_ROOT / "sumo" / "sim.sumocfg"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines" / "fixed_time"


@dataclass
class EpisodeMetrics:
    episode_idx: int
    seed: int
    steps: int
    avg_queue_length: float
    avg_waiting_time: float
    total_arrivals: int
    throughput_per_step: float


def parse_phase_durations(raw: str) -> list[int]:
    durations = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not durations:
        raise ValueError("phase durations must not be empty")
    if any(x <= 0 for x in durations):
        raise ValueError("phase durations must be positive integers")
    return durations


def run_one_episode(
    *,
    sumo_binary: str,
    sumocfg_path: Path,
    max_steps: int,
    seed: int,
    phase_durations: list[int],
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

        phase_idx = 0
        phase_time_remaining = phase_durations[phase_idx]
        traci.trafficlight.setPhase(tl_id, phase_idx)

        queue_trace: list[float] = []
        waiting_trace: list[float] = []
        total_arrivals = 0
        actual_steps = 0

        for step in range(max_steps):
            if phase_time_remaining <= 0:
                phase_idx = (phase_idx + 1) % len(phase_durations)
                traci.trafficlight.setPhase(tl_id, phase_idx)
                phase_time_remaining = phase_durations[phase_idx]

            traci.simulationStep()
            phase_time_remaining -= 1
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
            episode_idx=-1,  # filled by caller
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
        "baseline": "fixed_time",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "config": {
            "sumocfg_path": str(args.sumocfg),
            "sumo_binary": args.sumo_binary,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "phase_durations": args.phase_durations,
            "seeds": args.seeds,
        },
        "aggregates": aggregates,
        "episodes_data": [asdict(e) for e in episodes],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(episodes[0]).keys()),
        )
        writer.writeheader()
        for ep in episodes:
            writer.writerow(asdict(ep))

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-time baseline and save comparable metrics."
    )
    parser.add_argument("--sumocfg", type=Path, default=DEFAULT_SUMOCFG)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--phase-durations",
        default="42,3,42,3",
        help="Comma-separated durations (in simulation steps) for each phase index.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=4100,
        help="First seed to use when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional explicit comma-separated episode seeds.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output path prefix. Defaults to results/baselines/fixed_time/fixed_time_<timestamp>.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.sumocfg.exists():
        raise FileNotFoundError(f"SUMO config not found: {args.sumocfg}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")

    phase_durations = parse_phase_durations(args.phase_durations)

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [args.base_seed + i for i in range(args.episodes)]
    if len(seeds) != args.episodes:
        raise ValueError("Number of seeds must match --episodes")
    args.seeds = seeds

    episodes: list[EpisodeMetrics] = []
    for episode_idx, seed in enumerate(seeds):
        metrics = run_one_episode(
            sumo_binary=args.sumo_binary,
            sumocfg_path=args.sumocfg,
            max_steps=args.max_steps,
            seed=seed,
            phase_durations=phase_durations,
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
        prefix = RESULTS_DIR / f"fixed_time_{stamp}"

    json_path, csv_path = save_results(episodes=episodes, output_prefix=prefix, args=args)
    print(f"\nSaved fixed-time baseline metrics:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()

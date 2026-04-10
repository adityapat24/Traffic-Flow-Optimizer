from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent, PPOConfig
from env.traffic_env import TrafficEnv
from experiment_utils import (
    DEFAULT_EXPERIMENT_CONFIG,
    collect_run_metadata,
    load_json as load_json_file,
    resolve_path,
    set_global_seed,
)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


## evaluate_policy and above are all same as train_dqn.py
def evaluate_policy(env: TrafficEnv, agent: PPOAgent | None, episodes: int, max_steps: int) -> dict:
    rewards = []
    throughputs = []
    waits = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        step_count = 0

        episode_reward = 0.0
        episode_throughput = 0.0
        wait_trace = []

        ## PPO difference: collect full trajectory during episode
        ## rest of loop is same as dqn, just action selection is different
        while not done and step_count < max_steps:
            if agent is None:
                action = env.action_space.sample()
            else:
                action, _, _ = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            episode_reward += float(reward)
            episode_throughput += float(info.get("cars_through", 0.0))
            wait_trace.append(float(info.get("total_wait", 0.0)))

            state = next_state
            step_count += 1

        rewards.append(episode_reward)
        throughputs.append(episode_throughput)
        waits.append(float(np.mean(wait_trace)) if wait_trace else 0.0)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_throughput": float(np.mean(throughputs)),
        "mean_wait": float(np.mean(waits)),
        "rewards": rewards,
        "throughputs": throughputs,
        "waits": waits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_config.json")
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    args = parser.parse_args()

    experiment_cfg: dict = {}
    if args.experiment_config and args.experiment_config.exists():
        experiment_cfg = load_json_file(args.experiment_config)
        args.config = experiment_cfg.get("agents", {}).get("ppo_config_path", args.config)

    config = load_config(resolve_path(args.config))
    if experiment_cfg.get("reproducibility", {}).get("base_seed") is not None:
        config["seed"] = int(experiment_cfg["reproducibility"]["base_seed"])
    set_seed(int(config["seed"]))
    set_global_seed(int(config["seed"]))

    device = torch.device(
        "cuda" if bool(config["use_cuda"]) and torch.cuda.is_available() else "cpu"
    )

    env = TrafficEnv(
        sumocfg_path=config["sumocfg_path"],
        sumo_binary=config["sumo_binary"],
        normalize_observations=bool(config["normalize_observations"]),
    )

    _state, _ = env.reset(seed=int(config["seed"]))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    ## use PPO Agent, rest of code and approach are mostly similar to DQN in train_dqn.py
    agent = PPOAgent(
        PPOConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(config["hidden_dim"]),
            learning_rate=float(config["learning_rate"]),
            gamma=float(config["gamma"]),
            adv_estimate_lambda=float(config["adv_estimate_lambda"]),
            clip_epsilon=float(config["clip_epsilon"]),
            update_epochs=int(config["update_epochs"]),
            device=device,
        )
    )

    num_episodes = int(config["num_episodes"])
    max_steps = int(config["max_steps_per_episode"])
    save_every = int(config["save_every"])

    episode_rewards = []
    episode_avg_waits = []
    episode_throughputs = []
    episode_losses = []

    best_reward = float("-inf")

    ## PPO: each episode is one rollout which triggers one PPO update
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=int(config["seed"]) + episode)
        done = False
        step_count = 0

        total_reward = 0.0
        total_throughput = 0.0
        wait_trace = []

        while not done and step_count < max_steps:
            ## sample from policy for update step later
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = bool(terminated or truncated)
            manual_done = step_count + 1 >= max_steps
            final_done = done or manual_done

            ## on policy, store full rollout data for update
            agent.store_step(
                state,
                action,
                log_prob,
                float(reward),
                final_done,
                value,
            )

            total_reward += float(reward)
            total_throughput += float(info.get("cars_through", 0.0))
            wait_trace.append(float(info.get("total_wait", 0.0)))

            state = next_state
            step_count += 1

            if manual_done:
                break

        ## PPO: update happens after full episode, not every step like DQN
        last_value = 0.0

        ## instead of learn(), runs value policy update using above data to get adv and obj 
        avg_loss = agent.update(last_value)
        avg_wait = float(np.mean(wait_trace)) if wait_trace else 0.0

        ## loss averaged over multiple epochs instead of step by step
        episode_rewards.append(total_reward)
        episode_avg_waits.append(avg_wait)
        episode_throughputs.append(total_throughput)
        episode_losses.append(avg_loss)

        print(
            f"Episode {episode:03d} | "
            f"Reward {total_reward:8.2f} | "
            f"AvgWait {avg_wait:8.2f} | "
            f"Throughput {total_throughput:6.2f} | "
            f"Loss {avg_loss:8.4f}"
        )

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / "ppo_best.pt"))

        if episode % save_every == 0:
            agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / f"ppo_ep_{episode}.pt"))

    env.close()

    ## rest is same approach as dqn file
    agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / "ppo_final.pt"))

    metrics = {
        "episode_rewards": episode_rewards,
        "episode_avg_waits": episode_avg_waits,
        "episode_throughputs": episode_throughputs,
        "episode_losses": episode_losses,
    }
    metadata = collect_run_metadata(
        config_path=args.experiment_config if args.experiment_config.exists() else None,
        resolved_config=experiment_cfg,
        script_name="train_ppo.py",
    )
    metrics["metadata"] = metadata
    with open(PROJECT_ROOT / "results" / "metrics" / "ppo_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "ppo_reward_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(episode_avg_waits)
    plt.xlabel("Episode")
    plt.ylabel("Average Wait")
    plt.title("PPO Avg Wait per Episode")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "ppo_wait_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(episode_throughputs)
    plt.xlabel("Episode")
    plt.ylabel("Throughput")
    plt.title("PPO Throughput per Episode")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "ppo_throughput_curve.png")
    plt.close()

    eval_env = TrafficEnv(
        sumocfg_path=config["sumocfg_path"],
        sumo_binary=config["sumo_binary"],
        normalize_observations=bool(config["normalize_observations"]),
    )

    trained_eval = evaluate_policy(
        eval_env,
        agent=agent,
        episodes=int(config["eval_episodes"]),
        max_steps=max_steps,
    )
    random_eval = evaluate_policy(
        eval_env,
        agent=None,
        episodes=int(config["eval_episodes"]),
        max_steps=max_steps,
    )
    eval_env.close()

    comparison = {
        "trained_policy": trained_eval,
        "random_policy": random_eval,
        "improvement_reward": trained_eval["mean_reward"] - random_eval["mean_reward"],
        "improvement_throughput": trained_eval["mean_throughput"] - random_eval["mean_throughput"],
        "improvement_wait": random_eval["mean_wait"] - trained_eval["mean_wait"],
        "metadata": metadata,
    }

    with open(PROJECT_ROOT / "results" / "metrics" / "ppo_vs_random.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Final Comparison ===")
    print(json.dumps(comparison, indent=2))

    # ===== Save per-episode CSV (Ticket 13) =====
    from results_utils import write_csv

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for i in range(len(episode_rewards)):
        csv_rows.append({
            "episode": i + 1,
            "reward": float(episode_rewards[i]),
            "avg_wait": float(episode_avg_waits[i]),
            "throughput": float(episode_throughputs[i]),
            "loss": float(episode_losses[i]),
        })

    write_csv(
        metrics_dir / "ppo_training_metrics.csv",
        csv_rows,
        ["episode", "reward", "avg_wait", "throughput", "loss"],
    )

if __name__ == "__main__":
    main()

from __future__ import annotations

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

from agents.dqn_agent import DQNAgent, DQNConfig
from env.traffic_env import TrafficEnv


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(env: TrafficEnv, agent: DQNAgent | None, episodes: int, max_steps: int) -> dict:
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

        while not done and step_count < max_steps:
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, epsilon=0.0)

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
        "waits": waits
    }


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "dqn_config.json")
    set_seed(int(config["seed"]))

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

    agent = DQNAgent(
        DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(config["hidden_dim"]),
            learning_rate=float(config["learning_rate"]),
            gamma=float(config["gamma"]),
            buffer_capacity=int(config["buffer_capacity"]),
            batch_size=int(config["batch_size"]),
            target_update_freq=int(config["target_update_freq"]),
            device=device,
        )
    )

    num_episodes = int(config["num_episodes"])
    max_steps = int(config["max_steps_per_episode"])
    warmup_steps = int(config["warmup_steps"])

    epsilon = float(config["epsilon_start"])
    epsilon_decay = float(config["epsilon_decay"])
    epsilon_min = float(config["epsilon_min"])
    save_every = int(config["save_every"])

    episode_rewards = []
    episode_avg_waits = []
    episode_throughputs = []
    episode_losses = []

    global_step = 0
    best_reward = float("-inf")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=int(config["seed"]) + episode)
        done = False
        step_count = 0

        total_reward = 0.0
        total_throughput = 0.0
        wait_trace = []
        loss_trace = []

        while not done and step_count < max_steps:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = bool(terminated or truncated)
            manual_done = step_count + 1 >= max_steps
            final_done = done or manual_done

            agent.store_transition(state, action, float(reward), next_state, final_done)

            total_reward += float(reward)
            total_throughput += float(info.get("cars_through", 0.0))
            wait_trace.append(float(info.get("total_wait", 0.0)))

            if global_step >= warmup_steps and agent.can_learn():
                loss = agent.learn()
                loss_trace.append(loss)

            state = next_state
            step_count += 1
            global_step += 1

            if manual_done:
                break

        avg_wait = float(np.mean(wait_trace)) if wait_trace else 0.0
        avg_loss = float(np.mean(loss_trace)) if loss_trace else 0.0

        episode_rewards.append(total_reward)
        episode_avg_waits.append(avg_wait)
        episode_throughputs.append(total_throughput)
        episode_losses.append(avg_loss)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(
            f"Episode {episode:03d} | "
            f"Reward {total_reward:8.2f} | "
            f"AvgWait {avg_wait:8.2f} | "
            f"Throughput {total_throughput:6.2f} | "
            f"Epsilon {epsilon:.3f}"
        )

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / "dqn_best.pt"))

        if episode % save_every == 0:
            agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / f"dqn_ep_{episode}.pt"))

    env.close()

    agent.save(str(PROJECT_ROOT / "results" / "checkpoints" / "dqn_final.pt"))

    metrics = {
        "episode_rewards": episode_rewards,
        "episode_avg_waits": episode_avg_waits,
        "episode_throughputs": episode_throughputs,
        "episode_losses": episode_losses,
    }
    with open(PROJECT_ROOT / "results" / "metrics" / "dqn_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "dqn_reward_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(episode_avg_waits)
    plt.xlabel("Episode")
    plt.ylabel("Average Wait")
    plt.title("DQN Average Wait per Episode")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "dqn_wait_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(episode_throughputs)
    plt.xlabel("Episode")
    plt.ylabel("Throughput")
    plt.title("DQN Throughput per Episode")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "plots" / "dqn_throughput_curve.png")
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
        "improvement_wait": random_eval["mean_wait"] - trained_eval["mean_wait"]
    }

    with open(PROJECT_ROOT / "results" / "metrics" / "dqn_vs_random.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Final Comparison ===")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

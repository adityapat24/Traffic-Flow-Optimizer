from __future__ import annotations

from dataclasses import dataclass

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

## approach: Keep similar to dqn_agent.py but with on-policy logic and PPO updates.
# PPO will use ActorCritic instead of DQN's Q-network (PPO on policy vs DQN off policy).
@dataclass
class PPOConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    adv_estimate_lambda: float
    clip_epsilon: float ## limits how policy can change during update
    update_epochs: int ## number of passes per update step
    device: torch.device


## Actor - decides what action to take
## Critic - estimates value of current state
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        ## separate heads for policy choice and value
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.model = ActorCritic(
            cfg.state_dim,
            cfg.action_dim,
            cfg.hidden_dim,
        ).to(self.device)

        ## same as dqn using Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # instead of replay buffer, stores full rollout data to get advantages and update policy
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        ## convert state to tensor
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        ## forward pass, no grad since not updating just selecting action
        with torch.no_grad():
            logits, value = self.model(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            ## stochastic policy sampling instead of DQN's epsilon-greedy
            action = dist.sample()
            log_prob = dist.log_prob(action)

        ## return action and whats needed for PPO update (log_prob and value)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    ## instead of dqn storing transitions, ppo stores full rollout data to get advantages and update policy
    def store_step(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear_buffer(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    ## converts rewards and values into returns and advantages for PPO update step
    ## advantages are how much better outcome was vs critics prediction
    ## returns are what the critic should have predicited
    def compute_returns_and_advantages(self, last_value: float):
        returns = []
        advantages = []

        adv_estimate = 0.0
        next_value = last_value

        ## go backwards through rewards and values to get advantage estimates
        for step in reversed(range(len(self.rewards))):
            ## delta is reward + (gamma * next value) - current value
            ## aka how wrong was critics estimate
            delta = (
                self.rewards[step]
                + self.cfg.gamma * next_value * (1 - self.dones[step])
                - self.values[step]
            )

            ## get future advantage * gamma and store
            adv_estimate = delta + self.cfg.gamma * self.cfg.adv_estimate_lambda * (1 - self.dones[step]) * adv_estimate
            advantages.insert(0, adv_estimate)

            next_value = self.values[step]

        ## target values for critic
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

    ## Takes stored rollout data and updates with gradient
    def update(self, last_value: float) -> float:
        returns, advantages = self.compute_returns_and_advantages(last_value)

        ## convert to tensores
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        ## old policy before update
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        ## normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0

        ## loop over epochs and update policy with loss
        for _ in range(self.cfg.update_epochs):
            ## forward pass
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)

            ## encourage exploration
            entropy = dist.entropy().mean()

            ## compare old and new policy
            ratio = torch.exp(new_log_probs - old_log_probs)

            ## use clamped and not clamped objective to get final loss
            ## clamp to prevent too large policy updates
            objective = ratio * advantages
            clamped_objective = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages

            policy_loss = -torch.min(objective, clamped_objective).mean()

            ## train critic to predict returns using mse loss
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)

            ## should potentially move these into cfg, for now hard coding
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            ## clear old gradients, compute new ones, apply using step
            ## same approach as dqn
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        ## on policy, so clear buffer to not reuse old data
        self.clear_buffer()

        ## return average loss per epoch
        return total_loss / self.cfg.update_epochs

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
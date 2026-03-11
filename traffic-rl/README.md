# AI-Driven Traffic Flow Optimization (`traffic-rl`)

This project implements an AI-driven system to optimize urban traffic flow via intelligent traffic signal control in a simulated environment.

The core idea is to use **reinforcement learning (RL)** to dynamically control traffic lights based on real-time traffic conditions, reducing congestion, vehicle idle times, and overall travel time.

## Project Structure

The repository is organized as:

- `env/` – Gymnasium-compatible environment wrapper around the SUMO simulation (to be implemented).
- `agents/` – RL agent implementations (e.g., DQN, PPO) (to be implemented).
- `baselines/` – Static and actuated baseline controllers for comparison (to be implemented).
- `sumo/` – SUMO configuration files, network definitions, and route files.
- `scripts/` – Training, evaluation, and analysis scripts.
- `results/` – Logs, metrics, and plots from experiments.
- `requirements.txt` – Python dependencies for the project.

## Key Components (Planned)

- **Simulation**: SUMO (Simulation of Urban Mobility) as the traffic simulator.
- **Control Interface**: TraCI to read simulation state (e.g., queue lengths, waiting times) and manipulate traffic lights.
- **RL Environment**: OpenAI Gym/Gymnasium wrapper exposing states, actions, and rewards.
- **RL Agents**: Centralized single-agent controllers using algorithms like Deep Q-Learning (DQN) or Proximal Policy Optimization (PPO).
- **Baselines**: Fixed-time (static) and actuated (sensor-based) controllers for benchmarking.

## Installation

1. **Create and activate a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

From the `traffic-rl` directory:

```bash
pip install -r requirements.txt
```

Depending on your platform, you may also need to install SUMO separately. Refer to the official SUMO installation guide.


## Next Steps

- Implement the SUMO network and routes in `sumo/`.
- Build the Gymnasium environment in `env/`.
- Implement baseline controllers in `baselines/`.
- Implement and train RL agents in `agents/` and `scripts/`.

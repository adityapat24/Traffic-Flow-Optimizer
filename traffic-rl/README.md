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

### 1. Install SUMO

Download and install the official macOS installer:

https://sumo.dlr.de/docs/Installing/index.html

Verify installation:

```bash
sumo --version
```

### 2. Set Sumo Environment Variables

Add the following to your shell configuration (~/.zshrc):

```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo"
export PATH="/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin:$PATH"
```

Reload your shell and verify

```bash
source ~/.zshrc
echo $SUMO_HOME
```

### 3. Verify SUMO + TraCI Integration

python scripts/test_traci.py

Expected:

```bash
Step 10 complete.
SUMO connection closed.
```

## Running the single intersection simulation

From the `traffic-rl` directory, run:

```bash
sumo-gui -c sumo/sim.sumocfg
```

If you want, a gui-less verion

```bash
sumo -c sumo/sim.sumocfg
```

## Next Steps

- Implement the SUMO network and routes in `sumo/`.
- Build the Gymnasium environment in `env/`.
- Implement baseline controllers in `baselines/`.
- Implement and train RL agents in `agents/` and `scripts/`.

## Fixed-time baseline (Ticket 6)

Run a reproducible static-timer benchmark:

```bash
python scripts/run_fixed_time_baseline.py --episodes 5 --max-steps 1000
```

Outputs are written to:

- `results/baselines/fixed_time/fixed_time_<timestamp>.json`
- `results/baselines/fixed_time/fixed_time_<timestamp>.csv`

To match exact route randomness/seeds with RL experiments, pass explicit seeds:

```bash
python scripts/run_fixed_time_baseline.py --episodes 3 --seeds 4100,4101,4102
```

## Actuated baseline (Ticket 7)

Run a rule-based sensor-like controller benchmark:

```bash
python scripts/run_actuated_baseline.py --episodes 5 --max-steps 1000
```

Outputs are written to:

- `results/baselines/actuated/actuated_<timestamp>.json`
- `results/baselines/actuated/actuated_<timestamp>.csv`

To run fixed-time and actuated with the same seeds for side-by-side comparison:

```bash
python scripts/run_fixed_time_baseline.py --episodes 3 --seeds 4100,4101,4102
python scripts/run_actuated_baseline.py --episodes 3 --seeds 4100,4101,4102
```

## Unified evaluation and travel-time MSE

Run the standardized evaluation pipeline with:

```bash
python scripts/run_standard_eval.py --controllers fixed_time,actuated,ppo --sumocfgs sumo/sim.sumocfg --seeds 4100,4101,4102,4103,4104 --horizon 1000 --ppo-checkpoint results/checkpoints/ppo_best.pt
```

The evaluation report now includes:

- mean waiting time
- mean queue length
- throughput
- mean travel time
- travel-time MSE

### Travel-time MSE definition

The travel-time MSE metric is computed reproducibly for every evaluation episode:

- **Predicted travel time**: for each departed vehicle, sum the free-flow travel times of every route edge plus the internal SUMO junction connector between each consecutive edge pair
- **Segment free-flow time**: `segment_length / min(segment_speed_limit, vehicle_max_speed_at_departure)`
- **Actual travel time**: `arrival_sim_time - departure_sim_time`
- **Per-vehicle squared error**: `(actual_travel_time - predicted_travel_time)^2`
- **Episode MSE**: mean squared error over all vehicles that both departed and arrived within the episode horizon

Each evaluation JSON report stores both the MSE values and the methodology text used to compute them, so the metric can be reproduced from the saved configuration.

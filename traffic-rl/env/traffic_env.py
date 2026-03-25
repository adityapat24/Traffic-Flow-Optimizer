from __future__ import annotations
"""Gymnasium wrapper around a SUMO simulation controlled via TraCI."""
"Test comment"

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci


def setup_sumo_tools() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def _default_sumocfg() -> Path:
    return Path(__file__).resolve().parent.parent / "sumo" / "sim.sumocfg"


@dataclass(frozen=True)
class TrafficEnvContract:
    """
    Environment API contract for consumers (agents, trainers, evaluators).

    Observation schema:
      - Index 0: current traffic light phase index.
      - For each controlled lane, three values in this exact order:
          1) queue length (halting vehicle count)
          2) cumulative waiting time (seconds)
          3) traffic density proxy (lane occupancy in [0, 1])
    Action schema:
      - 0: keep current traffic-light phase.
      - 1: switch to next traffic-light phase.
    """

    action_meanings: tuple[str, str] = ("keep_current_phase", "switch_phase")

    @staticmethod
    def observation_feature_names(lane_ids: list[str]) -> list[str]:
        features = ["phase_index"]
        for lane_id in lane_ids:
            features.append(f"{lane_id}_queue_length")
            features.append(f"{lane_id}_waiting_time")
            features.append(f"{lane_id}_traffic_density")
        return features


class TrafficEnv(gym.Env):
    """
    Single-intersection traffic light control.

    Observation: traffic light phase index, then for each controlled approach lane
    (unique lanes): queue length, total waiting time, and traffic density proxy.

    Actions: 0 = keep current phase, 1 = advance to the next phase in the program.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sumocfg_path: str | Path | None = None,
        sumo_binary: str = "sumo",
        normalize_observations: bool = False,
        waiting_time_norm_seconds: float = 300.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        setup_sumo_tools()
        self.sumocfg_path = Path(sumocfg_path) if sumocfg_path else _default_sumocfg()
        self.sumo_binary = sumo_binary
        self.normalize_observations = normalize_observations
        self.waiting_time_norm_seconds = waiting_time_norm_seconds
        if render_mode is not None:
            self.metadata = {**self.metadata, "render_modes": [render_mode]}

        self._tl_id: str | None = None
        self._lane_ids: list[str] = []
        self._feature_names: list[str] = []
        self._lane_lengths: dict[str, float] = {}
        self.contract = TrafficEnvContract()

        # Placeholder bounds; actual vectors are built in reset() from live TraCI state.
        self._obs_dim = 13  # 1 phase + 4 unique lanes × 3 features for `intersection.net.xml`
        low = np.full(self._obs_dim, -np.inf, dtype=np.float32)
        high = np.full(self._obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        #### Traffic Signal Safety Constraints ##### 
        self.min_green_duration = 10.0  # seconds min before allowing switch 
        self.yellow_duration = 3.0  # seconds for yellow phase 
        self.all_red_duration = 0.0 # no all red in this network 

        # phase config from `intersection.net.xml`:
        self.green_phases = [0, 2]  # indices of green phases in the program
        self.yellow_phases = [1, 3]  # indices of yellow phases in the program
        self.num_phases = 4 # total number of phases in the program 

        # track phases 
        self.current_phase = 0
        self.phase_start = None 
        self.in_yellow = False
        self.yellow_start_time = None 

        # reward penalties for constraint violations 
        self.invalid_switch_penalty = -10.0
        self.valid_switch_reward = 1.0

    def _num_phases(self) -> int:
        import traci

        assert self._tl_id is not None
        logics = traci.trafficlight.getAllProgramLogics(self._tl_id)
        return len(logics[0].phases)

    def observation_feature_names(self) -> list[str]:
        """Return ordered observation feature names for the current network."""
        return list(self._feature_names)

    def action_meanings(self) -> tuple[str, str]:
        """Return semantic meaning for actions in action_space."""
        return self.contract.action_meanings

    def _get_raw_obs(self) -> np.ndarray:
        import traci

        assert self._tl_id is not None
        phase = float(traci.trafficlight.getPhase(self._tl_id))
        parts: list[float] = [phase]
        for lane_id in self._lane_ids:
            # Queue length: only vehicles effectively halted on this lane.
            parts.append(float(traci.lane.getLastStepHaltingNumber(lane_id)))
            parts.append(float(traci.lane.getWaitingTime(lane_id)))
            # Occupancy is provided as percentage in [0, 100]; normalize to [0, 1].
            occupancy = float(traci.lane.getLastStepOccupancy(lane_id)) / 100.0
            parts.append(occupancy)
        obs = np.asarray(parts, dtype=np.float32)
        if obs.shape[0] != self._obs_dim:
            raise RuntimeError(
                f"Observation length {obs.shape[0]} does not match observation_space "
                f"shape ({self._obs_dim}). Update _obs_dim for this network."
            )
        return obs

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize observation components to reduce scale mismatch."""
        obs = raw_obs.astype(np.float32, copy=True)
        phase_count = max(self._num_phases(), 1)
        obs[0] = obs[0] / max(phase_count - 1, 1)

        for i, lane_id in enumerate(self._lane_ids):
            base = 1 + 3 * i
            lane_length = max(float(self._lane_lengths.get(lane_id, 7.5)), 7.5)
            lane_capacity = max(lane_length / 7.5, 1.0)
            obs[base] = obs[base] / lane_capacity
            obs[base + 1] = obs[base + 1] / max(self.waiting_time_norm_seconds, 1.0)
            obs[base + 2] = np.clip(obs[base + 2], 0.0, 1.0)
        return obs
    
    def get_yellow_phase_for_green(self, green_phase: int) -> int:
        """ Map green phase idx to its corresponding yellow phase"""
        # phase 0 (green) -> phase 1 (yellow)
        # phase 2 (green) -> phase 3 (yellow)
        if green_phase == 0:
            return 1
        elif green_phase == 2:
            return 3
        else:
            raise ValueError(f"Invalid green phase index: {green_phase}, must be 0 or 2 for this network.")

    def _handle_transitions(self, current_time: float) -> None:
        """internal: auto advance phase transitions 
        Yellow -> next green when yellow elapsed """
        if self.in_yellow and current_time - self.yellow_start_time >= self.yellow_duration:
            self._advance_to_next_green(current_time)

    def _advance_to_next_green(self, current_time: float) -> None:
        """internal: advance from yellow to next green phase"""

        assert self._tl_id is not None
        self.current_phase = (self.current_phase + 2) % self.num_phases
        assert self.current_phase in self.green_phases, f"Expected to switch to green phase, got {self.current_phase}"
        traci.trafficlight.setPhase(self._tl_id, self.current_phase)
        self.phase_start_time = current_time
        self.in_yellow = False 


    def _get_obs(self) -> np.ndarray:
        raw = self._get_raw_obs()
        if self.normalize_observations:
            return self._normalize_obs(raw)
        return raw

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        import traci

        if not self.sumocfg_path.exists():
            raise FileNotFoundError(f"SUMO config not found: {self.sumocfg_path}")

        # Always close prior session before new episode to avoid stale sessions.
        self.close()
        cfg = str(self.sumocfg_path.resolve())
        try:
            traci.start([self.sumo_binary, "-c", cfg])
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start SUMO binary '{self.sumo_binary}'. Ensure SUMO is "
                "installed and the binary is on PATH, or pass sumo_binary explicitly."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start SUMO with config '{cfg}'. Original error: {exc}"
            ) from exc

        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            self.close()
            raise RuntimeError("No traffic lights found in the simulation.")
        self._tl_id = tls_ids[0]
        self._lane_ids = sorted(
            dict.fromkeys(traci.trafficlight.getControlledLanes(self._tl_id))
        )
        if not self._lane_ids:
            self.close()
            raise RuntimeError(f"No controlled lanes for traffic light '{self._tl_id}'.")
        self._feature_names = self.contract.observation_feature_names(self._lane_ids)
        self._lane_lengths = {
            lane_id: float(traci.lane.getLength(lane_id)) for lane_id in self._lane_ids
        }

        self._obs_dim = 1 + 3 * len(self._lane_ids)
        low = np.full(self._obs_dim, -np.inf, dtype=np.float32)
        high = np.full(self._obs_dim, np.inf, dtype=np.float32)
        if self.normalize_observations:
            low = np.zeros(self._obs_dim, dtype=np.float32)
            high = np.full(self._obs_dim, np.inf, dtype=np.float32)
            high[0] = 1.0
            for i in range(len(self._lane_ids)):
                density_idx = 1 + 3 * i + 2
                high[density_idx] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_phase = 0
        self.phase_start_time = traci.simulation.getTime()
        self.in_yellow = False
        traci.trafficlight.setPhase(self._tl_id, self.current_phase)
        
        traci.simulationStep()
        return self._get_obs(), {}

    def step(
        self, action: SupportsFloat | np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        import traci

        if not traci.isLoaded():
            raise RuntimeError("No active SUMO connection; call reset() first.")

        current_time = traci.simulation.getTime()
        
        # ===== Handle ongoing transitions (auto-advance yellow→green) =====
        self._handle_transitions(current_time)
        
        # ===== Process action with safety constraints =====
        a = int(action) if not isinstance(action, np.ndarray) else int(action.item())
        
        reward = 0.0
        
        if a == 1:  # Switch action
            assert self._tl_id is not None
            time_in_green = current_time - self.phase_start_time
            
            if self.in_yellow:
                # Cannot switch during yellow transition
                reward = self.invalid_switch_penalty
            elif time_in_green < self.min_green_duration:
                # Minimum green time not met
                reward = self.invalid_switch_penalty
            else:
                # ===== VALID SWITCH: Initiate yellow transition =====
                yellow_phase = self.get_yellow_phase_for_green(self.current_phase)
                traci.trafficlight.setPhase(self._tl_id, yellow_phase)
                traci.trafficlight.setPhaseDuration(self._tl_id, self.yellow_duration)
                self.in_yellow = True
                self.yellow_start_time = current_time
                reward = self.valid_switch_reward
        
        # ===== Run simulation and collect observation =====
        traci.simulationStep()
        
        obs = self._get_obs()
        terminated = bool(traci.simulation.getMinExpectedNumber() <= 0)
        truncated = False
        info: dict[str, Any] = {
            "current_phase": self.current_phase,
            "in_yellow": self.in_yellow,
            "time_in_phase": current_time - self.phase_start_time,
        }
        
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the active TraCI connection if one exists."""
        try:
            import traci
        except Exception:
            self._tl_id = None
            self._lane_ids = []
            self._feature_names = []
            self._lane_lengths = {}
            return

        try:
            if traci.isLoaded():
                traci.close()
        finally:
            self._tl_id = None
            self._lane_ids = []
            self._feature_names = []
            self._lane_lengths = {}

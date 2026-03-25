"""Gymnasium wrapper around a SUMO simulation controlled via TraCI."""
"Test comment"
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

        traci.simulationStep()
        return self._get_obs(), {}

    def step(
        self, action: SupportsFloat | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        import traci

        if not traci.isLoaded():
            raise RuntimeError("No active SUMO connection; call reset() first.")

        a = int(action) if not isinstance(action, np.ndarray) else int(action.item())

        if a == 1:
            assert self._tl_id is not None
            n = self._num_phases()
            cur = traci.trafficlight.getPhase(self._tl_id)
            traci.trafficlight.setPhase(self._tl_id, (cur + 1) % n)

        traci.simulationStep()

        obs = self._get_obs()
        reward = 0.0
        terminated = bool(traci.simulation.getMinExpectedNumber() <= 0)
        truncated = False
        info: dict[str, Any] = {}
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

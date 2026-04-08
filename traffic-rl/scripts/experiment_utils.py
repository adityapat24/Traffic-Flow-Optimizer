from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment_config.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def episode_seeds(*, episodes: int, base_seed: int, explicit_csv: str) -> list[int]:
    if explicit_csv.strip():
        seeds = [int(x.strip()) for x in explicit_csv.split(",") if x.strip()]
    else:
        seeds = [base_seed + i for i in range(episodes)]
    if len(seeds) != episodes:
        raise ValueError("Number of seeds must match --episodes")
    return seeds


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_run_metadata(
    *,
    config_path: Path | None,
    resolved_config: dict[str, Any],
    script_name: str,
) -> dict[str, Any]:
    return {
        "script": script_name,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git_output(["git", "status", "--porcelain"])),
        "experiment_config_path": str(config_path) if config_path else None,
        "experiment_config_snapshot": resolved_config,
    }

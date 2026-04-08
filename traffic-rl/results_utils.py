from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_line_plot(
    values: Iterable[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    ensure_parent(path)
    plt.figure(figsize=(8, 5))
    plt.plot(list(values))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_bar_plot(
    labels: list[str],
    values: list[float],
    *,
    yerr: list[float] | None,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    ensure_parent(path)
    plt.figure(figsize=(8, 5))
    x = range(len(labels))
    plt.bar(x, values, yerr=yerr, capsize=5 if yerr else 0)
    plt.xticks(list(x), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

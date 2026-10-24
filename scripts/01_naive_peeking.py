"""Experiment 01: false-positive rate of 'stop as soon as p < 0.05' versus number of looks."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import false_positive_rate

LOOKS = [1, 2, 5, 10, 20, 50, 100]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = [(k, false_positive_rate(n_looks=k)) for k in LOOKS]

    with open(RESULTS / "01_naive_peeking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_looks", "false_positive_rate"])
        writer.writerows((k, f"{rate:.4f}") for k, rate in rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([k for k, _ in rows], [rate for _, rate in rows], marker="o")
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("number of looks (log scale)")
    ax.set_ylabel("false-positive rate")
    ax.set_title("A/A tests: stop at the first p < 0.05")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "01_naive_peeking.png", dpi=150)

    for k, rate in rows:
        print(f"{k:>4} looks: {rate:.1%}")


if __name__ == "__main__":
    main()

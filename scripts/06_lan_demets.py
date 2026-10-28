"""Experiment 06: Lan–DeMets OBF alpha spending on regular and irregular peek schedules."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peeking import peeking_with_effect
from peeking.effect import critical_z
from peeking.lan_demets import (
    equal_batch_sizes,
    false_positive_rate_lan_demets,
    info_fractions_from_batches,
    lan_demets_obf_z_criticals,
    lan_demets_z_criticals_for_batches,
    misindexed_obf_z_criticals,
    peeking_with_effect_lan_demets,
)
from peeking.obrien_fleming import false_positive_rate_obrien_fleming, peeking_with_effect_obrien_fleming
from peeking.simulate import ever_significant, simulate_aa_z_batches

TOTAL_USERS = 20_000
N_LOOKS = 10
BASE_RATE = 0.10
TRUE_LIFTS = [0.005, 0.010]
N_SIMS = 20_000
RESULTS = Path(__file__).resolve().parents[1] / "results"

EQUAL_BATCHES = equal_batch_sizes(N_LOOKS, TOTAL_USERS)
# Front-loaded small peeks, then larger batches (same 20k total).
IRREGULAR_BATCHES = np.array(
    [500, 500, 1000, 1500, 2000, 2500, 3000, 4000, 3500, 1500], dtype=int
)
SCHEDULES = {
    "equal": EQUAL_BATCHES,
    "irregular": IRREGULAR_BATCHES,
}


def _naive_fpr_batches(batch_sizes: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z = simulate_aa_z_batches(N_SIMS, batch_sizes, BASE_RATE, rng)
    return float(ever_significant(z, alpha=0.05).mean())


def _misindexed_fpr(batch_sizes: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z = simulate_aa_z_batches(N_SIMS, batch_sizes, BASE_RATE, rng)
    wrong = misindexed_obf_z_criticals(
        len(batch_sizes), total_users_per_arm=TOTAL_USERS, base_rate=BASE_RATE, alpha=0.05
    )
    return float(ever_significant(z, z_critical=wrong).mean())


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fpr_rows = []
    boundary_rows = []
    power_rows = []

    for name, batches in SCHEDULES.items():
        frac = info_fractions_from_batches(batches, total_users=TOTAL_USERS)
        z_ld = lan_demets_z_criticals_for_batches(batches, TOTAL_USERS)
        naive_fpr = _naive_fpr_batches(batches)
        ld_fpr = false_positive_rate_lan_demets(batches, total_users_per_arm=TOTAL_USERS, n_sims=N_SIMS)
        mis_fpr = _misindexed_fpr(batches)
        obf_fpr = (
            false_positive_rate_obrien_fleming(n_looks=N_LOOKS, n_sims=N_SIMS)
            if name == "equal"
            else float("nan")
        )
        fpr_rows.append((name, naive_fpr, ld_fpr, mis_fpr, obf_fpr))
        print(
            f"{name:>10}: naive FPR {naive_fpr:.1%}, Lan–DeMets {ld_fpr:.1%}, "
            f"mis-indexed OBF {mis_fpr:.1%}"
            + (f", equal OBF {obf_fpr:.1%}" if name == "equal" else "")
        )
        for k, (t, z) in enumerate(zip(frac, z_ld), start=1):
            boundary_rows.append((name, k, t, z))

    for true_lift in TRUE_LIFTS:
        for name, batches in SCHEDULES.items():
            naive = peeking_with_effect(
                n_sims=N_SIMS,
                n_looks=N_LOOKS,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
                batch_sizes=batches,
            )
            ld = peeking_with_effect_lan_demets(
                n_sims=N_SIMS,
                batch_sizes=batches,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
            )
            obf_p_correct = float("nan")
            if name == "equal":
                obf = peeking_with_effect_obrien_fleming(
                    n_sims=N_SIMS,
                    n_looks=N_LOOKS,
                    total_users_per_arm=TOTAL_USERS,
                    rate_control=BASE_RATE,
                    rate_treatment=BASE_RATE + true_lift,
                    seed=0,
                )
                obf_p_correct = obf.p_correct
            power_rows.append(
                (
                    name,
                    true_lift,
                    naive.p_correct,
                    ld.p_correct,
                    obf_p_correct,
                    naive.p_inconclusive,
                    ld.p_inconclusive,
                )
            )
            print(
                f"  {name}, lift {true_lift:.3f}: naive {naive.p_correct:.1%}, "
                f"Lan–DeMets {ld.p_correct:.1%}"
                + (f", OBF {obf_p_correct:.1%}" if name == "equal" else "")
            )

    with open(RESULTS / "06_lan_demets_fpr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "naive_fpr", "lan_demets_fpr", "misindexed_obf_fpr", "obf_fpr_equal_only"])
        for name, naive_fpr, ld_fpr, mis_fpr, obf_fpr in fpr_rows:
            writer.writerow(
                [
                    name,
                    f"{naive_fpr:.4f}",
                    f"{ld_fpr:.4f}",
                    f"{mis_fpr:.4f}",
                    "" if np.isnan(obf_fpr) else f"{obf_fpr:.4f}",
                ]
            )

    with open(RESULTS / "06_lan_demets_boundaries.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "look", "info_fraction", "lan_demets_z"])
        for row in boundary_rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}"])

    with open(RESULTS / "06_lan_demets_power.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "schedule",
                "true_lift",
                "naive_p_correct",
                "lan_demets_p_correct",
                "obf_p_correct_equal_only",
                "naive_p_inconclusive",
                "lan_demets_p_inconclusive",
                "n_sims",
            ]
        )
        for row in power_rows:
            writer.writerow(
                [
                    row[0],
                    f"{row[1]:.3f}",
                    f"{row[2]:.4f}",
                    f"{row[3]:.4f}",
                    "" if np.isnan(row[4]) else f"{row[4]:.4f}",
                    f"{row[5]:.4f}",
                    f"{row[6]:.4f}",
                    N_SIMS,
                ]
            )

    _plot(fpr_rows, power_rows, boundary_rows)
    print(f"wrote {RESULTS / '06_lan_demets_fpr.csv'}")


def _plot(fpr_rows: list, power_rows: list, boundary_rows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    labels = [r[0] for r in fpr_rows]
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - width, [r[1] for r in fpr_rows], width, label="naive p < 0.05")
    ax.bar(x, [r[2] for r in fpr_rows], width, label="Lan–DeMets OBF")
    ax.bar(x + width, [r[3] for r in fpr_rows], width, label="mis-indexed OBF")
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title("Irregular peeks: spending vs fixed look index")
    ax.legend(fontsize=7)

    ax = axes[1]
    for schedule, linestyle in (("equal", "-"), ("irregular", "--")):
        subset = [r for r in boundary_rows if r[0] == schedule]
        ax.plot(
            [r[2] for r in subset],
            [r[3] for r in subset],
            marker="o",
            linestyle=linestyle,
            label=f"{schedule} schedule",
        )
    ax.axhline(critical_z(0.05), color="gray", linestyle=":", label="fixed-sample 1.96")
    ax.set_xlabel("information fraction")
    ax.set_ylabel("Lan–DeMets z threshold")
    ax.set_title("Boundaries follow information, not look count")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS / "06_lan_demets.png", dpi=150)


if __name__ == "__main__":
    main()

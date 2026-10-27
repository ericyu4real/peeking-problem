"""Experiment 05: O'Brien–Fleming boundary — strict early, lenient late;
compare A/A error rate and power with Pocock.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import false_positive_rate, peeking_with_effect
from peeking.obrien_fleming import (
    false_positive_rate_obrien_fleming,
    obrien_fleming_z_criticals,
    peeking_with_effect_obrien_fleming,
)
from peeking.pocock import false_positive_rate_pocock, peeking_with_effect_pocock, pocock_z_critical

LOOKS = [1, 2, 5, 10, 20, 50, 100]
TRUE_LIFTS = [0.005, 0.010]
BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fpr_rows = []
    boundary_rows = []
    power_rows = []

    for n_looks in LOOKS:
        z_obf = obrien_fleming_z_criticals(n_looks, total_users_per_arm=TOTAL_USERS, base_rate=BASE_RATE)
        z_pocock = pocock_z_critical(n_looks, total_users_per_arm=TOTAL_USERS, base_rate=BASE_RATE)
        naive_fpr = false_positive_rate(n_looks=n_looks, n_sims=N_SIMS)
        obf_fpr = false_positive_rate_obrien_fleming(n_looks=n_looks, n_sims=N_SIMS)
        pocock_fpr = false_positive_rate_pocock(n_looks=n_looks, n_sims=N_SIMS)
        fpr_rows.append((n_looks, naive_fpr, obf_fpr, pocock_fpr))
        boundary_rows.append((n_looks, z_obf[0], z_obf[-1], z_pocock))
        print(
            f"{n_looks:>4} looks: OBF first={z_obf[0]:.3f} final={z_obf[-1]:.3f}, "
            f"Pocock z*={z_pocock:.3f}, naive FPR {naive_fpr:.1%}, "
            f"OBF FPR {obf_fpr:.1%}, Pocock FPR {pocock_fpr:.1%}"
        )

    for true_lift in TRUE_LIFTS:
        for n_looks in LOOKS:
            naive = peeking_with_effect(
                n_sims=N_SIMS,
                n_looks=n_looks,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
            )
            obf = peeking_with_effect_obrien_fleming(
                n_sims=N_SIMS,
                n_looks=n_looks,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
            )
            pocock = peeking_with_effect_pocock(
                n_sims=N_SIMS,
                n_looks=n_looks,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
            )
            power_rows.append(
                (
                    n_looks,
                    true_lift,
                    naive.p_correct,
                    obf.p_correct,
                    pocock.p_correct,
                    naive.p_inconclusive,
                    obf.p_inconclusive,
                    pocock.p_inconclusive,
                )
            )
            print(
                f"  lift {true_lift:.3f}, {n_looks:>3} looks: "
                f"naive {naive.p_correct:.1%}, OBF {obf.p_correct:.1%}, "
                f"Pocock {pocock.p_correct:.1%}"
            )

    with open(RESULTS / "05_obrien_fleming_fpr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_looks", "naive_fpr", "obf_fpr", "pocock_fpr"])
        for n_looks, naive_fpr, obf_fpr, pocock_fpr in fpr_rows:
            writer.writerow([n_looks, f"{naive_fpr:.4f}", f"{obf_fpr:.4f}", f"{pocock_fpr:.4f}"])

    with open(RESULTS / "05_obrien_fleming_boundaries.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_looks", "obf_z_first", "obf_z_final", "pocock_z_critical"])
        for n_looks, z_first, z_final, z_pocock in boundary_rows:
            writer.writerow([n_looks, f"{z_first:.4f}", f"{z_final:.4f}", f"{z_pocock:.4f}"])

    with open(RESULTS / "05_obrien_fleming_power.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_looks",
                "true_lift",
                "naive_p_correct",
                "obf_p_correct",
                "pocock_p_correct",
                "naive_p_inconclusive",
                "obf_p_inconclusive",
                "pocock_p_inconclusive",
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
                    f"{row[4]:.4f}",
                    f"{row[5]:.4f}",
                    f"{row[6]:.4f}",
                    f"{row[7]:.4f}",
                    N_SIMS,
                ]
            )

    _plot(fpr_rows, power_rows)
    print(f"wrote {RESULTS / '05_obrien_fleming_fpr.csv'}")


def _plot(fpr_rows: list, power_rows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    looks = [r[0] for r in fpr_rows]
    ax.plot(looks, [r[1] for r in fpr_rows], marker="o", label="naive p < 0.05")
    ax.plot(looks, [r[2] for r in fpr_rows], marker="^", label="O'Brien–Fleming")
    ax.plot(looks, [r[3] for r in fpr_rows], marker="s", label="Pocock")
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title("O'Brien–Fleming holds the error rate")
    ax.legend(fontsize=7)

    ax = axes[1]
    for true_lift, marker in ((0.005, "o"), (0.010, "s")):
        subset = [r for r in power_rows if abs(r[1] - true_lift) < 1e-9]
        looks = [r[0] for r in subset]
        ax.plot(
            looks,
            [r[2] for r in subset],
            marker=marker,
            linestyle="-",
            label=f"naive, lift {true_lift:.3f}",
        )
        ax.plot(
            looks,
            [r[3] for r in subset],
            marker=marker,
            linestyle="--",
            label=f"OBF, lift {true_lift:.3f}",
        )
        ax.plot(
            looks,
            [r[4] for r in subset],
            marker=marker,
            linestyle=":",
            label=f"Pocock, lift {true_lift:.3f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("correct-call rate")
    ax.set_title("Power: O'Brien–Fleming vs Pocock")
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(RESULTS / "05_obrien_fleming.png", dpi=150)


if __name__ == "__main__":
    main()

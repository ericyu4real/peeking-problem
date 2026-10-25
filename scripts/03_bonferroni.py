"""Experiment 03: Bonferroni across looks — does it hold the false-positive rate,
and how much power does it cost versus naive peeking?
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import false_positive_rate, peeking_with_effect
from peeking.bonferroni import false_positive_rate_bonferroni, peeking_with_effect_bonferroni

LOOKS = [1, 2, 5, 10, 20, 50, 100]
TRUE_LIFTS = [0.005, 0.010]
BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fpr_rows = []
    power_rows = []

    for n_looks in LOOKS:
        naive_fpr = false_positive_rate(n_looks=n_looks, n_sims=N_SIMS)
        bonf_fpr = false_positive_rate_bonferroni(n_looks=n_looks, n_sims=N_SIMS)
        fpr_rows.append((n_looks, naive_fpr, bonf_fpr))
        print(
            f"{n_looks:>4} looks: naive FPR {naive_fpr:.1%}, "
            f"Bonferroni FPR {bonf_fpr:.1%}"
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
            bonf = peeking_with_effect_bonferroni(
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
                    bonf.p_correct,
                    naive.p_inconclusive,
                    bonf.p_inconclusive,
                    naive.mean_users_at_decision,
                    bonf.mean_users_at_decision,
                )
            )
            print(
                f"  lift {true_lift:.3f}, {n_looks:>3} looks: "
                f"naive correct {naive.p_correct:.1%}, "
                f"Bonferroni correct {bonf.p_correct:.1%}"
            )

    with open(RESULTS / "03_bonferroni_fpr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_looks", "naive_fpr", "bonferroni_fpr"])
        for n_looks, naive_fpr, bonf_fpr in fpr_rows:
            writer.writerow([n_looks, f"{naive_fpr:.4f}", f"{bonf_fpr:.4f}"])

    with open(RESULTS / "03_bonferroni_power.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_looks",
                "true_lift",
                "naive_p_correct",
                "bonferroni_p_correct",
                "naive_p_inconclusive",
                "bonferroni_p_inconclusive",
                "naive_mean_users",
                "bonferroni_mean_users",
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
                    f"{row[6]:.1f}",
                    f"{row[7]:.1f}",
                    N_SIMS,
                ]
            )

    _plot(fpr_rows, power_rows)
    print(f"wrote {RESULTS / '03_bonferroni_fpr.csv'}")


def _plot(fpr_rows: list, power_rows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    looks = [r[0] for r in fpr_rows]
    ax.plot(looks, [r[1] for r in fpr_rows], marker="o", label="naive p < 0.05")
    ax.plot(looks, [r[2] for r in fpr_rows], marker="s", label="Bonferroni")
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title("Bonferroni holds the error rate")
    ax.legend(fontsize=8)

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
            label=f"Bonferroni, lift {true_lift:.3f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("correct-call rate")
    ax.set_title("Power cost of Bonferroni")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS / "03_bonferroni.png", dpi=150)


if __name__ == "__main__":
    main()

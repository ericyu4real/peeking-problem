"""Experiment 07: mSPRT for two proportions — always-valid stopping vs naive peeking."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import false_positive_rate, peeking_with_effect
from peeking.msprt import (
    default_tau,
    false_positive_rate_msprt,
    msprt_threshold,
    peeking_with_effect_msprt,
)
from peeking.pocock import false_positive_rate_pocock, peeking_with_effect_pocock

LOOKS = [1, 2, 5, 10, 20, 50, 100]
TRUE_LIFTS = [0.005, 0.010]
BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
TAU = default_tau(BASE_RATE)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fpr_rows = []
    power_rows = []

    print(f"mSPRT mixing std tau={TAU:.4f}, threshold Lambda={msprt_threshold(0.05):.1f}")

    for n_looks in LOOKS:
        naive_fpr = false_positive_rate(n_looks=n_looks, n_sims=N_SIMS)
        msprt_fpr = false_positive_rate_msprt(
            n_looks=n_looks, tau=TAU, n_sims=N_SIMS, base_rate=BASE_RATE
        )
        pocock_fpr = false_positive_rate_pocock(n_looks=n_looks, n_sims=N_SIMS)
        fpr_rows.append((n_looks, naive_fpr, msprt_fpr, pocock_fpr))
        print(
            f"{n_looks:>4} looks: naive FPR {naive_fpr:.1%}, "
            f"mSPRT FPR {msprt_fpr:.1%}, Pocock FPR {pocock_fpr:.1%}"
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
            msprt = peeking_with_effect_msprt(
                n_sims=N_SIMS,
                n_looks=n_looks,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                tau=TAU,
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
                    msprt.p_correct,
                    pocock.p_correct,
                    naive.p_inconclusive,
                    msprt.p_inconclusive,
                    pocock.p_inconclusive,
                )
            )
            print(
                f"  lift {true_lift:.3f}, {n_looks:>3} looks: "
                f"naive {naive.p_correct:.1%}, mSPRT {msprt.p_correct:.1%}, "
                f"Pocock {pocock.p_correct:.1%}"
            )

    with open(RESULTS / "07_msprt_fpr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_looks", "naive_fpr", "msprt_fpr", "pocock_fpr", "tau", "lambda_threshold"])
        for n_looks, naive_fpr, msprt_fpr, pocock_fpr in fpr_rows:
            writer.writerow(
                [
                    n_looks,
                    f"{naive_fpr:.4f}",
                    f"{msprt_fpr:.4f}",
                    f"{pocock_fpr:.4f}",
                    f"{TAU:.4f}",
                    f"{msprt_threshold(0.05):.4f}",
                ]
            )

    with open(RESULTS / "07_msprt_power.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_looks",
                "true_lift",
                "naive_p_correct",
                "msprt_p_correct",
                "pocock_p_correct",
                "naive_p_inconclusive",
                "msprt_p_inconclusive",
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
    print(f"wrote {RESULTS / '07_msprt_fpr.csv'}")


def _plot(fpr_rows: list, power_rows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    looks = [r[0] for r in fpr_rows]
    ax.plot(looks, [r[1] for r in fpr_rows], marker="o", label="naive p < 0.05")
    ax.plot(looks, [r[2] for r in fpr_rows], marker="s", label="mSPRT")
    ax.plot(looks, [r[3] for r in fpr_rows], marker="^", label="Pocock")
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("planned looks")
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title("mSPRT holds error under peeking")
    ax.legend(fontsize=7)

    ax = axes[1]
    for true_lift, color in ((0.005, "C0"), (0.010, "C1")):
        subset = [r for r in power_rows if r[1] == true_lift]
        ax.plot(
            [r[0] for r in subset],
            [r[2] for r in subset],
            color=color,
            linestyle=":",
            alpha=0.6,
            label=f"naive lift {true_lift:.3f}",
        )
        ax.plot(
            [r[0] for r in subset],
            [r[3] for r in subset],
            color=color,
            marker="o",
            label=f"mSPRT lift {true_lift:.3f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("planned looks")
    ax.set_ylabel("correct-call rate")
    ax.set_title("Power vs naive peeking")
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(RESULTS / "07_msprt.png", dpi=150)


if __name__ == "__main__":
    main()

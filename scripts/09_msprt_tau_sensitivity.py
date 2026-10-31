"""Experiment 09: mSPRT sensitivity to mixing variance tau; how to choose it."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking.msprt import (
    default_tau,
    false_positive_rate_msprt,
    msprt_threshold,
    peeking_with_effect_msprt,
)

BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
N_LOOKS = 100
TRUE_LIFTS = [0.005, 0.010]
# Absolute lift scale: fractions of the 10% baseline (same rule as default_tau at 0.10).
TAU_VALUES = [BASE_RATE * f for f in (0.02, 0.05, 0.10, 0.20, 0.50)]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []

    print(
        f"mSPRT tau sweep at {N_LOOKS} looks, "
        f"Lambda threshold={msprt_threshold(0.05):.1f}, n_sims={N_SIMS}"
    )

    for tau in TAU_VALUES:
        fpr = false_positive_rate_msprt(
            n_looks=N_LOOKS,
            tau=tau,
            n_sims=N_SIMS,
            base_rate=BASE_RATE,
            seed=0,
        )
        for true_lift in TRUE_LIFTS:
            summary = peeking_with_effect_msprt(
                n_sims=N_SIMS,
                n_looks=N_LOOKS,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                tau=tau,
                seed=0,
            )
            rows.append(
                (
                    tau,
                    tau / BASE_RATE,
                    true_lift,
                    fpr,
                    summary.p_correct,
                    summary.p_inconclusive,
                    summary.mean_users_at_decision,
                )
            )
        print(
            f"  tau={tau:.4f} ({tau / BASE_RATE:.0%} of baseline): "
            f"FPR {fpr:.1%}, "
            f"power lift 0.005 {rows[-2][4]:.1%}, lift 0.010 {rows[-1][4]:.1%}"
        )

    default = default_tau(BASE_RATE)
    print(f"  default_tau={default:.4f} (10% of baseline) for reference")

    with open(RESULTS / "09_msprt_tau_sensitivity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tau",
                "tau_fraction_of_baseline",
                "true_lift",
                "fpr_aa",
                "p_correct",
                "p_inconclusive",
                "mean_users_at_decision",
                "n_looks",
                "n_sims",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row[0]:.4f}",
                    f"{row[1]:.4f}",
                    f"{row[2]:.3f}",
                    f"{row[3]:.4f}",
                    f"{row[4]:.4f}",
                    f"{row[5]:.4f}",
                    f"{row[6]:.1f}",
                    N_LOOKS,
                    N_SIMS,
                ]
            )

    _plot(rows)
    print(f"wrote {RESULTS / '09_msprt_tau_sensitivity.csv'}")


def _plot(rows: list[tuple]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    taus = sorted({r[0] for r in rows})
    fpr_by_tau = {r[0]: r[3] for r in rows if r[2] == TRUE_LIFTS[0]}

    ax = axes[0]
    ax.plot(
        [t / BASE_RATE for t in taus],
        [fpr_by_tau[t] for t in taus],
        marker="o",
        color="C2",
    )
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xlabel("tau as fraction of baseline rate")
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title(f"mSPRT FPR vs tau ({N_LOOKS} looks)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for true_lift, color in zip(TRUE_LIFTS, ("C0", "C1")):
        subset = [r for r in rows if r[2] == true_lift]
        ax.plot(
            [r[1] for r in subset],
            [r[4] for r in subset],
            marker="s",
            color=color,
            label=f"true lift {true_lift:.3f}",
        )
    ax.set_xlabel("tau as fraction of baseline rate")
    ax.set_ylabel("correct-call rate")
    ax.set_title(f"Power vs tau ({N_LOOKS} looks)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "09_msprt_tau_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()

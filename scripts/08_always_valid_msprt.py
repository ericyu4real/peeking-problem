"""Experiment 08: always-valid p-values from mSPRT under continuous monitoring."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import false_positive_rate
from peeking.always_valid import (
    false_positive_rate_always_valid_continuous,
    false_positive_rate_naive_continuous,
)
from peeking.msprt import (
    default_tau,
    false_positive_rate_always_valid_msprt,
    false_positive_rate_msprt,
)

BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
TAU = default_tau(BASE_RATE)
# Dense monitoring: check every 100 or 50 users per arm (200 or 400 looks).
MONITOR_STEPS = [100, 50]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []

    for step in MONITOR_STEPS:
        n_looks = TOTAL_USERS // step
        naive_fpr = false_positive_rate_naive_continuous(
            n_looks=n_looks,
            total_users_per_arm=TOTAL_USERS,
            base_rate=BASE_RATE,
            n_sims=N_SIMS,
        )
        av_fpr = false_positive_rate_always_valid_continuous(
            n_looks=n_looks,
            total_users_per_arm=TOTAL_USERS,
            base_rate=BASE_RATE,
            tau=TAU,
            n_sims=N_SIMS,
        )
        msprt_fpr = false_positive_rate_msprt(
            n_looks=n_looks,
            total_users_per_arm=TOTAL_USERS,
            base_rate=BASE_RATE,
            tau=TAU,
            n_sims=N_SIMS,
        )
        av_via_p = false_positive_rate_always_valid_msprt(
            n_looks=n_looks,
            total_users_per_arm=TOTAL_USERS,
            base_rate=BASE_RATE,
            tau=TAU,
            n_sims=N_SIMS,
        )
        # Coarse daily peek for reference (same total n, 10 looks).
        ref_looks = 10
        naive_coarse = false_positive_rate(n_looks=ref_looks, n_sims=N_SIMS)
        rows.append(
            (
                step,
                n_looks,
                naive_fpr,
                av_fpr,
                msprt_fpr,
                av_via_p,
                naive_coarse,
            )
        )
        print(
            f"step {step:>3} ({n_looks:>3} checks): naive continuous FPR {naive_fpr:.1%}, "
            f"always-valid mSPRT FPR {av_fpr:.1%} (Lambda rule {msprt_fpr:.1%}, "
            f"p-rule {av_via_p:.1%}); naive at 10 coarse looks {naive_coarse:.1%}"
        )

    with open(RESULTS / "08_always_valid_fpr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "users_per_check",
                "n_checks",
                "naive_continuous_fpr",
                "always_valid_fpr",
                "msprt_lambda_fpr",
                "always_valid_p_fpr",
                "naive_10_looks_fpr",
                "tau",
            ]
        )
        for step, n_looks, naive_fpr, av_fpr, msprt_fpr, av_via_p, naive_coarse in rows:
            writer.writerow(
                [
                    step,
                    n_looks,
                    f"{naive_fpr:.4f}",
                    f"{av_fpr:.4f}",
                    f"{msprt_fpr:.4f}",
                    f"{av_via_p:.4f}",
                    f"{naive_coarse:.4f}",
                    f"{TAU:.4f}",
                ]
            )

    _plot(rows)
    print(f"wrote {RESULTS / '08_always_valid_fpr.csv'}")


def _plot(rows: list) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"every {r[0]} users\n({r[1]} checks)" for r in rows]
    x = range(len(rows))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        [r[2] for r in rows],
        width,
        label="naive p < 0.05 each check",
        color="C0",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [r[3] for r in rows],
        width,
        label="always-valid mSPRT p",
        color="C1",
    )
    ax.axhline(0.05, color="gray", linestyle="--", label="nominal 5%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("false-positive rate (A/A)")
    ax.set_title("Continuous monitoring: naive p vs always-valid mSPRT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "08_always_valid.png", dpi=150)


if __name__ == "__main__":
    main()

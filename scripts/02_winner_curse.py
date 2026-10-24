"""Experiment 02: with a real lift, how often naive peeking calls the right winner,
and how biased the estimated lift is at the moment it stops.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peeking import (
    critical_lift,
    fixed_sample_direction_probabilities,
    fixed_sample_mean_lift_given_correct,
    peeking_with_effect,
)

LOOKS = [1, 2, 5, 10, 20, 50, 100]
TRUE_LIFTS = [0.005, 0.010]
BASE_RATE = 0.10
TOTAL_USERS = 20_000
N_SIMS = 20_000
BY_LOOK = 20
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summaries = []
    analytic_rows = []

    for true_lift in TRUE_LIFTS:
        p_correct, p_wrong = fixed_sample_direction_probabilities(
            TOTAL_USERS, BASE_RATE, BASE_RATE + true_lift
        )
        conditional = fixed_sample_mean_lift_given_correct(
            TOTAL_USERS, BASE_RATE, BASE_RATE + true_lift
        )
        analytic_rows.append((true_lift, p_correct, p_wrong, conditional))
        print(
            f"true lift {true_lift:.3f}, one-look normal formula: "
            f"correct {p_correct:.4f}, wrong {p_wrong:.4f}, "
            f"mean lift | correct stop {conditional:.5f}"
        )

        for n_looks in LOOKS:
            summary = peeking_with_effect(
                n_sims=N_SIMS,
                n_looks=n_looks,
                total_users_per_arm=TOTAL_USERS,
                rate_control=BASE_RATE,
                rate_treatment=BASE_RATE + true_lift,
                seed=0,
            )
            summaries.append(summary)
            print(
                f"  {n_looks:>3} looks: correct {summary.p_correct:.1%}, "
                f"wrong {summary.p_wrong:.1%}, none {summary.p_inconclusive:.1%}, "
                f"lift|correct {summary.mean_lift_at_correct_stop:.5f}, "
                f"lift at end {summary.mean_lift_at_final:.5f}, "
                f"users {summary.mean_users_at_decision:.0f}"
            )

    with open(RESULTS / "02_winner_curse.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_looks",
                "true_lift",
                "p_correct",
                "p_wrong",
                "p_inconclusive",
                "mean_lift_at_correct_stop",
                "mean_lift_at_any_stop",
                "mean_lift_at_final",
                "mean_lift_at_decision",
                "mean_users_at_decision",
                "n_sims",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.n_looks,
                    f"{summary.true_lift:.3f}",
                    f"{summary.p_correct:.4f}",
                    f"{summary.p_wrong:.4f}",
                    f"{summary.p_inconclusive:.4f}",
                    f"{summary.mean_lift_at_correct_stop:.5f}",
                    f"{summary.mean_lift_at_any_stop:.5f}",
                    f"{summary.mean_lift_at_final:.5f}",
                    f"{summary.mean_lift_at_decision:.5f}",
                    f"{summary.mean_users_at_decision:.1f}",
                    summary.n_sims,
                ]
            )

    with open(RESULTS / "02_winner_curse_analytic.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["true_lift", "p_correct", "p_wrong", "mean_lift_at_correct_stop"]
        )
        for true_lift, p_correct, p_wrong, conditional in analytic_rows:
            writer.writerow(
                [
                    f"{true_lift:.3f}",
                    f"{p_correct:.4f}",
                    f"{p_wrong:.4f}",
                    f"{conditional:.5f}",
                ]
            )

    with open(RESULTS / "02_winner_curse_by_look.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "true_lift",
                "look",
                "users_per_arm",
                "n_correct_stops",
                "n_wrong_stops",
                "mean_lift_at_correct_stop",
                "critical_lift",
            ]
        )
        for summary in summaries:
            if summary.n_looks != BY_LOOK:
                continue
            for look in range(summary.n_looks):
                users = summary.users_per_look * (look + 1)
                mean_lift = summary.mean_lift_correct_by_look[look]
                writer.writerow(
                    [
                        f"{summary.true_lift:.3f}",
                        look + 1,
                        users,
                        int(summary.n_correct_by_look[look]),
                        int(summary.n_wrong_by_look[look]),
                        "" if mean_lift != mean_lift else f"{mean_lift:.5f}",
                        f"{critical_lift(users, BASE_RATE, BASE_RATE + summary.true_lift):.5f}",
                    ]
                )

    _plot(summaries)
    print(f"wrote {RESULTS / '02_winner_curse.csv'}")


def _is_lift(summary, true_lift: float) -> bool:
    # rate_treatment - rate_control is not always the decimal written in the call.
    return abs(summary.true_lift - true_lift) < 1e-9


def _plot(summaries: list) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))

    for ax, true_lift, title in (
        (axes[0, 0], 0.005, "Calls, true lift 0.005"),
        (axes[0, 1], 0.010, "Calls, true lift 0.010"),
    ):
        rows = [s for s in summaries if _is_lift(s, true_lift)]
        looks = [s.n_looks for s in rows]
        ax.plot(looks, [s.p_correct for s in rows], marker="o", label="correct direction")
        ax.plot(looks, [s.p_wrong for s in rows], marker="o", label="wrong direction")
        ax.plot(looks, [s.p_inconclusive for s in rows], marker="o", label="no call")
        ax.set_xscale("log")
        ax.set_xlabel("number of looks")
        ax.set_ylabel("share of simulations")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    for true_lift, marker, color in ((0.005, "o", "C0"), (0.010, "s", "C1")):
        rows = [s for s in summaries if _is_lift(s, true_lift)]
        looks = [s.n_looks for s in rows]
        ax.plot(
            looks,
            [s.mean_lift_at_correct_stop for s in rows],
            marker=marker,
            color=color,
            label=f"at stop, true {true_lift:.3f}",
        )
        ax.plot(
            looks,
            [s.mean_lift_at_final for s in rows],
            marker=marker,
            color=color,
            linestyle="none",
            markerfacecolor="none",
            label=f"at planned end, true {true_lift:.3f}",
        )
        ax.axhline(true_lift, color=color, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("mean estimated lift")
    ax.set_title("Lift at a correct stop vs at the planned end")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for summary in summaries:
        if summary.n_looks != BY_LOOK or not _is_lift(summary, 0.005):
            continue
        users = [summary.users_per_look * (k + 1) for k in range(summary.n_looks)]
        means = summary.mean_lift_correct_by_look
        counts = summary.n_correct_by_look
        xs = [u for u, c in zip(users, counts) if c >= 30]
        ys = [m for m, c in zip(means, counts) if c >= 30]
        boundary = [
            critical_lift(u, BASE_RATE, BASE_RATE + summary.true_lift) for u in users
        ]
        ax.plot(users, boundary, linestyle="--", color="gray", label="significance boundary")
        ax.plot(xs, ys, marker="o", label="mean lift at that stop")
        ax.axhline(summary.true_lift, color="black", linestyle=":", label="true lift 0.005")
    ax.set_xlabel("users per arm when the test stops")
    ax.set_ylabel("mean estimated lift")
    ax.set_title("20 looks, true lift 0.005, by stopping time")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "02_winner_curse.png", dpi=150)


if __name__ == "__main__":
    main()

# Research log

## 2026-10-31 — 09 mSPRT sensitivity to mixing variance τ

Swept τ from 2% to 50% of the 10% baseline on 100 equal looks (20,000 users per arm, 20,000 sims). A/A false-positive rate rises from 0.01% at τ = 0.002 to 2.26% at τ = 0.02, all at or below 5%. With true lift 0.010, correct-call rate peaks at 68.9% for the default τ = 0.01 versus 10.2% at τ = 0.002 and 56.7% at τ = 0.05; lift 0.005 stays below 15.6% across the grid. Tests check null-ratio monotonicity in τ, nominal error control across τ, and higher power when τ matches detectable effects.

Next: Bayesian stop when P(B > A) > 95% (experiment 10).

## 2026-10-30 — 08 Always-valid p-values from mSPRT

Defined always-valid p-values ``p_t = min(1, 1/max Λ)`` from the experiment 07 mixture ratio; stopping on ``p_t ≤ 0.05`` is pathwise identical to ``Λ ≥ 20``. On A/A paths with 20,000 users per arm, checking every 100 users (200 looks) gives naive continuous FPR 42.2% versus 1.8% for always-valid mSPRT; every 50 users (400 looks) gives 46.4% versus 1.9%. Coarse naive peeking at 10 batches stays at 19.2%. Tests check non-increasing ``p_t``, equivalence of the two stopping rules, and nominal error control on dense schedules.

Next: sensitivity of mSPRT to mixing variance (experiment 09).

## 2026-10-29 — 07 mSPRT for two proportions

Implemented the Bernoulli mixture likelihood ratio with a normal N(0, τ²) prior on the lift (default τ = 0.01). Stop when Λ ≥ 20 (α = 0.05). On equal batches with 20,000 users per arm, mSPRT A/A false-positive rate is 0.9% at 10 looks and 1.7% at 100 looks versus 19.2% and 37.1% naive; Pocock stays at 5.0%. With true lift 0.010 and 100 looks, correct-call rate is 68.9% for mSPRT versus 76.1% Pocock and 93.5% naive. Tests check the null closed form, single-look FPR ≤ α, and nominal error control across the look grid.

Next: always-valid p-values from mSPRT (experiment 08).

## 2026-10-28 — 06 Lan–DeMets alpha spending

Implemented O'Brien–Fleming alpha spending with boundaries keyed to information fraction (cumulative sample / planned total), using the same binomial-calibrated multiplier ``c`` as experiment 05. On 10 looks and 20,000 users per arm, equal batches give Lan–DeMets FPR 5.0% (same as calibrated OBF). An irregular front-loaded schedule (500-user early peeks up to 4,000-user batches) gives naive FPR 23.9% versus Lan–DeMets 4.8%; mis-indexing look ``k`` as fraction ``k/K`` instead of the true information fraction yields 5.7%. With true lift 0.010, correct-call rate is 89.3% on the irregular schedule versus 93.1% naive. Tests check spending at ``t=1``, equal-batch boundaries matching OBF, and nominal FPR on both schedules.

Next: mSPRT for two proportions.

## 2026-10-27 — 05 O'Brien–Fleming boundary

Calibrated O'Brien–Fleming thresholds c·√(K/k) at each look so A/A false-positive rate is 5% (20,000 calibration sims per K). At 100 looks the first look needs |z| > 21.83 versus 2.18 at the final look; Pocock uses 2.865 at every look. With true lift 0.010 and 100 looks, correct-call rate is 89.0% for O'Brien–Fleming versus 76.1% Pocock and 93.5% naive. Tests check nominal FPR and final-look c against classical OBF tables.

Next: Lan–DeMets alpha spending for unplanned look times.

## 2026-10-26 — 04 Pocock boundary

Calibrated a constant two-sided z boundary at each of K equally spaced looks so A/A false-positive rate is 5% (20,000 calibration sims per K). At 100 looks, z* is 2.865; Pocock FPR is 5.0% versus 37.1% naive and 0.9% Bonferroni. With true lift 0.010 and 100 looks, correct-call rate is 76.1% for Pocock versus 51.4% Bonferroni and 93.5% naive. Tests check nominal FPR after calibration and K=2,5 boundaries against classical Pocock tables.

Next: O'Brien–Fleming boundary and power comparison with Pocock.

## 2026-10-25 — 03 Bonferroni across looks

Bonferroni (p < 0.05/K at each of K looks) holds the A/A false-positive rate at or below 5% for every peek count in the grid; at 100 looks it is 0.86% versus 37.1% naive. With true lift 0.010 and 100 looks, correct-call rate falls from 93.5% to 51.4% and inconclusive runs rise to 48.6%. Tests check the union-bound ceiling on FPR and one-look power against the normal formula.

Next: Pocock boundary calibrated to 5% overall.

## 2026-10-24 — 02 Winner's curse

Simulated naive peeking with true lifts of 0.005 and 0.010 (control rate 10%, 20,000 users per arm, 20,000 simulations). One look matches the truncated-normal formula: a true lift of 0.005 is estimated at 0.00806 given a correct call (formula 0.00805). With 100 looks that conditional estimate is 0.01968, while the planned-end average stays at 0.00498. On 20 looks, stopping at the first batch (1,000 users) averages 0.03245 on a correct call.

Next: Bonferroni across looks, and how much power it costs.

## 2026-10-24 — 01 Naive peeking

Built the A/A simulator (two-proportion z-test on cumulative data, equal batches) and measured the false-positive rate of "stop at the first p < 0.05". One look gives 5.0%, 10 looks 19.7%, 100 looks 37.1%, matching Armitage et al. (1969). Tests pin the 1-look and 10-look values and check monotonicity.

Next: add a true effect and measure how inflated the estimated lift is at the moment of stopping.

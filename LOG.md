# Research log

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

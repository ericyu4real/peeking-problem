# Research log

## 2026-10-24 — 02 Winner's curse

Simulated naive peeking with true lifts of 0.005 and 0.010 (control rate 10%, 20,000 users per arm, 20,000 simulations). One look matches the truncated-normal formula: a true lift of 0.005 is estimated at 0.00806 given a correct call (formula 0.00805). With 100 looks that conditional estimate is 0.01968, while the planned-end average stays at 0.00498. On 20 looks, stopping at the first batch (1,000 users) averages 0.03245 on a correct call.

Next: Bonferroni across looks, and how much power it costs.

## 2026-10-24 — 01 Naive peeking

Built the A/A simulator (two-proportion z-test on cumulative data, equal batches) and measured the false-positive rate of "stop at the first p < 0.05". One look gives 5.0%, 10 looks 19.7%, 100 looks 37.1%, matching Armitage et al. (1969). Tests pin the 1-look and 10-look values and check monotonicity.

Next: add a true effect and measure how inflated the estimated lift is at the moment of stopping.

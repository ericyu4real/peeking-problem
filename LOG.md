# Research log

## 2026-10-24 — 01 Naive peeking

Built the A/A simulator (two-proportion z-test on cumulative data, equal batches) and measured the false-positive rate of "stop at the first p < 0.05". One look gives 5.0%, 10 looks 19.7%, 100 looks 37.1%, matching Armitage et al. (1969). Tests pin the 1-look and 10-look values and check monotonicity.

Next: add a true effect and measure how inflated the estimated lift is at the moment of stopping.

# Roadmap

Each item should fit in one focused session: code + tests + a short write-up. Split anything bigger. Add new questions to **Ideas** when you find them.

## Done

- [x] 01 Naive peeking: false-positive rate vs number of looks (A/A)
- [x] 02 Peeking with a real effect: does early stopping still reach the right answer, and how biased is the estimated lift at stopping time? (winner's curse)
- [x] 03 Bonferroni across looks: simple fix, measure how much power it loses
- [x] 04 Pocock boundary: constant threshold calibrated by simulation to give 5% overall

## Next

- [x] 05 O'Brien–Fleming boundary: strict early, lenient late; compare power with Pocock
- [ ] 06 Lan–DeMets alpha spending: allow unplanned / irregular look times
- [ ] 07 mSPRT (mixture sequential probability ratio test) for two proportions
- [ ] 08 Always-valid p-values from mSPRT; verify the false-positive rate holds under continuous monitoring
- [ ] 09 Sensitivity of mSPRT to the mixing variance; how to choose it
- [ ] 10 Bayesian "stop when P(B > A) > 95%": does it control false positives? (spoiler: check it)
- [ ] 11 Bayesian expected-loss stopping rule
- [ ] 12 Head-to-head: false-positive rate, power, expected sample size, and bias of the estimated lift for every method at several effect sizes
- [ ] 13 Irregular traffic: weekday/weekend swings in batch size
- [ ] 14 Novelty effect: a lift that decays over time; which methods get fooled?
- [ ] 15 Continuous (revenue-like, heavy-tailed) metric instead of conversion
- [ ] 16 CUPED variance reduction combined with sequential testing
- [ ] 17 Many metrics at once: peeking × multiple comparisons
- [ ] 18 Practical guide in the README: "if you must peek, do this"

## Ideas

- Interactive plot of boundaries
- Analytical approximation vs simulation for naive peeking (Brownian-motion crossing)
- Median-unbiased estimate of the lift at the look where a naive peeking test stops
- If collection runs to the planned end anyway, how biased is that final lift conditional on an earlier look already having p < 0.05?

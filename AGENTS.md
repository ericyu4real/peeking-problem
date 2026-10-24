# Instructions for the daily research agent

This repo answers one question: how badly does peeking at A/B tests inflate false positives, and which remedy works best. Each session makes one real, tested step of progress.

## Each session

1. Read `README.md`, `ROADMAP.md`, and the last few entries of `LOG.md`.
2. Pick the first unchecked item under **Next**. If it is too big for one session, split it in `ROADMAP.md` and do the first part. If you notice a worthwhile follow-up question, add it under **Ideas** or **Next**.
3. Implement it in `src/peeking/` with a script in `scripts/` (numbered like `02_...`) that writes outputs to `results/`.
4. Add pytest tests for the new logic. Prefer checks against known theory or analytic values over "it runs".
5. Run `pytest`. Do not commit if any test fails; fix it first.
6. Run the experiment script and copy the real numbers into a new section under **Findings so far** in `README.md`. Never invent or round-trip numbers you did not compute.
7. Tick the item in `ROADMAP.md` and add a dated entry to `LOG.md` (what was done, key number, what's next).
8. Commit and push to `main` (see below).

## Commit rules

```bash
git config user.name "Eric Yu"
git config user.email "ericjiangshanyu@gmail.com"
git add -A
git commit -m "<NN> <short description of the finding>"
git push origin HEAD:main
```

- Push directly to `main`; do not open a pull request.
- One meaningful commit per session (a second small fix commit is fine). No empty commits, no backdated commits.
- Keep simulations fast enough that the full test suite runs in under a minute.

## Content rules

- Do not mention any company, employer, product, or interview anywhere in code, docs, data, or commit messages.
- Use simulated data only, or clearly cited public sources.
- Keep the writing plain and factual.

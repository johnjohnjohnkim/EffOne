# AGENTS.md

Instructions for any coding agent (Codex, Claude Code, etc.) working in this repo.
Read `HANDOFF.md` for the goal, decisions, phase list and progress log. Update its progress log whenever you finish a unit of work.

## Project summary

EffOne predicts F1 qualifying and races for the current season (win, podium and top-10 probabilities, plus season champion simulation), using fastf1 data. Python backend and ML, FastAPI API, Next.js + TypeScript frontend, deployed with Docker on AWS; the frontend is a static export on Cloudflare Pages.

## Environment

- Windows 11. The primary shell is PowerShell; Git Bash is also available.
- Python **3.12** managed with `uv`. Do not use the system Python 3.14.
- Node 24 for `web/`.
- Never commit `data/`, fastf1 cache, model artifacts, `.env` files or credentials.

## Rules

1. **Do not skip phases.** Follow the order in `HANDOFF.md`. Stop at the review checkpoints (after phases 6 and 7) and ask the user.
2. **No data leakage.** A feature for a given session may only use data available before that session. Every feature builder needs a test proving it.
3. **Walk-forward validation only.** Never random k-fold. Compare every model with the baselines ("grid = finish", "previous race = finish"); a new model must beat them on held-out seasons.
4. **Training data is 2018 onward.** Do not add pre-2018 data to training. It may be used later as a test set only.
5. **Circuit-history rule:** past races at a circuit count toward track-specific features only if at least 25% of the current grid's drivers raced there that year. Otherwise fall back to circuit-type features with shrinkage.
6. **Refreshes are incremental.** The scheduled job updates features and ratings and reruns simulations. It must not do a full retrain. Full retrains are manual.
7. **Ingestion must be idempotent and resumable.** Cache fastf1 data, retry on incomplete sessions, and make reruns add nothing new.
8. **Output probabilities, not single picks.** Use Monte Carlo and check calibration.
9. **Log predictions before the race** and score them afterwards.
10. Strategy (pit stops) and safety-car prediction are out of scope until the user says otherwise.

## AWS and cost safety

- The user is new to AWS. Explain each step, and say what it costs before creating anything.
- Do not run `terraform apply`, create or delete cloud resources, or change IAM without the user's explicit approval in the current conversation.
- Use least-privilege credentials, not root. Never write secrets to files in the repo.
- Recommend a billing alarm as the very first AWS step.

## Git

- Commit or push only when the user asks.
- Do **not** add AI attribution to commits or PRs: no `Co-Authored-By` trailer and no "Generated with ..." line. Commits should show only the user as author.
- Keep `data/`, caches, `.env` and model artifacts out of git (see `.gitignore`).

## Code conventions

- Python: type hints, `ruff` for lint and format, `pytest` for tests. Keep modules small: `ml/ingest`, `ml/features`, `ml/models`, `ml/evaluation`, `ml/simulation`.
- API: FastAPI with Pydantic schemas; load models once at startup; enable CORS only for the configured site origin.
- Web: Next.js App Router with TypeScript, static export, no server-only features (Cloudflare Pages hosts static files). Read the API base URL from an environment variable.
- Match the style of the existing code. Keep comments sparse and explain *why*, not what.
- Add tests alongside each phase. Do not mark a phase done unless its "done when" criterion in `HANDOFF.md` is met.

## Commands (fill in as they are created)

```
# install uv (once): pip install --user uv   (then call it as `python -m uv ...` if `uv` is not on PATH)
# setup:   python -m uv sync
# test:    python -m uv run pytest
# lint:    python -m uv run ruff check . && python -m uv run ruff format --check .
# ingest:  (to be defined in phase 1)
# api:     (to be defined in phase 8)
# web:     (to be defined in phase 9)
```

## Handoff checklist

When stopping, whether you are out of tokens or the session is ending:
1. Make sure the tests pass, or note exactly which fail and why.
2. Add a dated entry to the progress log in `HANDOFF.md`: what was done, what is in progress, the next concrete step, and any decisions or surprises.
3. Update the "Commands" section above if new commands exist.
4. Do not leave half-finished edits that break the build without saying so in the log.

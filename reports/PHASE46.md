# Phase 4.6 — replay integrity hardening

This milestone changes replay plumbing only. The Phase 1–4 models and the
fixed Phase 4.5 hierarchical allocation and EV rule are untouched. September
13, 2026 remains `diagnostic_exhibition_slate = TRUE`. Its prediction CSV,
quote CSV, settlement CSV, audit, and sensitivity results were not rewritten.
The original prediction SHA-256 remains
`41d9e8f6af1a7550248c04abfa90dc4b2d20ca69e12e408d4b44c23dc79ffc62`;
the settled CSV SHA-256 remains
`cf52d52962acdef895a2ff9e038c6dca5f6fd258d852f77e0011244df46b9f93`.

The prospective collector projects live event and market payloads to an
allowlisted pregame JSON view. It uses only the unchanged frozen 2025 prior
football history and the Phase 4 market-only team model. Market updates after
T−60 are dropped. The Stage A worker runs in its own Python process, reads
that closed-schema JSON alone, and rejects any unknown field in nested player,
quote, and availability records. It imports neither network nor settlement
readers. Quotes and predictions freeze in separate files with SHA-256 hashes
in a manifest. Stage A predictions contain no settlement/result placeholders.
Stage B is a distinct command, verifies all three Stage A hashes before
requesting a score, and fetches a game summary only when its scoreboard status
is final. Its output lives under `reports/phase46_settlement/`, outside the
Stage A artifact tree. These are code and data-interface barriers; they
are not an operating-system security sandbox against a malicious actor with
access to the workspace.

The prospective baseline intentionally retains 2025-only player history and
the existing 2021–2023 team-model fit. It does not assimilate 2026 games or
update player allocation during this infrastructure phase. Probabilities may
therefore become stale as the 2026 season proceeds; this is a model limitation,
not a reason to use September 13 outcomes for an adjustment.

The T−60 prediction row now keeps `eligible_at_prediction_time`,
`inactive_known_at_prediction_time`, `ultimately_played`, and
`settlement_status` separately. An official NFL inactive/out record published
and retrieved by T−60 prevents a bet but leaves the quote and probability
visible. Eligibility additionally requires evidence of complete official
status coverage for that player's team and an affirmative official active
status for the player; missing or conflicting evidence fails closed for
betting. Later non-participation never changes the frozen selection. A DNP
is graded only with affirmative participation evidence and a sourced rule
for that specific sportsbook. Otherwise it remains pending. Absence from an
ESPN statistical boxscore is treated as unknown participation, not DNP.

Every captured quote retains book, American price, timestamp, and age at
T−60. The player row retains book count, best and median prices, and a
descriptive flag when the best decimal payoff exceeds the median by at least
25%. That flag has no effect on eligibility or the unchanged bet rule. The
threshold is a fixed data-quality marker, not a number chosen from September
13 outcomes.

On this Windows host, `scripts/phase46_tasks.ps1` registered one daily
discovery task and one T−60 task per currently listed future event. Live
event discovery was verified; the API documents that endpoint as no-quota.
The event task trigger was checked against its actual `NextRunTime` after
registration. The next listed Monday game, for example, was scheduled for
**September 14 at 19:15 EDT**, exactly 60 minutes before its listed 20:15
EDT kickoff. The task permits battery operation and requests wake-to-run.
It still requires the machine, account session, network, API key, and source
data to be available. A missed T−60 window fails closed; it does not fabricate
or backfill a prospective price.

No official machine-readable full inactive feed is currently integrated.
Until a complete contemporaneous official source is supplied in the documented
availability-file format, automated predictions and quote files can freeze,
but **no wager is eligible**. An operator-provided file's source/timestamps
are checked against T−60, but the operator must ensure the cited source truly
documents complete game-day status. This is a deliberate remaining limit,
not a reason to reuse the September 13 retrospective injury feed.

No September 13 outcome entered a prediction, feature, model, rule, or
threshold decision in this phase. The September 13 one-day ROI remains a
flawed exhibition diagnostic and is not restated as a corrected portfolio.
No Phase 5 model or team TD environment work was started.

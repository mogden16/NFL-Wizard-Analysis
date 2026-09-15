# Receptions backfill runbook

This runbook is for the 2023 and 2024 regular-season `player_receptions`
snapshots only. The 2025 holdout is sealed.

1. Confirm the available quota is at least the value printed by:

   ```powershell
   .venv\Scripts\nfl-td.exe receptions-backfill-plan
   ```

2. Run the guarded download. It performs no target request unless the full
   remaining credit requirement is available:

   ```powershell
   .venv\Scripts\nfl-td.exe receptions-backfill --execute --available-credits <remaining>
   ```

   Cached requests are reused. Use `--force` only for an explicitly approved
   repair; it intentionally bypasses the immutable request cache.

3. Normalize cached responses without network access:

   ```powershell
   .venv\Scripts\nfl-td.exe receptions-backfill-normalize
   ```

The raw response cache is content-addressed under
`data/raw/the_odds_api/`; request indexes omit credentials. Normalization
rejects market updates after the requested T-60 cutoff, pairs Over and Under
only within the same sportsbook/player/exact line, and preserves unmatched or
ambiguous player names for audit.

Do not join predictions or evaluate outcomes until raw-response hashes,
returned timestamps, missing games, and player matching have been reviewed.

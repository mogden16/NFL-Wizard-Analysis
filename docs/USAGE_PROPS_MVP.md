# NFL Usage Prop Research MVP

This extension reuses the Phase 2 lagged player feature tables, the frozen
season-specific nflverse player-stat snapshots, `market_math` American-odds
conversion, the existing Odds API event client pattern, and the Typer test and
reporting setup. It does not alter any ATD model and does not read 2025.

The documented Odds API identifiers are `player_receptions` and
`player_rush_attempts` (both Over/Under markets). The live scanner groups
quotes by player, prop, and exact line, pairs Over and Under within each book,
and applies proportional no-vig probabilities. It requires three paired books
for a consensus comparison; all quotes and lower-coverage markets remain in
the output. Median prices are calculated from decimal payoffs before converting
back to a descriptive American equivalent.

Validation uses 2017–2023 training and 2024 validation. There are 53,816
training and 8,108 validation player-games per prop. The EWMA historical usage
baseline is retained for live means because the Poisson challenger did not
improve the primary count errors:

| Prop | Model MAE | EWMA MAE | Model RMSE | EWMA RMSE | Model deviance | EWMA deviance |
|---|---:|---:|---:|---:|---:|---:|
| Receptions | 1.187 | 1.117 | 1.805 | 1.753 | 1.817 | 2.241 |
| Rushing attempts | 1.552 | 1.233 | 3.102 | 2.771 | 2.627 | 2.336 |

The receptions Poisson model lowers count deviance but worsens MAE/RMSE; the
rushing-attempt model is worse on all three metrics. This MVP therefore uses
EWMA means for the live display. The mean model remains frozen at EWMA. Distribution calibration compares Poisson, training-only Negative Binomial, and empirical residual methods in `reports/usage_prop_calibration.json`. Negative Binomial is selected by 2024 probability log loss for both props (receptions 0.752 versus Poisson 0.768; rushing attempts 0.535 versus 0.552). Training variance/mean is 3.33 for receptions and 11.02 for rushing attempts, so Poisson understates dispersion. Because no timestamped historical receptions or rushing-attempt lines are archived, the market-matched validation population is zero and both markets remain `RESEARCH_ONLY`; no profitability conclusion is drawn.

Run:

```powershell
./.venv/Scripts/nfl-td.exe usage-prop-validate
./.venv/Scripts/nfl-td.exe usage-prop-scan`r`n./.venv/Scripts/nfl-td.exe usage-prop-audit
```

The current live run found 20 player-lines (18 with a historical EWMA match)
and 118 underlying sportsbook quotes. Thirteen lines had at least three paired
books and therefore a no-vig consensus comparison. The largest displayed
disagreements were receptions: Marvin Mims Jr. +47.0 percentage points, Xavier
Worthy +36.5 pp, Evan Engram +35.6 pp, Travis Kelce +31.3 pp, and Noah Gray
+25.8 pp. These are model-versus-market diagnostics, not bets or guarantees.

The disagreement audit in `reports/usage_prop_disagreement_audit.json` records prior games, source-game counts, team changes, exact lines, quote timestamps, and identity checks without loading current outcomes or 2025. The main limitation is historical role continuity: live 2026 players are
matched to their latest available frozen 2024 EWMA history by normalized name,
while the roster supplies only descriptive current team labels. Players with
no safe prior match receive no model probability. Route participation is not
used.



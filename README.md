# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/taigedi123/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/taigedi123/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts each of Grand Theft Auto VI's 55 PlayStation 5 trophies' earn
rate on Exophase at the case target, as five quantiles (0.05 / 0.25 /
0.5 / 0.75 / 0.95). Entry for make.fun's
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty ($250 in USDC). Built by an AI agent for that bounty.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

A trophy's earn rate is earners divided by players tracked, and the two
move differently, so they are forecast separately and divided at the end:

- **Players tracked** — a power law in time since release, fitted in
  log space (with a quadratic term where the data supports it, since
  the growth curve is curved on that axis), on the last 24 hours of
  reads, falling back to longer windows when the recent window holds
  too few distinct values.
- **Earners per trophy** — the same time axis, one fit per trophy,
  each slope shrunk toward the pooled slope so small-count trophies
  borrow the crowd's growth while trophies with a real trend keep it.
  Story trophies grow fast, platinum and grind trophies slowly; the
  fit learns that from the history.
- **Median rate** — forecast earners over forecast players. Over a
  week the player base grows faster than most trophies are earned, so
  most medians fall below the last read (dilution); easy early
  trophies whose earners outrun the player base rise.
- **Spread** — a logit-normal band, asymmetric (wider below the
  median: a refresh can halve rates overnight, the upside is bounded
  by play time and by 100%), combining a random-walk term, the fits'
  residual scatter, and a refresh-shock term. A trophy at 100% in
  every read keeps a median of 100 with a fixed downside tail.
- **Flat history** (no read ever changed, as on launch night) — no
  trend can be fitted, so the forecast assumes the pending first
  refresh the kit documents (tracked players 638 → 3,868, newcomers
  earning at 15% of the early cohort's rate) and widens the spread a
  lot.

Fully deterministic: no randomness, no clock reads beyond the case
file itself, so two runs produce byte-identical output.

## Validation (local)

On the kit's practice cases (`p00`, `p01`):

- both outputs valid: exactly 55 rows, 5 non-decreasing quantiles
  each in 0–100, files under 65,536 bytes;
- two runs byte-identical;
- runtime ~0.07 s for both cases (limit 8 s per case).

The practice answers are the refreshed Exophase rates, which are not
in the kit and cannot be fetched, so no exact practice loss can be
computed locally. Against an honest proxy of the refresh (players
638 → 3,868, newcomers earning at 0–30% of the early cohort's rate)
this forecaster scores roughly 0.34–0.45 pinball loss, against
0.78–1.69 for the house baseline — but that proxy is an estimate,
not a measured grade. On synthetic final-like cases (342 reads, 6.5-day
horizon, three different growth/refresh regimes) it scores 0.30–0.40
against 0.28–0.71 for the baseline: it wins when the truth moves and
roughly ties when it barely does. The final is a 6.5-day extrapolation
whose true player growth is unknown, so the honest spread is wide and
the grade will depend on how far reality drifts from the fitted trends.

## License

MIT — see [LICENSE](LICENSE).

# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DianChairo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DianChairo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each GTA VI trophy's Exophase earn rate one week after launch, as
five quantiles per trophy, scored by pinball loss on the logit scale.

Method: per trophy, a least-squares line of logit(rate) against log-time
over the last 24 hours of reads, read at the target with the extrapolated
move damped (x0.6) and shrunk toward the trophy's tier mean, anchored on
the last observed rate. The spread combines the house baseline's random
walk (0.5 per root-day), the trend's estimation error, and a 0.5-logit
floor for the refresh jumps no history can foresee — on launch night one
refresh cut every non-100% rate to roughly a fifth, which is why the
house baseline's narrow band scores about 1.51 on the practice. The
platinum is capped at the rarest other trophy's quantile at each level,
since earning every trophy is required for it. Standard library only,
offline, deterministic.

Local validation: valid forecasts for both practice cases (55 trophies
each, five non-decreasing quantiles, under the 65,536-byte cap),
byte-identical across repeat runs, about 0.06 s per run.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8tpwxchz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

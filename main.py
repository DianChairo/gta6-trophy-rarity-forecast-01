"""Forecast GTA VI's trophy earn rates on Exophase one week after launch.

Method: for each trophy, work on the logit scale, where earn rates move
roughly additively. Two forces move a rate: players keep earning the
trophy (accumulation) while new players keep joining Exophase's tracked
pool (dilution, which drags every rate down at each refresh -- on launch
night one refresh cut every non-100% rate to roughly a fifth of itself). The case's
history already nets those forces, so per trophy we fit a least-squares
line of logit(rate) against log-time, x = ln(1 + hours since the release
day began), over the last 24 hours of reads. The line is read at the
target, but the extrapolated move is damped (x0.6) and shrunk toward the
trophy's tier mean, because a log-time line fitted on two days of data
overshoots over a 4.5-day horizon. The median is anchored on the last
observed rate, not on the fitted value, so a single noisy read cannot
drag the forecast.

The spread is the part the practice punishes: the house persistence
baseline scores about 1.51 on the practice because its narrow band sits
outside every rate after one refresh. So the spread combines three
terms: the baseline's random walk (0.5 * sqrt(horizon in days)), the
trend's own estimation error (slope standard error times the
extrapolation distance), and a flat 0.5-logit floor for the refresh
jumps no history can foresee. The platinum is capped at the rarest other
trophy's quantile at each level: earning every trophy is required for it,
so its rate can never exceed any other trophy's.

Interface (per the competition rules): read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, else /input), write /output/<id>.out
(MAKEFUN_OUTPUT_DIR, else /output) as {"trophies": [{"id": n, "q": [q05,
q25, q50, q75, q95]}, ...]}. Standard library only, offline,
deterministic, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The release day begins at 00:00 UTC on the game's release date; log-time
# is measured from there.
RELEASE_START = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Only the most recent reads carry the current regime (player growth
# slowing, progression accumulating); older reads are a different launch.
WINDOW_HOURS = 24.0
# How much of the fitted trend's extrapolated move to trust.
DAMP = 0.6
# Shrinkage strength of a trophy's own slope toward its tier mean:
# weight w = n / (n + SHRINK_K) on the trophy's own slope.
SHRINK_K = 3.0
# The random-walk spread per square-root day, as in the house baseline.
RW_SIGMA = 0.5
# Flat floor for refresh jumps no history can foresee (one launch-night
# refresh moved every non-100% rate about 1.5 logits).
SHOCK = 0.5
# Widest spread allowed, and the largest median move from the last rate.
MAX_SPREAD = 2.5
MIN_SPREAD = 0.35
MAX_MOVE = 2.5
# Fallback rate (percent) for a trophy with no usable read at all.
FALLBACK_RATE = 1.0


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    """logit of a percent, with the rate clamped to [0.01, 99.99]."""
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    """Percent whose logit is lg, guarded against overflow."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def log_time(when):
    """ln(1 + hours since the release day began), floored at 0."""
    hours = max(0.0, (when - RELEASE_START).total_seconds() / 3600.0)
    return math.log1p(hours)


def fit_line(points):
    """Least-squares line through (x, y) points.

    Returns (slope, intercept, slope_se, resid_std), or None when the
    points do not pin a line down (fewer than 3 points or no x spread).
    """
    n = len(points)
    if n < 3:
        return None
    xs = [p[0] for p in points]
    if max(xs) - min(xs) < 1e-9:
        return None
    mx = sum(xs) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [p[1] - (intercept + slope * p[0]) for p in points]
    dof = max(1, n - 2)
    var = sum(e * e for e in resid) / dof
    slope_se = math.sqrt(var / sxx)
    return slope, intercept, slope_se, math.sqrt(var)


def last_valid_rate(history, index):
    """The trophy's rate in the latest read that carries it, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return None


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_time(target)
    x_last = log_time(at)

    # The fitting window: the last 24 hours of reads, or all of them when
    # the case holds fewer than 3.
    cutoff = at.timestamp() - WINDOW_HOURS * 3600.0
    window = [r for r in history if parse_time(r["at"]).timestamp() >= cutoff]
    if len(window) < 3:
        window = history
    window_x = [log_time(parse_time(r["at"])) for r in window]

    # One least-squares line of logit(rate) on log-time per trophy.
    own_slope = {}
    slope_se = {}
    for i, trophy in enumerate(trophies):
        points = []
        for x, read in zip(window_x, window):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                points.append((x, logit(rates[i])))
        fit = fit_line(points)
        if fit is None:
            own_slope[i] = None
            slope_se[i] = 0.0
        else:
            own_slope[i] = fit[0]
            slope_se[i] = fit[2]

    # Tier means, for shrinking trophies whose own line is thin or absent.
    tier_mean = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [own_slope[i] for i, t in enumerate(trophies)
                if t["tier"] == tier and own_slope[i] is not None]
        tier_mean[tier] = sum(vals) / len(vals) if vals else 0.0
    fitted = [v for v in own_slope.values() if v is not None]
    global_mean = sum(fitted) / len(fitted) if fitted else 0.0

    dx = x_target - x_last
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_valid_rate(history, i)
        anchor = logit(rate if rate is not None else FALLBACK_RATE)
        if own_slope[i] is None:
            slope = tier_mean.get(trophy["tier"], global_mean)
            fit_unc = 0.5
        else:
            npts = sum(1 for read in window
                       if i < len(read.get("rates") or [])
                       and read["rates"][i] is not None)
            w = npts / (npts + SHRINK_K)
            slope = w * own_slope[i] + (1.0 - w) * tier_mean.get(
                trophy["tier"], global_mean)
            fit_unc = slope_se[i] * dx
        move = DAMP * slope * dx
        move = max(-MAX_MOVE, min(MAX_MOVE, move))
        median = anchor + move
        spread = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                           + fit_unc ** 2 + SHOCK ** 2)
        spread = min(MAX_SPREAD, max(MIN_SPREAD, spread))
        quantiles = [round4(clamp_rate(inv_logit(median + z * spread)))
                     for z in Z]
        # Keep the row non-decreasing even if rounding ties two neighbours.
        cleaned = []
        floor = 0.0
        for q in quantiles:
            if q < floor:
                q = floor
            cleaned.append(q)
            floor = q
        rows.append({"id": trophy["id"], "q": cleaned})

    # The platinum needs every other trophy, so its rate can never exceed
    # any other trophy's: cap it at the rarest other trophy, level by level.
    platinum = [r for r, t in zip(rows, trophies) if t["tier"] == "platinum"]
    if platinum:
        row = platinum[0]
        for k in range(5):
            cap = min(r["q"][k] for r in rows if r is not row)
            if row["q"][k] > cap:
                row["q"][k] = cap
        for k in range(1, 5):
            if row["q"][k] < row["q"][k - 1]:
                row["q"][k] = row["q"][k - 1]

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

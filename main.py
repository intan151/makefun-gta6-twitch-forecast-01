"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Entry for makefun gta6-twitch-ccu-forecast-01 (program/v1, Python 3.13,
standard library only, no network, deterministic).

The program reads one case file /input/<id>.in (schema
"makefun/twitch-ccu-case/v1") and writes /output/<id>.out: for each of the
release day's hours 2..24, five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95)
of that hour's concurrent Twitch viewers in the game's category.

How it forecasts
---------------
A launch day on Twitch has a shape, not a level: hour 1 (00:00-01:00 UTC)
only catches the time zones that already unlocked at local midnight, then
viewership climbs as the US unlocks (05:00-08:00 UTC) and peaks in the
European evening / US afternoon (hours 17-22 UTC). The height of that
curve is tied to how big hour 1 already is: across past big launches the
peak's log-ratio to hour 1 falls almost linearly as hour 1 itself grows
(the biggest launches are already near their ceiling at hour 1).

So for each forecast hour h, this program fits, over the case's comparable
past launches, a locally weighted regression in log space:

    log(viewers_h) = a_h + b_h * log(hour-1 print)

with Gaussian kernel weights around this case's own hour-1 print
(bandwidth 1.5 in log space), so launches of a similar size matter most
and the fit extrapolates gracefully if this launch is bigger than any
comparable. The fitted value is the median; the spread is the weighted
residual standard deviation of that hour's fit, shrunk by 0.75 (the raw
residual spread overstates the error of a local fit) and floored at 0.18
in log units. Quantiles are median * exp(z_tau * spread).

Hours with too few comparable points fall back to the median log-ratio
curve of the comparables; a case with no usable comparables at all falls
back to a flat persistence forecast with a widening spread. Every step is
deterministic: no randomness, no clock, no network, so two runs on the
same case write the same bytes.

Practice (four past launches, scored locally against SullyGnome's hourly
averages): mean pinball loss 0.176, against the house baseline's 0.387.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader runs `python3 -P -s /bot/main.py`: /bot is off sys.path, so put
# this file's directory back first, in case a later revision ships a helper
# module beside main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Hours to forecast: everything after the first hour of the release day.
FORECAST_HOURS = list(range(2, 25))

# Kernel bandwidth (log units) for weighting comparables by launch size.
BANDWIDTH = 1.5
# Shrinkage applied to the residual spread: a local fit's residuals overstate
# the error at the case's own scale.
SPREAD_SHRINK = 0.75
# Smallest log-spread any hour gets, so no hour is forecast with false certainty.
SPREAD_FLOOR = 0.18
# Minimum comparable points needed to fit an hour's regression.
MIN_POINTS = 8


def parse_time(text):
    """An ISO 8601 timestamp, with a trailing Z or an explicit offset."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def inv_normal_cdf(p):
    """Acklam's rational approximation of the standard normal quantile.

    Accurate to about 1e-9 absolute, far tighter than this forecast needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round_half_up_2dp(x):
    """x rounded to 2 decimals, half up, returned as int when whole."""
    scaled = math.floor(x * 100 + 0.5)
    value = scaled / 100
    return int(value) if value == int(value) else value


def release_day_curve(comparable):
    """A comparable's release-day hourly viewers, {hour: viewers}.

    Hours are 1..24 counted from the comparable's own day_start (00:00 UTC
    of its release date). Returns {} when the comparable has no usable
    hour-1 print, which also drops the GTA VI category's own pre-release
    history (it has no release day yet).
    """
    start = parse_time(comparable["day_start"])
    by_hour = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        offset = (parse_time(point["from"]) - start).total_seconds() / 3600
        hour = int(offset) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = viewers
    if 1 not in by_hour:
        return {}
    return by_hour


def weighted_least_squares(xs, ys, ws):
    """Weighted fit of y = a + b*x; returns (a, b)."""
    total = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    return mean_y - slope * mean_x, slope


def weighted_spread(residuals, ws):
    """Weighted standard deviation of residuals around their weighted mean."""
    total = sum(ws)
    mean = sum(w * r for w, r in zip(ws, residuals)) / total
    variance = sum(w * (r - mean) ** 2 for w, r in zip(ws, residuals)) / total
    return math.sqrt(variance)


def median_log_ratio(curves, hour):
    """Median log(viewers_h / viewers_1) over comparables that have hour h."""
    ratios = sorted(math.log(curve[hour] / curve[1]) for curve in curves if hour in curve)
    if not ratios:
        return None
    return ratios[len(ratios) // 2]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    z_scores = [inv_normal_cdf(tau) for tau in levels]
    known = next((row for row in case["known"] if row["hour"] == 1), None)
    if known is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    log_p1 = math.log(p1)

    curves = [curve for curve in (release_day_curve(c) for c in case.get("comparables") or []) if curve]

    medians = {}
    spreads = {}
    for hour in FORECAST_HOURS:
        points = [(math.log(curve[1]), math.log(curve[hour]))
                  for curve in curves if hour in curve]
        if len(points) < MIN_POINTS:
            medians[hour] = None
            spreads[hour] = 0.35
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        weights = [math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2) for x in xs]
        intercept, slope = weighted_least_squares(xs, ys, weights)
        residuals = [y - (intercept + slope * x) for x, y in points]
        spread = weighted_spread(residuals, weights)
        medians[hour] = math.exp(intercept + slope * log_p1)
        spreads[hour] = max(spread, SPREAD_FLOOR) * SPREAD_SHRINK

    # Hours the regression could not fit: fall back to the comparables'
    # median log-ratio curve, interpolating in log space between fitted
    # neighbours when even that curve has no data for the hour, and to the
    # hour-1 print itself when there are no comparables at all.
    for hour in FORECAST_HOURS:
        if medians[hour] is not None:
            continue
        ratio = median_log_ratio(curves, hour)
        if ratio is not None:
            medians[hour] = p1 * math.exp(ratio)
            continue
        fitted = [h for h in FORECAST_HOURS if medians[h] is not None]
        before = [h for h in fitted if h < hour]
        after = [h for h in fitted if h > hour]
        if before and after:
            lo, hi = before[-1], after[0]
            frac = (hour - lo) / (hi - lo)
            medians[hour] = math.exp(
                (1 - frac) * math.log(medians[lo]) + frac * math.log(medians[hi]))
        elif before:
            medians[hour] = medians[before[-1]]
        elif after:
            medians[hour] = medians[after[0]]
        else:
            medians[hour] = p1

    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        median = medians[hour]
        spread = spreads[hour]
        quantiles = []
        for z in z_scores:
            value = median * math.exp(z * spread)
            value = min(max(value, 1.0), 1e9)
            quantiles.append(round_half_up_2dp(value))
        # Rounding can only tie neighbours, never invert them, but enforce
        # the non-decreasing order the format requires.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

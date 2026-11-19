"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Model (per case, standard library only, fully deterministic, no randomness):
  A trophy's earn rate is earners / players-tracked, and the two move
  differently, so they are forecast separately and divided at the end.

  1. Players tracked N(t): fit ln N against ln(1 + hours since release)
     by least squares (a power law in time since release, the usual
     shape of an adoption curve), then read the line at the target.
     The slope is clamped to [0, 2.5] and the forecast to
     [N_last, 50 * N_last] so a noisy fit cannot explode. Where the
     window holds at least 4 distinct time values, a quadratic in
     ln(1 + hours) is fitted instead: the growth curve is curved in
     this axis (its local slope rises through the week), and the
     straight line's average slope understates the growth still to
     come. Earner fits use the fitted curve's local slope at the
     target, shrunk toward the pooled slope, and grow from the fitted
     value at the last read.
  2. Earners E_i(t) per trophy: fit ln(E_i + 0.5) against the same time
     axis, where E_i = rate_i * N / 100 from each read. Each trophy's
     slope is shrunk toward the pooled (median) slope, with the weight
     growing in how many distinct earner counts the trophy actually
     showed -- small-count trophies borrow the crowd's growth, trophies
     with a real trend of their own keep it. Story/chapter trophies
     grow fast, platinum and grind trophies slowly; the fit learns
     that from the history instead of assuming it.
     Both fits use only the last 24 hours of reads: launch night is a
     different regime (the page sat unrefreshed for hours, then
     jumped) and fitting over the whole history drags the slopes off.
     Each series falls back to the whole history when its recent
     window holds too few distinct values to identify a slope.
  3. Median rate m_i = 100 * E_i(target) / N(target). Over a week the
     player base grows faster than most trophies are earned, so most
     medians fall below the last read (dilution); easy early trophies
     whose earners outrun the player base rise.
  4. Spread: a logit-normal band around the median whose sigma combines
     a random-walk term (0.5 * sqrt(days), the house baseline's), the
     fits' residual scatter scaled by how far the target extrapolates,
     and a refresh-shock term (Exophase refreshes on its own schedule
     and the answer is the page as of its last refresh, so the effective
     horizon is uncertain). The band is asymmetric -- wider below the
     median than above it, because a refresh can halve rates overnight
     while the upside is bounded by play time and by 100%.
  5. Flat history (no read ever changed, as on launch night): no trend
     can be fitted. If the case is cut within 48 h of release, the kit
     documents that Exophase had not refreshed since before launch and
     its first refresh took tracked players 638 -> 3,868, so the
     forecast assumes one such refresh before the target, with the new
     joiners earning at 15% of the early cohort's rate (the refresh
     lowered every rate but the 100% one, so the newcomers are far less
     progressed than the launch-night cohort). Otherwise a slow
     20%/day player growth with 10%/day earner growth is assumed.
     Either way the spread is widened a lot (sigma_shock 1.6): a
     pending refresh is the dominant uncertainty.
  6. A trophy at 100% in every read stays at a median of 100 (the
     refresh left it at 100%), with a fixed honest downside tail.

Quantiles are logit(m) + z_tau * sigma (down/up asymmetric), rounded
to 4 decimals half-up like the house baseline, clamped to [0.01, 99.99]
and forced non-decreasing after rounding.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (see the module docstring for why each exists) ---
SIGMA_RW = 0.5            # random-walk spread, logit units per sqrt(day)
SIGMA_SHOCK_RICH = 0.6    # refresh / stale-read uncertainty, rich history
SIGMA_SHOCK_FLAT = 1.6    # same, when a first refresh is still pending
SIGMA_DOWN = 1.25         # downside spread is wider than upside
SIGMA_UP = 0.85
FIRST_REFRESH_MULT = 6.0  # kit documents first refresh: 638 -> 3,868 players
NEW_COHORT_EARN_FRAC = 0.15  # newcomers earn at 15% of the early cohort's rate
FLAT_RECENT_HOURS = 48.0
FIT_WINDOW_HOURS = 24.0   # fit trends on the recent regime only (see below)
N_GROWTH_CAP = 50.0


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std) or None if x is constant."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return a, b, math.sqrt(var)


def ols_quad(xs, ys):
    """Least squares y ~ a + b*x + c*x^2. Returns (a, b, c, resid_std) or None."""
    n = len(xs)
    if n < 4:
        return None
    # Normal equations for [1, x, x^2]; solve 3x3 by Gaussian elimination.
    s = [0.0] * 5  # sums of x^0..x^4
    t = [0.0] * 3  # sums of y*x^0..x^2
    for x, y in zip(xs, ys):
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                t[k] += p * y
            p *= x
    a = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(3):
            if r != col:
                f = a[r][col] / a[col][col]
                for k in range(col, 4):
                    a[r][k] -= f * a[col][k]
    coef = [a[i][3] / a[i][i] for i in range(3)]
    resid = [y - (coef[0] + coef[1] * x + coef[2] * x * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 3)
    return coef[0], coef[1], coef[2], math.sqrt(var)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    days = case["horizon_minutes"] / 1440.0

    release_raw = case.get("game", {}).get("release_date")
    if release_raw:
        release = datetime.fromisoformat(release_raw).replace(tzinfo=timezone.utc)
    elif history:
        release = parse_time(history[0]["at"])
    else:
        release = parse_time(case["at"])
    target = parse_time(case["target"])
    h_target = max(0.0, (target - release).total_seconds() / 3600.0)
    x_target = math.log1p(h_target)
    hours_since_release = max(0.0, (parse_time(case["at"]) - release).total_seconds() / 3600.0)

    if not history:
        # No reads at all: a valid, very wide forecast beats a crash.
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.1, 1.0, 10.0, 90.0]} for t in trophies]}

    # Per-read (hours since release, players tracked, rates).
    reads = []
    for h in history:
        hh = max(0.0, (parse_time(h["at"]) - release).total_seconds() / 3600.0)
        reads.append((hh, float(h["players_tracked"]), h["rates"]))
    xs = [math.log1p(hh) for hh, _, _ in reads]
    n_last = reads[-1][1]
    # Last non-null rate per trophy.
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for _, _, rates in reversed(reads):
            if rates[i] is not None:
                r = float(rates[i])
                break
        last_rate.append(0.0 if r is None else r)

    flat = len({n for _, n, _ in reads}) < 2 or len(reads) < 3

    if flat:
        # No trend to fit: assume the pending first refresh (recent launch)
        # or slow growth (older flat case), and widen the spread a lot.
        if hours_since_release < FLAT_RECENT_HOURS:
            n_target = n_last * FIRST_REFRESH_MULT
            earners = []
            for i in range(len(trophies)):
                e_last = last_rate[i] * n_last / 100.0
                e_new = NEW_COHORT_EARN_FRAC * (last_rate[i] / 100.0) * (n_target - n_last)
                earners.append(e_last + e_new)
        else:
            n_target = n_last * (1.2 ** days)
            earners = [last_rate[i] * n_last / 100.0 * (1.1 ** days) for i in range(len(trophies))]
        sigma_shock = SIGMA_SHOCK_FLAT
        sigma_fit = 0.0
        medians = [100.0 * e / n_target for e in earners]
    else:
        # --- trends are fitted on the recent regime only ---
        # Launch night is a different regime (the page sat unrefreshed
        # for hours, then jumped), and a single power law fitted over
        # the whole history is dragged off by it: on synthetic
        # final-like data the all-reads fit overstates earner growth
        # by up to ~30% at the target, the last-24h fit by ~2-5%.
        # So each series is fitted on the last FIT_WINDOW_HOURS of
        # reads, falling back to the whole history when the window
        # holds too few distinct values to identify a slope (slow
        # refresh cadence), and to the pooled slope when a trophy
        # barely moved at all.
        at_ts = parse_time(case["at"]).timestamp()
        win_start = at_ts - FIT_WINDOW_HOURS * 3600.0

        def fit_series(pairs):
            """pairs: [(x, y, read_ts)]. Prefer the recent window; fall back to
            longer windows, then all reads, until one holds enough distinct
            values to identify a slope (slow refresh cadence)."""
            starts = [win_start, at_ts - 48 * 3600.0, at_ts - 96 * 3600.0, float("-inf")]
            for start in starts:
                subset = [(x, y) for x, y, ts in pairs if ts >= start]
                if len(subset) >= 3 and len({round(y, 6) for _, y in subset}) >= 3:
                    xs_s = [x for x, _ in subset]
                    ys_s = [y for _, y in subset]
                    # A quadratic needs at least 4 distinct x values or it
                    # just interpolates the refresh steps and extrapolates wild.
                    if len({round(x, 6) for x in xs_s}) >= 4:
                        q = ols_quad(xs_s, ys_s)
                        if q is not None:
                            return q  # (a, b, c, resid_std)
                    fit = ols(xs_s, ys_s)
                    if fit is not None:
                        return fit
            return None

        read_ts = [parse_time(h["at"]).timestamp() for h in history]

        # --- players tracked: power law in time since release ---
        fit_n = fit_series([(x, math.log(n), ts) for x, (_, n, _), ts in zip(xs, reads, read_ts)])
        if fit_n is None:
            n_target = n_last
        elif len(fit_n) == 4:
            a_n, b_n, c_n, _ = fit_n
            n_target = math.exp(a_n + b_n * x_target + c_n * x_target * x_target)
        else:
            a_n, b_n, _ = fit_n
            b_n = min(2.5, max(0.0, b_n))
            n_target = math.exp(a_n + b_n * x_target)
        n_target = min(N_GROWTH_CAP * n_last, max(n_last, n_target))

        # --- earners per trophy, slopes shrunk toward the pooled slope ---
        slopes, resids = [], []
        per_trophy = []
        for i in range(len(trophies)):
            pairs = []
            for (hh, n, rates), x, ts in zip(reads, xs, read_ts):
                r = rates[i]
                if r is None:
                    continue
                pairs.append((x, math.log(float(r) * n / 100.0 + 0.5), ts))
            fit = fit_series(pairs) if len(pairs) >= 3 else None
            if fit is None:
                per_trophy.append(None)
                continue
            distinct_e = len({round(y, 1) for _, y, _ in pairs})
            if len(fit) == 4:
                a_i, b_i, c_i, sd_i = fit
                # Local slope of the fitted curve at the target (for pooling).
                slope_target = b_i + 2.0 * c_i * x_target
                per_trophy.append((a_i, b_i, c_i, distinct_e))
                slopes.append(slope_target)
            else:
                a_i, b_i, sd_i = fit
                per_trophy.append((a_i, b_i, 0.0, distinct_e))
                slopes.append(b_i)
            resids.append(sd_i)
        pooled_slope = median(slopes) if slopes else 0.0
        pooled_slope = min(2.0, max(-0.5, pooled_slope))
        pooled_resid = median(resids) if resids else 0.0

        medians = []
        for i in range(len(trophies)):
            entry = per_trophy[i]
            if entry is None:
                b_i = pooled_slope
                a_i = math.log(last_rate[i] * n_last / 100.0 + 0.5) - b_i * xs[-1]
                e_target = math.exp(a_i + b_i * x_target)
            else:
                a_i, b_i, c_i, distinct_e = entry
                # Fitted log-earners at the last read, then grow from there
                # at the shrunk slope (the fitted curve's local slope at
                # the target, shrunk toward the pooled slope).
                slope_target = b_i + 2.0 * c_i * x_target
                w = distinct_e / (distinct_e + 4.0)
                b_shrunk = w * slope_target + (1.0 - w) * pooled_slope
                b_shrunk = min(3.0, max(-0.5, b_shrunk))
                y_last = a_i + b_i * xs[-1] + c_i * xs[-1] * xs[-1]
                e_target = math.exp(y_last + b_shrunk * (x_target - xs[-1]))
            medians.append(100.0 * e_target / n_target)
        sigma_shock = SIGMA_SHOCK_RICH
        sigma_fit = min(1.0, pooled_resid * (1.0 + math.log1p(days)))

    sigma = math.sqrt((SIGMA_RW * math.sqrt(days)) ** 2 + sigma_fit ** 2 + sigma_shock ** 2)
    sigma = min(2.5, max(0.35, sigma))
    sig_down, sig_up = sigma * SIGMA_DOWN, sigma * SIGMA_UP

    rows = []
    for i, trophy in enumerate(trophies):
        # A trophy at 100% in every read: the refresh left it at 100%.
        if last_rate[i] >= 100.0 and all(
            rates[i] is None or float(rates[i]) >= 100.0 for _, _, rates in reads
        ):
            rows.append({"id": trophy["id"], "q": [95, 99.5, 100, 100, 100]})
            continue
        m = clamp_rate(medians[i])
        lg = logit(m)
        qs = [
            inv_logit(lg + Z[0] * sig_down),
            inv_logit(lg + Z[1] * sig_down),
            m,
            inv_logit(lg + Z[3] * sig_up),
            inv_logit(lg + Z[4] * sig_up),
        ]
        qs = [clamp_rate(q) for q in qs]
        qs = [round4(q) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

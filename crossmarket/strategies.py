"""Each function of the completed morning and prior sessions casts one ballot."""

import numpy as np


def direction(up, down):
    return int(bool(up)) - int(bool(down))


def sign(x):
    return int(np.sign(x))


def ballot(bars, prior_prices, prior_moves, previous_close, training, prior_fades):
    opening, high, low, close, volume = (
        bars[k].to_numpy() for k in ("open", "high", "low", "close", "volume")
    )
    p, first = close[-1], opening[0]
    span = max(high.max() - low.min(), first * 1e-9)
    tick = first * 1e-9
    ret = np.diff(np.r_[first, close])
    move = p - first
    eff = move / max(sum(abs(ret)), tick)
    loc = (p - low.min()) / span
    third = np.array_split(ret, 3)
    legs = [sum(x) for x in third]
    med = np.median(high - low)
    last = close[-1] - opening[-1]
    hi, lo = max(high[:5]), min(low[:5])
    unit = (close - low.min()) / span
    activity, edges = np.histogram(unit[:20], bins=np.linspace(0, 1, 9))
    # Fixed bin edges use only information available at the 10:00 decision.
    centre = (edges[activity.argmax()] + edges[activity.argmax() + 1]) / 2
    gains = sum(np.maximum(ret[-14:], 0))
    losses = sum(np.maximum(-ret[-14:], 0))
    rsi = gains / max(gains + losses, tick)
    mid = np.mean(close[:20])
    sd = max(np.std(close[:20]), tick)
    av = bars.close.ewm(span=8, adjust=False).mean().to_numpy()
    slow = bars.close.ewm(span=21, adjust=False).mean().iloc[-1]
    co = np.correlate(ret[:-1] - ret[:-1].mean(), ret[1:] - ret[1:].mean())[0]
    trend = sign(move)
    gap = first - previous_close if previous_close is not None else None
    b = {
        "Opening range escape": direction(p > hi and loc > 0.75, p < lo and loc < 0.25),
        "Failed auction": direction(
            min(low[5:]) < lo and p > lo and legs[-1] > 0,
            max(high[5:]) > hi and p < hi and legs[-1] < 0,
        ),
        "Efficient trend": sign(eff) if abs(eff) > 0.4 else 0,
        "Three wave persistence": direction(min(legs) > 0, max(legs) < 0),
        "EMA pullback": direction(
            av[-1] > slow and low[-3:].min() < av[-3] < p,
            av[-1] < slow and high[-3:].max() > av[-3] > p,
        ),
        "Inside range release": direction(
            high[-2] <= high[-3] and low[-2] >= low[-3] and p > high[-3],
            high[-2] <= high[-3] and low[-2] >= low[-3] and p < low[-3],
        ),
        "Range contraction breakout": (
            direction(p > max(high[-6:-1]), p < min(low[-6:-1]))
            if np.mean(high[-10:-1] - low[-10:-1]) < med
            else 0
        ),
        "Shock retracement": -sign(last) if abs(last) > 2 * max(med, tick) else 0,
        "Wick rejection": direction(
            min(opening[-1], p) - low[-1] > 0.6 * (high[-1] - low[-1]),
            high[-1] - max(opening[-1], p) > 0.6 * (high[-1] - low[-1]),
        ),
        "RSI exhaustion": direction(rsi < 0.25 and last > 0, rsi > 0.75 and last < 0),
        "Band re-entry": direction(
            close[-2] < mid - 2 * sd <= p, close[-2] > mid + 2 * sd >= p
        ),
        "Median cross": direction(
            close[-2] <= np.median(close[:20]) < p,
            close[-2] >= np.median(close[:20]) > p,
        ),
        "Range midpoint retest": direction(
            legs[0] > 0 and min(low[-5:]) <= (hi + lo) / 2 < p and last > 0,
            legs[0] < 0 and max(high[-5:]) >= (hi + lo) / 2 > p and last < 0,
        ),
        "Profile migration": direction(
            unit[-1] > centre + 0.25 and np.median(unit[-10:]) > centre,
            unit[-1] < centre - 0.25 and np.median(unit[-10:]) < centre,
        ),
        "Serial dependence": sign(co) * sign(sum(ret[-3:]))
        if abs(co) > 0.2 * sum(ret**2)
        else 0,
        "Acceleration": direction(
            0 < legs[0] < legs[1] < legs[2], 0 > legs[0] > legs[1] > legs[2]
        ),
        "Volume confirmed impulse": None,
        "VWAP stretch fade": None,
        "Signed flow divergence": None,
        "Typical move breakout": None,
        "Trend with historical context": None,
        "Gap absorption": None,
        "Nearest morning analogues": None,
    }
    if np.isfinite(volume).all() and min(volume) >= 0 and sum(volume) > 0:
        vw = np.average((high + low + close) / 3, weights=volume)
        b["VWAP stretch fade"] = (
            -sign(p - vw) if abs(p - vw) > 0.35 * span and abs(eff) < 0.4 else 0
        )
        b["Volume confirmed impulse"] = (
            trend
            if sum(volume[-5:]) > 1.3 * sum(volume[:25]) / 5 and trend * (p - vw) > 0
            else 0
        )
        imbalance = np.dot(np.sign(ret), volume) / sum(volume)
        b["Signed flow divergence"] = (
            sign(imbalance) if imbalance * trend < -0.15 else 0
        )
    if len(prior_moves) == 14 and np.isfinite(prior_moves).all():
        b["Typical move breakout"] = (
            trend if abs(move / first) > np.mean(prior_moves) and abs(eff) > 0.2 else 0
        )
    pct60 = None
    if len(prior_prices) == 60 and np.isfinite(prior_prices).all():
        pct60 = 100 * (p / np.mean(prior_prices) - 1)
        b["Trend with historical context"] = (
            trend
            if trend * pct60 > 0 and trend * legs[-1] > 0 and abs(eff) > 0.2
            else 0
        )
    if gap is not None and len(prior_fades) == 5 and np.isfinite(prior_fades).all():
        b["Gap absorption"] = (
            trend
            if sum(prior_fades) >= 3
            and trend * gap < 0
            and abs(move) > abs(gap) / 2
            and abs(eff) > 0.2
            else 0
        )
    vector = np.array([eff, loc, legs[-1] / span, (p - first) / span])
    if len(training) >= 30:
        distances = [sum((x - vector) ** 2) for x, _ in training]
        labels = [training[i][1] for i in np.argsort(distances, kind="stable")[:7]]
        b["Nearest morning analogues"] = direction(sum(labels) >= 5, sum(labels) <= 2)
    return b, vector, pct60

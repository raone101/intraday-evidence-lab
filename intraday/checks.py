"""Separately derive the ballots with shifted rolling windows and match fills to backtesting.py."""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from .experiment import P, FEATURES


def independent_votes(bars, rows):
    local = bars.between_time("09:30", "09:59").copy()
    g = local.groupby(local.index.strftime("%Y-%m-%d"))
    first = local.between_time("09:30", "09:34")
    f = first.groupby(first.index.strftime("%Y-%m-%d"))
    open_ = g.open.first()
    end = g.close.last()
    vol = f.volume.sum()
    valid = (g.size() == 30) & (f.size() == 5) & (vol > 0) & (g.volume.sum() > 0)
    dates = valid[valid & valid.index.isin(rows.index)].index
    obs = pd.DataFrame(index=dates)
    ret = (end / open_ - 1).loc[dates]
    vol = vol.loc[dates]
    window = P["lookback_sessions"]
    rvol = vol / vol.shift(1).rolling(window).mean()
    sigma = ret.abs().shift(1).rolling(window).mean()
    dollar = (local.high + local.low + local.close) * local.volume / 3
    weighted = dollar.groupby(local.index.strftime("%Y-%m-%d")).sum() / g.volume.sum()
    obs["opening5_return"] = (f.close.last() / open_ - 1).loc[dates]
    obs["first30_return"] = ret
    obs["opening_relative_volume"] = rvol
    obs["above_range"] = (end / f.high.max() - 1).loc[dates]
    obs["above_band"] = end.loc[dates] / (open_.loc[dates] * (1 + sigma)) - 1
    obs["above_vwap"] = (end / weighted - 1).loc[dates]
    obs = obs.dropna()
    regular = rows.loc[rows.eligible]
    if list(obs.index) != list(regular.index):
        raise AssertionError("Independent eligible dates disagree.")
    np.testing.assert_allclose(obs[FEATURES], regular[FEATURES], atol=1e-12, rtol=1e-10)
    a = (
        (f.close.last().loc[obs.index] > open_.loc[obs.index])
        & (rvol.loc[obs.index] >= P["rvol_min"])
        & (obs.above_range > 0)
    )
    b = (obs.above_band > 0) & (obs.above_vwap > 0)
    np.testing.assert_array_equal(a, regular.breakout)
    np.testing.assert_array_equal(b, regular.band)
    return len(obs)


class Schedule(Strategy):
    def init(self):
        pass

    def next(self):
        event = self.data.Instruction[-1]
        if event == 1:
            self.buy(size=P["shares"])
        elif event == -1 and self.position:
            self.position.close()


def external_fills(bars, test, recorded):
    """The external engine consumes only daily decisions and raw OHLC, then fills its own orders."""
    view = (
        bars.loc[test.index[0] : test.index[-1]].between_time("09:30", "15:59").copy()
    )
    view.columns = view.columns.str.title()
    # Orders are staged one observed bar before the scheduled fill. A missing entry expires.
    # Do not silently accept a preceding fill from another session.
    planned = []
    for date, row in test.iterrows():
        day = view.loc[date:date]
        if row.entry_at not in day.index:
            continue
        exits = day.index[
            (day.index >= row.planned_exit)
            & (day.index < row.planned_exit + pd.Timedelta(minutes=10))
        ]
        if len(exits) == 0:
            if row.eligible:
                raise AssertionError(f"Missing exit on {date}")
            continue
        e, x = view.index.get_loc(row.entry_at), view.index.get_loc(exits[0])
        if e < 1 or x <= e or view.index[e - 1].date() != row.entry_at.date():
            raise AssertionError("Unexpected event order.")
        planned.append((date, e, x))
    totals = {}
    for version in P["variants"]:
        tape = view.copy()
        schedule = np.zeros(len(tape), dtype=np.int8)
        for date, e, x in planned:
            if test.at[date, version]:
                schedule[e - 1] = 1
                schedule[x - 1] = -1
        tape["Instruction"] = schedule
        stats = Backtest(
            tape,
            Schedule,
            cash=P["initial_usd"],
            commission=P["primary_cost_bps_each_side"] / 10000,
            trade_on_close=False,
            exclusive_orders=True,
        ).run()
        got = stats["_trades"]
        expected = recorded.loc[recorded.version == version]
        if len(got) != len(expected):
            raise AssertionError(f"{version}: {len(got)} trades vs {len(expected)}")
        if len(got):
            np.testing.assert_array_equal(
                got.Size.to_numpy(), np.full(len(got), P["shares"])
            )
            if list(got.EntryTime) != list(expected.entry_at) or list(
                got.ExitTime
            ) != list(expected.exit_at):
                raise AssertionError(f"{version}: fill timestamps differ")
            for field, column in [
                ("EntryPrice", "entry"),
                ("ExitPrice", "exit"),
                ("Commission", "cost_usd"),
                ("PnL", "net_usd"),
            ]:
                np.testing.assert_allclose(
                    got[field], expected[column], rtol=0, atol=1e-8
                )
        np.testing.assert_allclose(
            stats["Equity Final [$]"],
            P["initial_usd"] + expected.net_usd.sum(),
            atol=1e-7,
            rtol=0,
        )
        totals[version] = len(got)
    return totals

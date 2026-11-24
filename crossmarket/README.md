# Cross-market strategy arena

23 individually scored strategies read the same completed 30 minute window for each instrument. Select a symbol and date in the [offline replay](replay.html), read the original votes and reveal the outcome. Download the HTML and open it locally in a modern browser.

| Strategy family | Ballots and underlying hypotheses |
| --- | --- |
| Breakouts | Opening range escape trades beyond the first five-minute range. Inside range release follows a break of an enclosing candle. Range contraction breakout looks for a break after a quiet period. Typical move breakout requires the opening return to exceed the previous 14-session average absolute opening return. |
| Trend | Efficient trend measures net movement against the total path travelled. Three wave persistence checks three consecutive ten-minute changes. Acceleration requires successively stronger changes. EMA pullback looks for a retest in an established trend. Trend with historical context requires an efficient opening move, a confirming final ten minutes and agreement with the percentage distance from the previous 60 sessions’ 09:59 average. |
| Reversals | Failed auction fades an opening boundary breach after price returns inside. Shock retracement fades an unusually large final candle. Wick rejection follows a rejection at one extreme. RSI exhaustion requires an extreme gain/loss balance and a reversal. Band re-entry trades a return inside a two-standard-deviation band. Gap absorption requires the morning to retrace over half the overnight gap and at least three of the previous five sessions to have reversed the overnight direction during the day. |
| Price location | Median cross tests a crossing of the initial 20-minute median. Range midpoint retest tests a return to the opening midpoint followed by renewed movement. Profile migration compares recent closes with the most visited early closing-price bin. |
| Flow | Volume confirmed impulse requires a late volume surge and agreement with the volume-weighted price. VWAP stretch fade looks for a stretched price on an inefficient path. Signed flow divergence checks disagreement between the opening move and volume signed by minute price changes. Quote-only feeds abstain from these three. |
| Statistical | Serial dependence follows or fades recent movement according to measured one-minute autocovariance. Nearest morning analogues is a small machine-learning model. It finds seven similar earlier openings in the last 120 resolved examples, needs at least 30 examples, and votes only when five neighbours agree on direction. |

The opening breakout and expected-move hypotheses adapt ideas from Zarattini, Barbon and Aziz. The price-profile vote takes inspiration from Hung, Chen and Yu, using closing-price occupancy. EMA, RSI and band examples occur in the previously reviewed LEAN, NautilusTrader, Freqtrade and bbgo code. The thresholds, clock, combinations and analogue model here are project choices. None of the source papers validates this combined roster. [Paper register](../research/papers.md) · [Activity and AI studies](../research/papers-v2.md) · [Code-level repository review](../research/repositories.md)

## Execution and feedback

One ballot per symbol and date, using 09:30 through 09:59 New York minute bars. The hypothetical entry is the 10:01 open and exit the 11:00 open. This is a common US-morning comparison across markets. A London-open FX test or an Asian-session crypto test would be a different experiment. There is no across-symbol stock picker or portfolio allocation. A missing opening minute prevents a ballot. Missing execution prices or crossed opening bid/ask quotes leave the issued ballot unresolved. No forward fills.

Each instrument starts with equal weights. At least two directional forecasts are required. More signed weight wins and exact ties stay flat. Spot crypto and option contracts use long or cash execution. Bearish votes still count against buying. The ledger preserves both the forecast and the executable position. A bearish put forecast means the put premium is expected to fall. Winning and losing executable calls change only subsequent weights, using the existing [bounded voting formula](../intraday/voting.py). Unavailable, flat and unresolved calls earn no feedback. The comparison columns report equal voting and always buying the same interval. Cash earns zero.

Basis points use the entry midpoint or trade open as denominator. One basis point is 0.01%. Both legs incur scenario costs per side of 5 bp for shares, ETFs and options, 1 bp for futures, 0.2 bp for FX and 10 bp for crypto. FX and options also cross the recorded bid/ask spread. Other markets use minute trade opens plus assumed friction. These are underlying or premium returns per unit of entry exposure. They omit leverage, futures margin, option contract commissions, market impact, stock borrow availability and portfolio compounding.

## Data and outcomes

| Source | Instruments | Observations |
| --- | --- | --- |
| FirstRate Data rolling minute samples | TSLA, NVDA, AAPL, AMZN, META, MSFT, NFLX, F, QQQ, ES, NQ, GC | 9 to 20 November 2026, ten US mornings each. Futures use the vendor’s continuous front-month prices. Ford adds a lower-priced equity. There is no news or earnings feed in this test. |
| Pinned SPY minute archive from the existing research | SPY | 263 scored US sessions from July 2024 to July 2025. Older bars initialise features and analogue training. Voting weights start fresh in the scored period. |
| FXCM minute bid/ask files | EUR/USD, GBP/USD, USD/JPY | Four weeks in June and July 2020, 60 ballots, 54 resolved. Six had unusable execution quotes. |
| Binance spot monthly minute files | BTC/USDT, ETH/USDT | 91 consecutive dates per asset, April to June 2025, including weekends. |
| QuantConnect LEAN option quotes | AAPL 21 June 2014 expiry, $650 call and put | One session per contract, 6 June 2014. Contract selection is fixed for this historical example. This checks premium and spread accounting only. |

All 23 strategies ran in one experiment. The 627 symbol-sessions generated 14,421 original calls including abstentions. The weighted vote took 475 positions in 621 resolved symbol-sessions. On SPY its average net outcome was −10.24 bp per session, compared with −8.57 bp for equal votes. Opening range escape and profile migration had a 0.74 correlation between their signed SPY positions. More votes can count the same price move twice. The experiments share observations, thresholds are exploratory and the longest archive has already been used for research. None of these figures establishes out-of-sample performance on new markets.

[Per-instrument coverage, both votes and baseline](coverage.csv) · [Every strategy by instrument](strategies.csv) · [Session results](sessions.csv) · [Original calls, weights and counterfactual returns](votes.csv.gz) · [Exact source URLs and checksums](sources.json)

## Running the study

From the repository root, install the existing requirements, then run:

```sh
python -m crossmarket.run --cache data/markets --out runs/crossmarket
python -m crossmarket.replay runs/crossmarket
python -m unittest discover -s tests
```

The downloader verifies every raw input. FirstRate rotates its public samples. Retaining the checksum-matching downloads preserves a byte-for-byte rerun after rotation. The published vote ledger can still be inspected without downloading raw data.

The existing scheduled SPY paper runner also attempts the expanded ballot. It freezes it in a dated crossmarket.json after the opening window and before 10:01 New York, and writes per-strategy outcomes into close.json. A late morning run skips this ballot. Forward voting learns only from previously settled forward ballots. The multi-market archive is a historical replay. Other symbols have no scheduled forward feeds in this version.

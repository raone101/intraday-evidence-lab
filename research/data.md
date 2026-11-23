# Data and execution

| Archive | Coverage and use | Provenance |
|---|---|---|
| Primary SPY one-minute CSV | 640,753 bars from 2 January 2019 through 18 July 2025. Sources are dated, not live. All 263 scored sessions had complete morning features and scheduled entry and exit prices. | [Pinned source repository](https://github.com/Ascensao/Intraday-momentum-strategy/tree/5a9e956994d1ca9ef16e5e0ae6af4ce4e04e87eb) · [Download and SHA-256](../intraday/data.py) |
| Comparison SPY one-minute CSV | 4 January 2021 to 30 December 2024. The overlapping, warmed-up dates give 989 comparable ballots. This archive does not cover the full scored period. | [Pinned source repository](https://github.com/noiiiise/SPY-Intraday-Momentum/tree/d5af5981fd66477b5791a445b97077bcd8d07a01) · [Checksums and disagreements](../results/audit.json) |

The primary repository identifies Polygon as its source. These are third-party extracts with different nominal price levels on some dates. The exact adjustment basis and independence of their original vendors are unverified. Within-day returns reduce sensitivity to price-level adjustments. The cash P&L still scales with the archive’s reported share price. No raw market archive is redistributed here.

Index each row by its **bar start**. Interpret the primary archive’s unzoned timestamps in America/New_York, and convert offset-aware timestamps from the comparison archive into the same zone. Use the XNYS calendar for market open and close, including holidays, daylight saving and shortened sessions. Reject duplicate, unsorted, non-finite, non-positive-price or inconsistent OHLC rows. A complete opening is 30 one-minute bars with positive aggregate volume in both the five- and 30-minute windows.

Price and volume come from completed bars. The opening VWAP proxy weights each minute’s typical price (high + low + close divided by three) by its volume. The 10:05 entry uses the reported minute open. The exit uses the minute open ten minutes before the published close, or the first available minute thereafter within that session. A missing scheduled entry expires. One simulated share is affordable from the illustrative $10,000 starting balance throughout the test.

A cost of five basis points is 0.05% of traded value on **each** side. This is a scenario for combined frictions. Historical bars do not contain executable quotes, queue position, actual spread, brokerage fees, taxes or a measured fill delay. Cash earns zero. The experiment has no overnight or leveraged position. Closing drawdown is computed from end-of-day portfolio values and can understate peak intraday losses. The clock and cash benchmarks have the same experimental setting. A long-term investment benchmark would answer a different opportunity-cost question.

The second archive changed zero breakout votes and one band vote on 989 shared eligible sessions. The median absolute difference in the unfiltered intraday return was 0.067 basis points and the 95th percentile was 0.184 basis points. Cross-archive agreement checks implementation and data sensitivity. The same instrument over overlapping dates does not supply a new market sample.

## Forward input

The optional Alpaca adapter calls only the historical stock-bars endpoint. It requests up to 400 calendar days initially, then overlapping new completed bars, and pins the chosen IEX or SIP feed to a separate cache and ledger namespace. An IEX-only feed has different volume coverage from consolidated data. Fresh feed results form a new trial. Keep keys in local environment variables. Confirm the selected feed’s timeliness with the provider. [Alpaca bars](https://docs.alpaca.markets/us/reference/stockbars) · [Market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)

The CSV alternative expects chronological completed SPY bars with columns `caldt,open,high,low,close,volume`. A producer should replace the file atomically and include the previous trading session and sufficient older history. The price filter needs 14 valid openings and the model needs at least 60 resolved training sessions with both outcomes before the start of the current month. A late start cannot create an earlier decision. Replay and forward observations occupy different SQLite namespaces. Gaps remain visible and simulation never routes broker orders.

# When intraday strategies disagree

Can two agents examining the same opening tape reach opposite, defensible conclusions? This experiment holds the asset, information cutoff, entry, exit and costs fixed. Every specialist keeps its own ledger. The six two-bot combinations and the net-vote rule each get a separate virtual share. They never veto a specialist's decision. Use the replay to pick a side before seeing the outcome.

| Specialist | Rule and source of inspiration |
| --- | --- |
| Breakout | Trade a close outside the first five-minute range when first-five volume is at least its previous 14 valid openings’ mean. This is a two-sided adaptation of the original [opening-range research](../research/papers.md). |
| Failed break | If price crosses one end of that range then closes back inside, trade away from the crossed boundary. Abstain on a double sweep. This is a new competing hypothesis designed for the lab. |
| VWAP fade | Trade towards the opening volume-weighted average when the 09:59 close is at least 1.5 volume-weighted standard deviations away. Both the mean and dispersion use the typical minute price. This challenges the original momentum interpretation of VWAP. |
| Profile model | Fit a ridge regression (penalty 100) monthly using prior completed session outcomes and standardised opening features, including [time-at-price profile features](../research/papers-v2.md). Trade only when its predicted gross return exceeds 10 basis points in magnitude. This is the lab’s statistical adaptation. |

Signals use bars through 09:59 New York. Entry is the 10:05 bar open and exit is the first available bar open from ten minutes before the exchange closes. One share per book, five basis points of friction per side, no borrowing costs. The profile model needs 126 resolved sessions. Fit dates, votes, evidence and execution prices appear in the journal. The forecast target is the same intraday holding period used to score the rules.

All 263 sessions in the existing July 2024 to July 2025 evaluation appear, including flat days. There were opposing votes on 52 sessions. Breakout and failed-break rules can never confirm each other because their closing-price conditions exclude each other. The breakout and profile model agreed on only five trades. That is a follow-up observation from a reused period. It cannot establish that the pair generalises. The net-vote book needs a directional lead of two votes.

Run `python -m intraday.arena --data data/primary.csv --out runs/arena` after fetching the archive with the existing setup instructions. This regenerates the journal, comparisons and offline HTML. The four agents comprise three explicit rules and one fitted model. The simulator generates the explanations from the observed inputs. No language model or external AI key is needed for replay.

For future observations the hosted paper runner can write a timestamped `arena.json` before the entry deadline and per-book results in the evening close record. It uses the configured Alpaca IEX feed; its training history and volume coverage differ from the archive. A missing or late capture contributes no forward evidence.

# Forward journal

The first eligible session is 24 November 2026. The registered observation period ends on 28 May 2027. Each dated folder contains observations captured from the Alpaca IEX feed. The historical examples are in a [separate journal](../results/v2/journal.csv).

| File | What it tells you |
| --- | --- |
| `decision.json` | Which variants qualified before 10:05 New York, their opening features, and the actual recording time |
| `momentum60.json` | Percentage above or below the previous 60 same-time prices’ average, daily position and two-session confirmed target. A null target holds the previous position. |
| `forecast.json` | The two frozen probabilities from the registered model comparison, plus the latest training date |
| `close.json` | The simulated return of each participating variant after assumed costs, a check of the flags, and each available forecast's squared probability error |
| `close-gap.json` | A dated attempt with a missing entry or exit. A later close file can resolve the gap |

The close record also carries `carry60`, the marked profit or loss and position of the separate multi-day strategy, including overnight price changes and turnover. The final scheduled session closes any remaining position.

The two forecasts estimate whether a one-share position from 10:05 to 10:20 earns a positive return after costs. The paper rules hold until ten minutes before the exchange closes. The two outcomes answer different questions.

[GitHub Actions](https://github.com/raone101/intraday-evidence-lab/actions/workflows/paper-journal.yml) starts the morning job at 09:38 New York so it can fetch its training data before the opening window finishes at 10:00. Only completed bars through 09:59 enter the decision. Delayed jobs can miss the 10:05 deadline. The second job starts at 16:27 New York and uses that session's exchange close. On-time decisions persist even if forecasts fail. Reruns preserve the first recording. The workflow checks the rules deterministically. The two optional language-model reviewers run locally.

A missing journal date means there is no published observation. Check the run log for a scheduler, credential or feed error. A missing observation contributes neither a hypothetical trade nor a forecast score. The captured forecasts form a subset of the all-eligible-sessions research trial. The original trial can be scored separately on the complete feed history with `intraday.shadow run`.

The feed uses only the Alpaca stock-bars endpoint. Raw bars and secrets stay out of the journal. New data are single-exchange IEX observations. Their opening volume can differ from the historical archive. The training history starts 400 calendar days before the registration date and expands from that fixed start. A one-share virtual position and the original five basis points each side apply throughout.

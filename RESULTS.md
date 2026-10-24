# Results: XRPLarp

Metric: 100 × average precision over all 40,101 test transactions, scored by the organisers.

## Submitted: version 12

- `predictions_v12.csv`: 40,101 rows, one per TRANSACTION_HASH, scores 0–1.
- Method: equal-weight rank average of version 9 and version 11.
- Leaderboard score: [to be added after upload; provisional until organiser verification].

## Leaderboard history

| Version | What was added | Leaderboard |
| --- | --- | --- |
| v1 | Model (a), LightGBM only | 65.5 |
| v2 | + model (b), scheme-shape target | 67.1 |
| v3 | + XGBoost in the average | 67.4 |
| v4 | + link model and rebuilt groups | 70.3 |
| v5 | + cross-round links, per-account groups | 71.2 |
| v6.1 | + soft funding features | 71.9 |
| v6 | + probability-weighted smoothing, night flag | 72.0 |
| v7 | v6 + v6.1 combined | 72.4 |
| v8 | + neighbour and gap-regularity features | 73.1 |
| v9 | + soft cumulative balances, balance trajectories, typed neighbour channels | 73.4 |
| v11 | v8 + second-hop neighbour features | 73.3 |
| **v12** | rank blend of v9 and v11 | pending |

## Local validation (grouped 5-fold CV on train/, 2 repeats)

| Version | Local AP | Easy | Medium | Hard | Top-1% precision |
| --- | --- | --- | --- | --- | --- |
| v5 | 61.5 | 49.6 | 9.4 | 2.5 | 0.817 |
| v6 | 62.3 | 50.4 | 9.5 | 2.4 | 0.824 |
| v8 | 62.8 ± 0.6 | 50.5 | 9.7 | 2.6 | 0.810 |

v9, v11 and the v12 blend were not validated locally; their evidence is the leaderboard.
Link model: 0.99 AUC (within-8-hour links), 0.997 AP (cross-round links) on held-out training scenarios.

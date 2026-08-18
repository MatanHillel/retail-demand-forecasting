# Evaluation Report — Retail Demand Forecasting

*Run:* `20260818T095823Z-fdcd09` · *Generated:* 2026-08-18T09:58:24.060569+00:00 · *Data hash:* `not recorded in this run`

*Configuration:* active-product window k = 6 · seed 42 · train targets
2010-03..2011-05 · validation targets
2011-01..2011-05 · hold-out targets
2011-06..2011-11 · never scored: 2011-12 ·
champion gates: max |bias| 10.0 %, meaningful improvement
2.00 wMAPE points, tie band 1.00
points · default z 1.645 (offered: 1.28, 1.645, 2.05) · σ = 1.4826 x
MAD, fallback product -> abc_group -> global.

*Provenance:* every number below was computed by the pipeline and is traceable to a table under
`artifacts/reports/evaluation_tables/`, `artifacts/forecasts/` or `artifacts/reports/`. Narrative
may be enriched by the LLM agent in LLM mode; the same guard checks both versions (PRD §38).

## Comparison of all candidates

| Model | wMAPE | Bias | MAE | RMSE | n rows | Coverage | Negative share (raw) | Δ wMAPE vs B2 | Relative improvement | ≥ threshold vs B2 | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | 85.1 | 250.3 | 19,968 | 100.0 % | 0.0 % | -1.0 % | -1.8 % | False |  |
| B2_ma3 | 54.7 % | -17.4 % | 83.6 | 251.1 | 19,968 | 100.0 % | 0.0 % | 0.0 % | 0.0 % | False |  |
| B3_seasonal_naive | 89.6 % | 35.0 % | 118.6 | 359.8 | 16,529 | 82.8 % | 0.0 % | -34.8 % | -63.7 % | False | B3 (seasonal naive) has no observed month t-12 for every product; metrics computed only on rows where a seasonal forecast exists (see coverage_share). |
| M1_linear | 56.0 % | -18.1 % | 85.6 | 257.8 | 19,968 | 100.0 % | 13.9 % | -1.3 % | -2.4 % | False |  |
| M2_gbm_poisson | 52.6 % | 0.7 % | 80.3 | 237.0 | 19,968 | 100.0 % | 0.0 % | 2.1 % | 3.9 % | False |  |
| M3_gbm_squared | 54.3 % | 1.5 % | 83.0 | 230.9 | 19,968 | 100.0 % | 0.0 % | 0.4 % | 0.7 % | False |  |
| M4_gbm_absolute | 50.2 % | -25.8 % | 76.6 | 257.1 | 19,968 | 100.0 % | 0.2 % | 4.6 % | 8.4 % | False |  |

B3 (seasonal naive) has no observed month t-12 for every product; its metrics are computed only on
rows where a seasonal forecast exists — see the coverage column above.

## By hold-out month and by ABC group

ABC computed on the training window through 2011-05.

### By hold-out month

| Model | Month | wMAPE | Bias | MAE | RMSE | n rows |
|---|---|---|---|---|---|---|
| B1_last_month | 2011-06 | 61.0 % | 9.0 % | 67.2 | 207.1 | 3,284 |
| B1_last_month | 2011-07 | 54.9 % | 0.3 % | 64.6 | 170.3 | 3,289 |
| B1_last_month | 2011-08 | 59.3 % | -3.1 % | 73.8 | 223.0 | 3,310 |
| B1_last_month | 2011-09 | 57.0 % | -23.4 % | 94.1 | 239.8 | 3,324 |
| B1_last_month | 2011-10 | 55.2 % | -4.9 % | 97.9 | 275.5 | 3,373 |
| B1_last_month | 2011-11 | 51.0 % | -16.4 % | 111.6 | 344.7 | 3,388 |
| B2_ma3 | 2011-06 | 58.8 % | -0.7 % | 64.9 | 183.8 | 3,284 |
| B2_ma3 | 2011-07 | 53.6 % | -6.0 % | 63.0 | 167.4 | 3,289 |
| B2_ma3 | 2011-08 | 54.4 % | -4.4 % | 67.7 | 185.1 | 3,310 |
| B2_ma3 | 2011-09 | 53.9 % | -26.6 % | 89.1 | 225.6 | 3,324 |
| B2_ma3 | 2011-10 | 57.3 % | -22.6 % | 101.6 | 284.1 | 3,373 |
| B2_ma3 | 2011-11 | 52.1 % | -27.7 % | 114.0 | 384.6 | 3,388 |
| B3_seasonal_naive | 2011-06 | 94.9 % | 42.8 % | 94.5 | 266.1 | 2,734 |
| B3_seasonal_naive | 2011-07 | 82.5 % | 16.0 % | 84.2 | 205.6 | 2,693 |
| B3_seasonal_naive | 2011-08 | 108.2 % | 47.1 % | 114.6 | 375.4 | 2,722 |
| B3_seasonal_naive | 2011-09 | 92.3 % | 34.6 % | 132.0 | 429.4 | 2,787 |
| B3_seasonal_naive | 2011-10 | 88.9 % | 37.3 % | 134.6 | 363.8 | 2,817 |
| B3_seasonal_naive | 2011-11 | 78.9 % | 32.7 % | 150.1 | 450.1 | 2,776 |
| M1_linear | 2011-06 | 59.1 % | -5.5 % | 65.1 | 163.7 | 3,284 |
| M1_linear | 2011-07 | 53.0 % | -11.7 % | 62.3 | 160.1 | 3,289 |
| M1_linear | 2011-08 | 56.7 % | -6.1 % | 70.6 | 187.0 | 3,310 |
| M1_linear | 2011-09 | 55.9 % | -26.3 % | 92.3 | 229.6 | 3,324 |
| M1_linear | 2011-10 | 57.7 % | -21.7 % | 102.2 | 289.4 | 3,373 |
| M1_linear | 2011-11 | 54.6 % | -25.4 % | 119.4 | 414.1 | 3,388 |
| M2_gbm_poisson | 2011-06 | 56.7 % | 9.4 % | 62.5 | 173.8 | 3,284 |
| M2_gbm_poisson | 2011-07 | 48.4 % | -1.9 % | 56.9 | 150.2 | 3,289 |
| M2_gbm_poisson | 2011-08 | 56.7 % | 9.2 % | 70.6 | 185.7 | 3,310 |
| M2_gbm_poisson | 2011-09 | 52.0 % | -11.8 % | 85.9 | 217.5 | 3,324 |
| M2_gbm_poisson | 2011-10 | 57.0 % | 6.8 % | 101.1 | 270.4 | 3,373 |
| M2_gbm_poisson | 2011-11 | 47.3 % | -2.6 % | 103.6 | 356.5 | 3,388 |
| M3_gbm_squared | 2011-06 | 57.5 % | 8.5 % | 63.4 | 167.7 | 3,284 |
| M3_gbm_squared | 2011-07 | 50.2 % | 3.2 % | 59.1 | 147.4 | 3,289 |
| M3_gbm_squared | 2011-08 | 58.9 % | 11.2 % | 73.3 | 190.1 | 3,310 |
| M3_gbm_squared | 2011-09 | 54.8 % | -10.1 % | 90.5 | 219.7 | 3,324 |
| M3_gbm_squared | 2011-10 | 57.8 % | 5.1 % | 102.4 | 265.6 | 3,373 |
| M3_gbm_squared | 2011-11 | 49.2 % | -2.5 % | 107.7 | 335.9 | 3,388 |
| M4_gbm_absolute | 2011-06 | 50.5 % | -20.5 % | 55.6 | 154.8 | 3,284 |
| M4_gbm_absolute | 2011-07 | 46.7 % | -22.0 % | 54.9 | 158.6 | 3,289 |
| M4_gbm_absolute | 2011-08 | 48.6 % | -23.2 % | 60.6 | 190.0 | 3,310 |
| M4_gbm_absolute | 2011-09 | 52.7 % | -35.7 % | 87.1 | 233.9 | 3,324 |
| M4_gbm_absolute | 2011-10 | 51.9 % | -20.8 % | 92.0 | 281.6 | 3,373 |
| M4_gbm_absolute | 2011-11 | 49.3 % | -28.4 % | 108.0 | 417.1 | 3,388 |

### By ABC group (training-window ABC)

| Model | ABC class | wMAPE | Bias | MAE | RMSE | n rows |
|---|---|---|---|---|---|---|
| B1_last_month | A | 50.5 % | -6.6 % | 168.0 | 382.6 | 5,271 |
| B1_last_month | B | 61.2 % | -12.4 % | 66.7 | 214.1 | 5,658 |
| B1_last_month | C | 64.0 % | -9.7 % | 48.3 | 155.9 | 9,039 |
| B2_ma3 | A | 46.9 % | -11.2 % | 155.7 | 356.7 | 5,271 |
| B2_ma3 | B | 63.6 % | -20.3 % | 69.2 | 248.5 | 5,658 |
| B2_ma3 | C | 67.0 % | -30.7 % | 50.5 | 162.6 | 9,039 |
| B3_seasonal_naive | A | 75.8 % | 38.9 % | 250.1 | 546.6 | 4,771 |
| B3_seasonal_naive | B | 123.3 % | 42.0 % | 107.7 | 342.1 | 4,825 |
| B3_seasonal_naive | C | 128.2 % | -11.6 % | 35.8 | 146.9 | 6,933 |
| M1_linear | A | 46.2 % | -15.8 % | 153.6 | 366.5 | 5,271 |
| M1_linear | B | 62.8 % | -26.6 % | 68.4 | 262.6 | 5,658 |
| M1_linear | C | 75.1 % | -16.5 % | 56.7 | 159.0 | 9,039 |
| M2_gbm_poisson | A | 45.9 % | 0.6 % | 152.5 | 350.1 | 5,271 |
| M2_gbm_poisson | B | 59.3 % | -4.2 % | 64.6 | 232.5 | 5,658 |
| M2_gbm_poisson | C | 63.7 % | 5.4 % | 48.0 | 137.1 | 9,039 |
| M3_gbm_squared | A | 46.0 % | 0.6 % | 152.9 | 341.4 | 5,271 |
| M3_gbm_squared | B | 61.9 % | -3.9 % | 67.4 | 219.9 | 5,658 |
| M3_gbm_squared | C | 68.9 % | 8.7 % | 51.9 | 139.7 | 9,039 |
| M4_gbm_absolute | A | 45.0 % | -22.3 % | 149.5 | 381.4 | 5,271 |
| M4_gbm_absolute | B | 57.2 % | -32.0 % | 62.2 | 253.3 | 5,658 |
| M4_gbm_absolute | C | 57.1 % | -29.2 % | 43.1 | 145.1 | 9,039 |

## Champion decision trace

**Champion:** M2_gbm_poisson (ml). **Best gate-1-passing
baseline:** B1_last_month.
**Improvement over the best baseline:** 3.12 wMAPE points —
**meaningful:** True (threshold:
2.00 points).

Post-hoc bias correction is out of scope for this MVP and is not applied anywhere in this pipeline
(PRD §20).

| Model | wMAPE | Bias | Gate 1 (bias) | Months \|bias\| over threshold | Gate 2 rank | Gate 3 decision | Excluded |
|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | True | none | 3 | — | — |
| B2_ma3 | 54.7 % | -17.4 % | False | 2011-09, 2011-11 | — | — | — |
| B3_seasonal_naive | 89.6 % | 35.0 % | False | 2011-06, 2011-08, 2011-09, 2011-10, 2011-11 | — | — | reference_only model, partial hold-out coverage (share=0.827774) |
| M1_linear | 56.0 % | -18.1 % | False | 2011-09, 2011-11 | — | — | — |
| M2_gbm_poisson | 52.6 % | 0.7 % | True | none | 1 | — | — |
| M3_gbm_squared | 54.3 % | 1.5 % | True | none | 2 | — | — |
| M4_gbm_absolute | 50.2 % | -25.8 % | False | 2011-09, 2011-11 | — | — | — |

## Back-test consistency

The hold-out comparison above uses each fitted model's single *fixed* fit (through
2011-05), scored once against the whole hold-out. Baselines need no fitting, so
their hold-out rows are the hold-out slice of the rolling back-test. This section instead
re-scores every candidate at **every** rolling back-test origin, to show how stable each model's
wMAPE and bias are across origins rather than at the single hold-out fit.

| Model | Mean wMAPE | Std wMAPE | Min wMAPE | Max wMAPE | Mean bias | Months \|bias\| over threshold |
|---|---|---|---|---|---|---|
| B1_last_month | 66.8 % | 21.5 % | 47.8 % | 133.8 % | 3.9 % | 4 |
| B2_ma3 | 70.7 % | 24.5 % | 52.1 % | 130.4 % | 4.7 % | 6 |
| B3_seasonal_naive | 100.3 % | 14.4 % | 78.9 % | 122.8 % | 42.6 % | 11 |
| M1_linear | 70.9 % | 32.3 % | 53.0 % | 193.6 % | 3.5 % | 2 |
| M2_gbm_poisson | 66.9 % | 26.8 % | 46.8 % | 158.3 % | 9.2 % | 3 |
| M3_gbm_squared | 70.2 % | 29.3 % | 49.0 % | 168.1 % | 12.9 % | 5 |
| M4_gbm_absolute | 58.3 % | 16.0 % | 46.5 % | 114.0 % | -19.3 % | 10 |

## Inventory KPIs per policy

Two policies, simulated for the champion and B2 (the main baseline) on the identical hold-out
rows: `forecast_only` (target = forecast, no safety stock) and `forecast_plus_ss` (target =
forecast + z x robust σ). Output is always a **Recommended Target Inventory**, never an order
quantity — the dataset carries no on-hand or on-order data (PRD §7).

| Model | Policy | Fill rate | Stockout units | Excess units | Stockout SKU-month rate | Excess per unit shortage |
|---|---|---|---|---|---|---|
| M2_gbm_poisson | forecast_only | 73.3 % | 584,159 | 604,495 | 25.2 % | 1.03 |
| M2_gbm_poisson | forecast_plus_ss | 89.9 % | 221,840 | 2,295,561 | 7.3 % | 10.35 |
| B2_ma3 | forecast_only | 65.5 % | 754,376 | 446,805 | 39.1 % | 0.59 |
| B2_ma3 | forecast_plus_ss | 88.3 % | 256,271 | 2,210,581 | 11.3 % | 8.63 |

### z sensitivity (M2_gbm_poisson, forecast_plus_ss)

| z | Fill rate | Excess units | Stockout units |
|---|---|---|---|
| 1.28 | 88.0 % | 1,880,261 | 262,074 |
| 1.645 | 89.9 % | 2,295,561 | 221,840 |
| 2.05 | 91.4 % | 2,768,055 | 188,852 |

Excess is dominated by a few very large orders: for M2_gbm_poisson under
`forecast_plus_ss`, the `top_1pct_share` of SKU-months by excess account for
17.6 % of total excess units, and the `top_5pct_share` account for
42.2 % of 2,295,561 total excess units
(`excess_concentration.csv`).

σ source, champion, most recent hold-out month 2011-11
(3388 active products, `sigma_summary.csv`): product-level
86.0 %, ABC-group level 14.0 %, global level
0.0 %; zero-MAD share 0.0 %.

## Quarterly aggregation

| Model | Scope | Quarter | wMAPE | Bias | n rows |
|---|---|---|---|---|---|
| B2_ma3 | overall | all | 44.3 % | 3.0 % | 16,306 |
| B2_ma3 | quarter | 2010-Q3 | 43.9 % | -9.3 % | 3,350 |
| B2_ma3 | quarter | 2010-Q4 | 41.4 % | 0.5 % | 3,310 |
| B2_ma3 | quarter | 2011-Q1 | 69.3 % | 42.9 % | 3,329 |
| B2_ma3 | quarter | 2011-Q2 | 36.9 % | 4.4 % | 3,127 |
| B2_ma3 | quarter | 2011-Q3 | 35.7 % | -11.7 % | 3,190 |
| M2_gbm_poisson | overall | all | 40.6 % | 7.2 % | 16,306 |
| M2_gbm_poisson | quarter | 2010-Q3 | 46.8 % | -12.1 % | 3,350 |
| M2_gbm_poisson | quarter | 2010-Q4 | 42.0 % | 27.4 % | 3,310 |
| M2_gbm_poisson | quarter | 2011-Q1 | 49.1 % | 17.2 % | 3,329 |
| M2_gbm_poisson | quarter | 2011-Q2 | 33.9 % | 4.3 % | 3,127 |
| M2_gbm_poisson | quarter | 2011-Q3 | 31.5 % | -4.3 % | 3,190 |

This project trains and evaluates a single **one-step-ahead monthly model**: at any forecast
origin, it predicts only the single month that immediately follows. Every quarterly figure in
`quarterly_forecast.csv` is a **sum of three such one-step-ahead forecasts**, each produced at its
own origin (the month before the one it predicts) by the rolling back-test — never a separate
quarterly model, and never a single forecast covering all three months at once.

As a direct consequence, this MVP
**cannot forecast all three months of a quarter at the start of the quarter**.
The second and third months of any quarter can only be forecast once the preceding month's data
becomes available, one month at a time. A genuine start-of-quarter, three-month-ahead forecast
would require a separate multi-horizon or recursive forecasting approach; that is out of scope for
this MVP and is a candidate v2 extension (PRD §32, §50).

The rolling estimate for the current partial quarter is not an exception to this: it combines the
already-observed actual sales of the quarter's completed months with the single genuine
one-step-ahead forecast for the next month, and is therefore never treated as a scored, complete
quarter (`complete = False`, no `actual_sum`).

## Reproducibility

Seed: 42. Two runs on the same input data and configuration produce byte-identical
artifacts (CLAUDE.md §2 rule 16) — this run's data hash and configuration are recorded in the
header above.

**Versions:**
* `python 3.11.15`
* `pandas 2.2.3`
* `numpy 1.26.4`
* `sklearn 1.5.2`
* `crewai 0.86.0`
* `streamlit 1.39.0`

**Artifacts registered by this run:**
No artifacts were registered by this standalone run.

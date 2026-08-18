# Model Card — Retail Demand Forecasting

*Run:* `20260818T095202Z-00bbdd` · *Generated:* 2026-08-18T09:52:03.308929+00:00 · *Data hash:* `not recorded in this run`

*Provenance:* every number below was computed by the pipeline and is traceable to a table under
`artifacts/reports/evaluation_tables/`, `artifacts/forecasts/`, `artifacts/models/` or
`artifacts/contracts/`. Narrative may be enriched by the LLM agent in LLM mode; the same guard
checks both versions (PRD §38).

## 1. Model purpose

This model forecasts **SKU x month gross demand**: the number of units of one product expected to
sell in the following calendar month, for every product **active** in the 6 months
before the target month (PRD §14). Its forecast feeds a separate, deterministic inventory-policy
layer (PRD §24) that converts it into a **Recommended Target Inventory** using safety stock derived
from out-of-sample forecast errors; the model itself never predicts inventory. The dataset carries
no on-hand or on-order data, so this output is never an order quantity (PRD §7).

## 2. Training data summary

The panel used for training and evaluation covers **100,717** product-month rows across
**4,723** products, `2009-12` to `2011-11` complete, plus
the partial month `2011-12` (shown, never scored).

* **Training window:** `2010-03` to `2011-05` — 52,214 rows fed to
  each hold-out candidate model (`candidates_meta.json`).
* **Hold-out window:** `2011-06` to `2011-11` — months the model never saw
  during training, kept aside to test it honestly.
* **Operational refit:** `model.joblib` is the champion configuration refit through
  `2011-11` (`model_meta.json`), producing the `2011-12`
  forecast.
* **Active-product window:** k = 6 months (PRD §14).

**Cleaning summary** (`data_quality_findings.json` waterfall): 1,067,371 raw sales-line rows in,
1,003,338 rows out after cleaning and de-duplication.

**Source:** `UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci`. **Citation:** `Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. CC BY 4.0`.

## 3. Metrics

wMAPE and Bias are always reported together (PRD §23) — never one without the other.

| Model | wMAPE | Bias | MAE | RMSE | n rows | Coverage | Δ wMAPE vs B2 |
|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | 85.1 | 250.3 | 19,968 | 100.0 % | -1.0 % |
| B2_ma3 | 54.7 % | -17.4 % | 83.6 | 251.1 | 19,968 | 100.0 % | 0.0 % |
| B3_seasonal_naive | 89.6 % | 35.0 % | 118.6 | 359.8 | 16,529 | 82.8 % | -34.8 % |
| M1_linear | 56.0 % | -18.1 % | 85.6 | 257.8 | 19,968 | 100.0 % | -1.3 % |
| M2_gbm_poisson | 52.6 % | 0.7 % | 80.3 | 237.0 | 19,968 | 100.0 % | 2.1 % |
| M3_gbm_squared | 54.3 % | 1.5 % | 83.0 | 230.9 | 19,968 | 100.0 % | 0.4 % |
| M4_gbm_absolute | 50.2 % | -25.8 % | 76.6 | 257.1 | 19,968 | 100.0 % | 4.6 % |

**Champion:** M2_gbm_poisson (ml), selected by the PRD §20 gates
(bias, accuracy, inventory tie-break, meaningful improvement), executed by code — never picked by
hand. Best gate-1-passing baseline: B1_last_month.
Improvement over that baseline: 3.12 wMAPE points (meaningful:
True).

### By hold-out month

| Model | Month | wMAPE | Bias |
|---|---|---|---|
| B1_last_month | 2011-06 | 61.0 % | 9.0 % |
| B1_last_month | 2011-07 | 54.9 % | 0.3 % |
| B1_last_month | 2011-08 | 59.3 % | -3.1 % |
| B1_last_month | 2011-09 | 57.0 % | -23.4 % |
| B1_last_month | 2011-10 | 55.2 % | -4.9 % |
| B1_last_month | 2011-11 | 51.0 % | -16.4 % |
| B2_ma3 | 2011-06 | 58.8 % | -0.7 % |
| B2_ma3 | 2011-07 | 53.6 % | -6.0 % |
| B2_ma3 | 2011-08 | 54.4 % | -4.4 % |
| B2_ma3 | 2011-09 | 53.9 % | -26.6 % |
| B2_ma3 | 2011-10 | 57.3 % | -22.6 % |
| B2_ma3 | 2011-11 | 52.1 % | -27.7 % |
| B3_seasonal_naive | 2011-06 | 94.9 % | 42.8 % |
| B3_seasonal_naive | 2011-07 | 82.5 % | 16.0 % |
| B3_seasonal_naive | 2011-08 | 108.2 % | 47.1 % |
| B3_seasonal_naive | 2011-09 | 92.3 % | 34.6 % |
| B3_seasonal_naive | 2011-10 | 88.9 % | 37.3 % |
| B3_seasonal_naive | 2011-11 | 78.9 % | 32.7 % |
| M1_linear | 2011-06 | 59.1 % | -5.5 % |
| M1_linear | 2011-07 | 53.0 % | -11.7 % |
| M1_linear | 2011-08 | 56.7 % | -6.1 % |
| M1_linear | 2011-09 | 55.9 % | -26.3 % |
| M1_linear | 2011-10 | 57.7 % | -21.7 % |
| M1_linear | 2011-11 | 54.6 % | -25.4 % |
| M2_gbm_poisson | 2011-06 | 56.7 % | 9.4 % |
| M2_gbm_poisson | 2011-07 | 48.4 % | -1.9 % |
| M2_gbm_poisson | 2011-08 | 56.7 % | 9.2 % |
| M2_gbm_poisson | 2011-09 | 52.0 % | -11.8 % |
| M2_gbm_poisson | 2011-10 | 57.0 % | 6.8 % |
| M2_gbm_poisson | 2011-11 | 47.3 % | -2.6 % |
| M3_gbm_squared | 2011-06 | 57.5 % | 8.5 % |
| M3_gbm_squared | 2011-07 | 50.2 % | 3.2 % |
| M3_gbm_squared | 2011-08 | 58.9 % | 11.2 % |
| M3_gbm_squared | 2011-09 | 54.8 % | -10.1 % |
| M3_gbm_squared | 2011-10 | 57.8 % | 5.1 % |
| M3_gbm_squared | 2011-11 | 49.2 % | -2.5 % |
| M4_gbm_absolute | 2011-06 | 50.5 % | -20.5 % |
| M4_gbm_absolute | 2011-07 | 46.7 % | -22.0 % |
| M4_gbm_absolute | 2011-08 | 48.6 % | -23.2 % |
| M4_gbm_absolute | 2011-09 | 52.7 % | -35.7 % |
| M4_gbm_absolute | 2011-10 | 51.9 % | -20.8 % |
| M4_gbm_absolute | 2011-11 | 49.3 % | -28.4 % |

### By ABC group (training-window ABC)

| Model | ABC class | wMAPE | Bias |
|---|---|---|---|
| B1_last_month | A | 50.5 % | -6.6 % |
| B1_last_month | B | 61.2 % | -12.4 % |
| B1_last_month | C | 64.0 % | -9.7 % |
| B2_ma3 | A | 46.9 % | -11.2 % |
| B2_ma3 | B | 63.6 % | -20.3 % |
| B2_ma3 | C | 67.0 % | -30.7 % |
| B3_seasonal_naive | A | 75.8 % | 38.9 % |
| B3_seasonal_naive | B | 123.3 % | 42.0 % |
| B3_seasonal_naive | C | 128.2 % | -11.6 % |
| M1_linear | A | 46.2 % | -15.8 % |
| M1_linear | B | 62.8 % | -26.6 % |
| M1_linear | C | 75.1 % | -16.5 % |
| M2_gbm_poisson | A | 45.9 % | 0.6 % |
| M2_gbm_poisson | B | 59.3 % | -4.2 % |
| M2_gbm_poisson | C | 63.7 % | 5.4 % |
| M3_gbm_squared | A | 46.0 % | 0.6 % |
| M3_gbm_squared | B | 61.9 % | -3.9 % |
| M3_gbm_squared | C | 68.9 % | 8.7 % |
| M4_gbm_absolute | A | 45.0 % | -22.3 % |
| M4_gbm_absolute | B | 57.2 % | -32.0 % |
| M4_gbm_absolute | C | 57.1 % | -29.2 % |

## 4. Limitations

* **Twenty-four months of history only.** The dataset runs `2009-12` to
  `2011-11` complete; there is no earlier history to validate longer seasonal cycles.
* **December 2011 is partial** (`2011-12`) — shown as a partial actual, never scored,
  and it is the target of the latest operational forecast only.
* **No on-hand or on-order inventory data.** The output is a **Recommended Target Inventory**, never
  an order quantity (PRD §7); an actual replenishment quantity cannot be computed from this dataset.
* **Cold start.** A product with no observed month at or before the forecast origin gets no model
  forecast and no invented history (status `Insufficient History / New Product`).
* **Extreme orders.** A handful of very large wholesale lines exist in the data; safety stock uses a
  robust spread measure (median absolute deviation) rather than a plain standard deviation so a few
  outliers cannot inflate the recommendation for every similar product (PRD §26).
* **Gross demand, not net of returns.** The target is units sold on positive sales lines;
  cancellations and adjustments are never subtracted, because inventory was needed to fulfil the
  original order (PRD §9).
* **Left-censoring.** Products already selling in the first observed month may be older than the
  data shows; `product_age_months` is a lower bound, not a true age (PRD §47).
* **Service-level disclaimer:** `z = 1.645 does not guarantee a 95% fill rate; the achieved fill rate is measured in the back-test (PRD §25)`
* **Quarterly figures are sums of one-step-ahead monthly forecasts, not a genuine three-month-ahead
  forecast** — see the methodology note directly below.
* **σ fallback.** A product needs at least 6 out-of-sample
  residuals to use its own σ; otherwise the estimate falls back through
  product -> abc_group -> global (PRD §27).
* **Post-hoc bias correction is out of scope** for this MVP — it proved unstable and is explicitly
  excluded (PRD §20).

### Quarterly methodology note

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

## 5. Ethical considerations

* **No decisions about individuals.** This model forecasts aggregate product demand; it never
  scores, ranks or makes a decision about a person.
* **Customer ID is never a feature.** Rows without a `Customer ID` are kept (the sale still
  happened and still needed stock), and the identifier itself is diagnostic-only, never a model
  input.
* **Anonymised data.** The underlying dataset carries no names, addresses or other direct
  identifiers — only a numeric customer id, country and transaction detail.
* **Residual risks:**
  * *Blind reliance* — a planner treating the forecast or the recommended inventory as certain
    rather than an estimate with a measured error and fill rate.
  * *Overstock from poor uncertainty estimates* — σ estimated from too few residuals (product,
    then ABC-group, then global fallback) can misstate the safety stock for a thin-history product.
  * *Stockouts from a biased model* — a model whose bias gate would fail if re-checked out of band
    (e.g. after a demand shift the pipeline has not yet re-evaluated).
  * *Misinterpretation of safety stock* — `z = 1.645 does not guarantee a 95% fill rate; the achieved fill rate is measured in the back-test (PRD §25)`
* **Scope of the recommendation.** Every number here relies only on the historical sales recorded
  in this dataset; it carries no external signal (promotions, competitor activity, macroeconomic
  data) and should not be read as one.
* **Licensing and attribution.** Source: `UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci`. Citation: `Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. CC BY 4.0`. This
  attribution is carried into every derived artifact this pipeline produces.

## Configuration

* Active-product window: k = 6 months.
* Lead time: 1 month(s).
* Safety-stock z: 1.645 (offered: 1.28, 1.645, 2.05).
* σ: 1.4826 x MAD, minimum 6 residuals, fallback
  product -> abc_group -> global.
* Champion gates: max |bias| 10.0 %, monthly bias report threshold
  25.0 %, tie band 1.00 points,
  meaningful improvement 2.00 points, similar-fill-rate
  tolerance 1.0 %.
* Temporal split: train 2010-03..2011-05, validation
  2011-01..2011-05, hold-out
  2011-06..2011-11, never scored: 2011-12.

## Version

* Run: `20260818T095202Z-00bbdd`. Data hash: `not recorded in this run`. Seed: 42.
* Library versions:
  * `python 3.11.15`
  * `pandas 2.2.3`
  * `numpy 1.26.4`
  * `sklearn 1.5.2`
  * `crewai 0.86.0`
  * `streamlit 1.39.0`
* Artifacts registered by this run:
  * `artifacts/reports/evaluation_report.md` — 12,971 bytes

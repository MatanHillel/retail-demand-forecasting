# Product Requirements Document (PRD)

## Retail Demand Forecasting & Inventory Planning System

**Version:** 1.3 (reviewed and corrected — see Appendix B for the change log from v1.0)  
**Status:** MVP Specification — approved for implementation  
**Dataset:** Online Retail II (UCI Machine Learning Repository; Kaggle mirror)  
**Target Product:** Product-level (SKU × month) demand forecasting and target-inventory recommendation  
**Application:** Streamlit (optional HTML/CSS styling)  
**Machine Learning:** Python / Pandas / NumPy / Scikit-learn / Matplotlib / Seaborn  
**Orchestration:** CrewAI Flow (two crews, ≥ 3 agents each)  
**Course deliverables covered by this PRD:** repository + artifacts + documentation, business presentation (10–12 slides), demo recording (≤ 5 min)

> **Convention used in this document.** Numbers labelled *indicative* come from the feasibility analysis performed before this PRD (same dataset, cleaning rules of §10, k = 6). They exist to set expectations and acceptance thresholds. The final numbers reported by the project must be produced by the project's own pipeline and must never be hard-coded.

---

# 1. Research Question

> **How many units of each product are expected to be sold in each month (and, by aggregation, in each quarter), based on historical sales data — and what target inventory level is required according to that forecast in order to reduce both stockouts and excess inventory?**

בעברית: מהי כמות היחידות הצפויה להימכר מכל מוצר בכל חודש (ובאגרגציה לרבעון), על סמך נתוני המכירות ההיסטוריים, ומהי כמות המלאי הנדרשת בהתאם לתחזית זו כדי לצמצם חוסרים ומלאי עודף?

The question is implemented as two separate layers:

1. **Forecasting layer (machine learning).** Predict the number of units each active product will sell in the next month.
2. **Inventory-policy layer (business rule).** Convert the demand forecast into a recommended target inventory using forecast demand, forecast uncertainty and a stated service level.

The predictive model forecasts **demand**, not inventory. The inventory recommendation is calculated in a deterministic rule layer after the forecast is produced.

**Sub-questions (used to structure the evaluation report and the presentation):**

- SQ1 (descriptive, Data Analyst Crew): How is demand distributed over time, products and countries; how strong is seasonality; how long do products live in the catalog; how concentrated is demand?
- SQ2 (predictive, Data Scientist Crew): Which forecasting approach — naive baselines, Linear Regression, Gradient Boosting variants — is most accurate (wMAPE) and least biased (Bias) on a temporal hold-out?
- SQ3 (business): Which combination of forecast and inventory policy yields the best trade-off between fill rate and excess inventory in a back-test?

---

# 2. Product Summary

The product serves an inventory / operations manager who needs to know, for every active product: which products are expected to sell next month and how many units; which forecasts are uncertain; how much inventory to hold; how the forecast compares with the product's history; how the ML model compares with simple methods; and how the chosen inventory policy affects stockouts and excess.

The system converts historical transaction data into an analytical product that supports inventory-planning decisions.

---

# 3. Business Problem

A retail company manages thousands of products with very different sales patterns: months without sales, large one-time wholesale orders, strong seasonality, short lifecycles, products entering and leaving the catalog, and huge volume differences between products.

Without a forecasting system the company faces two risks:

**3.1 Stockouts** — less inventory than demand: lost sales, lower customer satisfaction, emergency replenishment.  
**3.2 Overstock** — more inventory than needed: capital tied up, storage cost, obsolescence.

The product supports a better balance between the two.

---

# 4. Product Vision

Transform raw sales history into a simple, actionable recommendation:

> **Product X is expected to sell 820 units next month. Its forecast uncertainty requires 116 units of safety stock. The recommended target inventory for the period is 936 units.**

The user does not need to understand Gradient Boosting, lag features, residuals or robust statistics. Technical complexity stays behind the interface.

---

# 5. Users

**5.1 Primary — Inventory / Operations Manager:** forecast by SKU, recommended target inventory, products with high stockout risk, CSV export, forecast-vs-history comparison.  
**5.2 Secondary — Business / Data Analyst:** trends, seasonality, product performance, forecast-vs-actual, product segmentation.  
**5.3 Technical — Data Scientist / Project Administrator:** model quality, baseline-vs-ML, wMAPE and Bias, residual analysis, model version, data validation, pipeline status and run history.

---

# 6. Product Scope

## 6.1 In scope (MVP)

The system forecasts **SKU × Month → Units Sold** for **active products** (definition in §14). The MVP provides: monthly demand forecast, quarterly aggregation of monthly forecasts, robust safety stock, recommended target inventory, model evaluation (baselines vs. ML, wMAPE + Bias), inventory-policy back-test (fill rate, stockout units, excess units), product-level history visualisation, pipeline / data-quality status, CSV export.

## 6.2 Out of scope (MVP)

Forecasts for products with no sales history (cold start); forecasting by country, store or warehouse; supplier management and real lead times; current on-hand / on-order inventory and purchase orders; price optimisation; true reorder-quantity calculation; direct 2- and 3-month-ahead forecasts; full-quarter forecast at the start of the quarter; forecasting returns / cancellations; customer-level analytics (RFM, churn) beyond descriptive EDA.

---

# 7. Target Inventory vs. Order Quantity

The dataset does not contain the company's inventory position, so the system cannot recommend "order 500 units". It recommends: *"to support expected demand and forecast uncertainty, the target inventory level should be 500 units."* In production the order quantity would be `Target Inventory − On Hand − On Order + Backorders`; those fields are not available. The MVP output is therefore named **Recommended Target Inventory**, never *Recommended Order Quantity*.

---

# 8. Dataset Overview

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 502 "Online Retail II" (Chen, 2019), licence CC BY 4.0. Kaggle mirror `mashlyn/online-retail-ii-uci` (single CSV, 94.85 MB, listed as CC0). Either source may be used; the origin (UCI) must be cited. |
| Business | UK-based, non-store online retailer of all-occasion gift-ware; many wholesale customers |
| Raw size | 1,067,371 transaction lines in two yearly sheets (2009–2010: 525,461; 2010–2011: 541,910) |
| Columns (8) | Invoice, StockCode, Description, Quantity, InvoiceDate, Price (GBP), Customer ID, Country |
| Period | 1 Dec 2009 – 9 Dec 2011. **24 full months: Dec 2009 – Nov 2011.** December 2011 is a **partial month** (through 9 Dec) and is never used as an evaluation month. December 2009 is complete. |
| Product key | `StockCode` (not `Description`). 24.5 % of stock codes appear with more than one description; ~0.4 % of rows have no description. `Description` is a display field only: **Canonical Description = the most frequent non-empty description for the StockCode.** |
| Known data-quality facts (indicative) | 22.8 % of rows have no Customer ID (14.8 % of revenue) — kept for demand; ~15.5 % of invoice numbers are cancellations (prefix "C") but only 1.8 % of rows; ~3,500 negative-quantity rows without "C" (stock adjustments); ~6,200 zero-price rows; 6 rows on "A" invoices (bad-debt adjustments, mostly negative price); ~3,500 rows with lower-case stock-code variants (e.g. `85123a` vs `85123A`); 12,133 exactly duplicated rows; 28 non-inventory stock codes (~4,700 rows; see §12); a single line can reach 80,995 units; almost no Saturday trading. |
| Countries | 43; United Kingdom ≈ 92 % of rows and ≈ 82 % of units. Country is not a forecasting dimension in the MVP (§6.2). |

---

# 9. Definition of Demand

The MVP forecasts **gross demand**: units sold to customers before subtracting later returns.

```text
Gross Demand(product, month) = Σ Quantity over positive sales lines of that product in that month
```

Rules:

- Cancellations (invoices starting with "C") and stock adjustments are **not** subtracted from the target. Rationale: inventory was required to fulfil the original order even if the item was later returned. Returns are preserved for EDA (`returned_units`) but are not the target.
- Rows **without Customer ID are included** — they are real sales (≈ 15 % of revenue) and demand does not depend on the customer being identified.
- Rows with `Price ≤ 0` are excluded from demand: in this dataset they are overwhelmingly stock adjustments, samples and data errors (no customer, empty or note-like descriptions). The exclusion count is logged and reported (§10). If a team review finds a material share of legitimate free-of-charge shipments, this rule may be revisited and documented in the contract.
- Country and Customer ID are not used as model inputs.

---

# 10. Data Cleaning Requirements

The cleaning pipeline must be deterministic, reproducible, config-driven and logged as a **row-count waterfall** (rows before / removed / after for every step, plus impact on total units). Steps, in order:

| # | Step | Rule | Indicative effect |
|---|---|---|---|
| 1 | Load raw data | Read both sheets (or the Kaggle CSV); record file hash, row count, column list | 1,067,371 rows |
| 2 | Validate raw schema | Required columns present; dtypes coercible; `InvoiceDate` parses; fail the flow otherwise | — |
| 3 | Normalise keys | `StockCode`: strip whitespace, upper-case (merges ≈ 3,500 lower-case variants such as `85123a`); `Invoice`: string; `Description`: strip | — |
| 4 | Convert dates | `InvoiceDate` → datetime; derive `month` (YYYY-MM); flag `is_partial_month` for 2011-12 | — |
| 5 | Remove cancellations and adjustments | `Invoice` starts with "C" (cancellations) or "A" (bad-debt adjustments) | −19,494 (C) / −6 (A) |
| 6 | Remove non-positive quantities | `Quantity ≤ 0` remaining after step 5 (stock adjustments) | −3,457 |
| 7 | Remove non-positive prices | `Price ≤ 0` | −2,745 |
| 8 | Remove non-inventory stock codes | Exclusion list `config/non_inventory_stockcodes.csv` (§12) | −4,667 |
| 9 | Remove exact duplicate rows | All 8 columns identical (§11) | −12,051 → 1,024,951 sales rows, ≈ £20.05 M |
| 10 | Canonical descriptions | Most frequent non-empty description per StockCode | — |
| 11 | Aggregate to Product × Month | Fields of §13 | ≈ 65–67 k non-zero product-months |
| 12 | Fill zero-sales months | For each product, every month from its first observed sale to Nov 2011 (plus Dec 2011 flagged partial) with `units_sold = 0` when no sales | ≈ 99 k rows (≈ 104 k incl. the partial Dec 2011 rows), ≈ 4,890 products |
| 13 | Data-quality report | Waterfall table, warnings (§11 threshold, abnormal quantities, partial month) | — |

Abnormal quantities (e.g. lines > 5,000 units) are **not removed** — they are real demand — but they are counted, listed in the data-quality report and handled downstream by robust statistics (§26).

---

# 11. Duplicate Handling

The raw data has fully identical rows and no line identifier, so a duplicate cannot be distinguished from a legitimate repeated line. **MVP policy: exact duplicate rows are removed.** The pipeline records rows before, duplicate count, rows after and the impact on total units, and raises a data-quality **warning if duplicates exceed 1.5 % of rows or 1 % of total units**. The count is computed by the pipeline, never hard-coded.

---

# 12. Non-Inventory StockCodes

The forecasting product operates on **inventory-bearing products only**. Rule of thumb: a product code matches `^\d{5}[A-Z]{0,2}$` or starts with `DCGS`; everything else is reviewed. Initial exclusion list (to be confirmed by the Data Quality Analyst and stored in `config/non_inventory_stockcodes.csv` with a reason column):

`POST` (postage), `DOT` (dotcom postage), `C2`, `C3` (carriage / adjustment), `M` / `m` (manual), `D` (discount), `S` (samples), `BANK CHARGES`, `ADJUST`, `ADJUST2`, `AMAZONFEE`, `CRUK` (commission), `TEST001`, `TEST002`, `B` (bad debt), `GIFT` and `gift_0001_10 … gift_0001_90` (gift vouchers), `PADS` (packaging, price 0.001 — review), `SP1002` (a real product with a non-standard code; only 3 return rows — documented, not modelled). Indicative total: 28 codes, ≈ 4,700 rows.

Every exclusion must be documented in the EDA report and in `dataset_contract.json`.

---

# 13. Processed Data Structure

## 13.1 `clean_transactions.parquet` (intermediate, audit)

The cleaned transaction-level table (≈ 1.02 M rows, ≈ 6–14 MB compressed) is kept in `data/processed/` for EDA on customers, countries and invoices and for auditability. It is not the hand-off artifact.

## 13.2 `clean_data.csv` (hand-off artifact, grain StockCode × Month)

| Field | Type | Description |
|---|---|---|
| `month` | YYYY-MM | Calendar month |
| `stock_code` | str | Product key |
| `description` | str | Canonical description |
| `units_sold` | int ≥ 0 | Gross units (target source) |
| `gross_revenue` | float ≥ 0 | Σ Quantity × Price |
| `avg_unit_price` | float | Revenue-weighted average price; last known price for zero months |
| `invoice_count` | int ≥ 0 | Distinct invoices |
| `sale_line_count` | int ≥ 0 | Positive sales lines |
| `customer_count` | int ≥ 0 | Distinct identified customers (diagnostic) |
| `max_line_qty` | int ≥ 0 | Largest single line (outlier diagnostic) |
| `returned_units` | int ≥ 0 | Units on "C" invoices in the month (EDA only, never a feature) |
| `is_partial_month` | bool | True for 2011-12 |

Rows exist for every month from the product's first observed sale to Nov 2011 (Dec 2011 flagged); `units_sold = 0` when no sales. No rows before the first observed sale. Primary key `(stock_code, month)`.

---

# 14. Active Product Definition

A product is **active** for target month *t* if it had at least one positive sale in the six months immediately preceding *t* (months *t−6 … t−1*): **k = 6**. Indicative effect: ≈ 70,700 product-month observations, ≈ 4,850 products, ≈ 27 % zero targets. k = 1 removes intermittent products too quickly (≈ 53 k rows, 12 % zeros); k = 12 keeps dead products longer (≈ 78 k rows, 34 % zeros). k is a configuration value (`model_config.yaml`), reported in the contract and the model card.

---

# 15. Cold Start

Products with no sales history receive no model forecast; status **Insufficient History / New Product**. Future information must never be used to create artificial history. Feasibility analysis showed that the first 14 days of a new product's sales predict its days 15–90 well (Spearman ≈ 0.84), so an early-sales cold-start rule is a Version 2 candidate.

---

# 16. Modelling Target and Forecast Origin

For product *i* and target month *t*:

```text
y(i,t) = gross units sold in month t
forecast origin = end of month t−1
allowed information = everything with month ≤ t−1
```

No feature may use information from month *t* or later. Calendar attributes of *t* (month-of-year, quarter) are allowed because they are known in advance. **Latest operational forecast:** origin = November 2011 (last full month) → target = December 2011. Actual December 2011 is partial and is shown only as "partial actual", never scored.

---

# 17. Feature Engineering and `features.csv`

One **global model** is trained across all eligible products (no per-SKU models). `stock_code` is an identifier, not a feature. Conventions (documented in the contract): for months before a product's first observed sale (products launched after Dec 2009) lags are 0; for months before the dataset start (Dec 2009) demand is *unobserved*, so rolling statistics use only observed months (truncated windows) and `product_age_months` is a lower bound (left-censoring, §47). The first target month is March 2010 (three observed lags).

| Feature | Definition (all relative to origin t−1) |
|---|---|
| `lag_1`, `lag_2`, `lag_3` | units in t−1, t−2, t−3 |
| `rolling_mean_3`, `rolling_mean_6` | mean units over t−3…t−1 and t−6…t−1 |
| `rolling_median_6` | median units over t−6…t−1 |
| `rolling_std_3` | std of units over t−3…t−1 |
| `rolling_max_6` | max units over t−6…t−1 (captures wholesale spikes) |
| `nonzero_months_6` | months with positive sales in t−6…t−1 |
| `months_since_last_sale` | months since last positive sale (as of t−1) |
| `product_age_months` | months since first observed sale (as of t−1) |
| `invoice_count_lag_1` | distinct invoices in t−1 (depth of demand vs. one big order) |
| `avg_unit_price_lag_1` | price in t−1 (last known if zero month) |
| `target_month_of_year`, `target_quarter` | calendar attributes of t |

`features.csv` columns: `stock_code`, `forecast_origin` (YYYY-MM = t−1), `target_month` (YYYY-MM = t), all features above, `y` (target), `is_active` (always true in the saved file). Primary key `(stock_code, target_month)`. Optional v2 features: `lag_12` (NaN-tolerant models only), `trend_3` = lag_1 − lag_3, ABC class computed on the training window only.

---

# 18. Features Not Used as Core Features

**18.1 `lag_12` / seasonal-naive** — many products have short lifecycles; a same-month-last-year forecast performs poorly at SKU level (indicative wMAPE 80–90 %). Not a core feature or main baseline; may be reported for reference.  
**18.2 ABC / XYZ** — used for reporting and evaluation. If used as a feature, computed only from information before the forecast origin (never from the whole dataset — leakage).

---

# 19. Forecasting Approaches

| Id | Approach | Definition | Role |
|---|---|---|---|
| B1 | Last month | ŷ(t) = y(t−1) | baseline |
| B2 | 3-month moving average | ŷ(t) = mean(y(t−1), y(t−2), y(t−3)) | **main baseline** |
| B3 | Seasonal naive (reference) | ŷ(t) = y(t−12) where available | reference only |
| M1 | Linear Regression | on the §17 features; negative predictions reported and clipped to 0 for business use | interpretable ML benchmark; not expected to beat B2 |
| M2 | Gradient Boosting — Poisson loss | `HistGradientBoostingRegressor(loss="poisson")` | **primary ML model** (count target, mean forecast, non-negative) |
| M3 | Gradient Boosting — squared error | `loss="squared_error"` | required second GBM variation |
| M4 | Gradient Boosting — absolute error | `loss="absolute_error"` | challenger (accuracy vs. bias trade-off, §20) |

At least two model variations must be compared (course requirement); the MVP compares four models plus baselines. Hyper-parameters are tuned on rolling-origin folds inside the training period only (small grid: learning rate, leaves, iterations); the final configuration is written to `model_config.yaml`.

---

# 20. Loss Function and Champion Policy

**Why this matters (indicative findings).** On the k = 6 panel with a June–November 2011 hold-out, Gradient Boosting with `absolute_error` reached wMAPE ≈ 49 % versus ≈ 55 % for baselines — but with **Bias ≈ −24 %** (systematic under-forecast; it predicts a median of a right-skewed distribution). Poisson / squared-error variants were nearly unbiased (Bias ≈ +1 %… +4 %) with wMAPE ≈ 53–55 %. Post-hoc multiplicative bias correction of the absolute-error model was unstable across periods (over-shooting on the hold-out) and is therefore **not** an MVP technique. Baseline B2 itself showed Bias ≈ −17 % on the same hold-out.

**Champion rules (apply to every candidate, baselines included):**

1. **Gate 1 — Bias:** |Bias| ≤ 10 % on the hold-out (and no single hold-out month with |Bias| > 25 % is silently accepted — report it).
2. **Gate 2 — Accuracy:** among candidates passing Gate 1, lowest wMAPE.
3. **Gate 3 — Inventory outcome:** if two candidates are within 1 wMAPE point, prefer the one with less excess at similar fill rate in the back-test (§29–30).
4. **Meaningful improvement:** an ML model is described as "improving on the baseline" only if it beats the best gate-passing baseline by ≥ 2 wMAPE points (absolute). Otherwise the report states transparently that simple methods are competitive — a legitimate finding.

The MVP primary Gradient Boosting model uses **Poisson loss** (natural for non-negative counts, unbiased in mean); squared error is the second variation; absolute error remains a challenger and is expected to fail Gate 1. The champion selection is executed by code (§37 step 7) and written to `evaluation_report.md` and `run_log.json`.

---

# 21. Train / Test Methodology

`train_test_split(shuffle=True)` is forbidden: rows of the same product in nearby months would leak. The split is **temporal**:

- **Training targets:** March 2010 – May 2011 (first target month needs 3 lags).
- **Internal validation for tuning:** rolling-origin folds inside training (e.g. targets 2011-01 … 2011-05).
- **Final hold-out:** targets **June 2011 – November 2011** (6 months, includes the September–November peak by design).
- December 2011 is never scored (partial).

---

# 22. Rolling-Origin Evaluation

Beyond the final hold-out, the pipeline runs rolling-origin back-testing: train through month X → forecast X+1, then X+1 → X+2, … from origin 2010-05 to 2011-10 (expanding window, same feature spec, same hyper-parameters). Purposes: (1) consistency of performance across origins; (2) **out-of-sample residuals for safety stock**. Every back-test prediction is stored in `artifacts/forecasts/backtest_predictions.csv` with `forecast_origin, target_month, stock_code, model, actual, prediction, residual`. Residuals used for sigma at evaluation month *t* must come from origins **before** *t*.

---

# 23. Forecast Accuracy Metrics

- **wMAPE (primary):** `Σ|Actual − Forecast| / Σ Actual` — well-defined with zero-demand months.
- **Bias:** `Σ(Forecast − Actual) / Σ Actual` — 0 % none; −20 % under-forecast; +20 % over-forecast. **wMAPE and Bias are always reported together.**
- **Secondary:** MAE, RMSE, wMAPE/Bias by month, by ABC group (ABC from training window), relative improvement vs. B2, share of negative predictions (M1).

Reporting granularity is mandatory: overall, per hold-out month, per ABC group, and per model.

---

# 24. Inventory Recommendation

Inventory is a **policy**, not a second ML model:

```text
Target Inventory = Forecast Demand During Lead Time + Safety Stock
MVP: Lead Time = 1 month  ⇒  Target Inventory = Next-Month Forecast + Safety Stock
```

Lead time and service level are declared assumptions in `config/inventory_policy.yaml` and in the model card.

---

# 25. Safety Stock

`Safety Stock = z × σ`, with `z = 1.645` by default (95 % one-sided normal assumption) and σ = robust forecast-error spread (§26). **The product must not claim that z = 1.645 guarantees a 95 % fill rate**; the achieved fill rate is measured empirically (§29–30). The z value is a user-adjustable slider in the app (e.g. 1.28 / 1.645 / 2.05).

---

# 26. Sigma Calculation

σ is computed **only from out-of-sample residuals** of the rolling-origin back-test (never from in-sample fit), per product, using the robust estimator

```text
e = Actual − Forecast;  MAD = median(|e − median(e)|);  Robust σ = 1.4826 × MAD
```

MAD limits the influence of extreme wholesale orders that inflate ordinary standard deviation. A product-level σ requires at least **6** out-of-sample residuals.

---

# 27. Hierarchical Sigma

Fallback hierarchy: **Level 1** product σ (≥ 6 residuals) → **Level 2** ABC-group σ (ABC on the training window) → **Level 3** global σ. The level used is stored per product (`sigma_source`) and shown in the app. All group statistics use only information available before the evaluation period.

---

# 28. Recommended Inventory Calculation

```text
Recommended Target Inventory = ceil( max(0, Forecast + z × Robust σ) )
Example: Forecast 820, σ 70, z 1.645 → 820 + 115.15 = 935.15 → 936 units
```

---

# 29. Inventory Simulation

For every SKU × month of the hold-out (same rows for every policy):

```text
Shortage = max(Actual − Target Inventory, 0)
Excess   = max(Target Inventory − Actual, 0)
Fulfilled = min(Target Inventory, Actual)
```

Policies compared: (a) forecast only (no safety stock); (b) forecast + robust safety stock (z = 1.645); (c) the same for the baseline B2 — so the business value of the ML forecast is measured at equal service level.

---

# 30. Inventory KPIs

Fill Rate = Σ Fulfilled / Σ Actual; Stockout Units = Σ Shortage; Excess Units = Σ Excess; Stockout SKU-Month Rate = share of rows with Actual > Target; **Excess-per-unit-of-shortage** (ratio) as a summary of the trade-off. Reported overall, by month and by ABC group. Indicative expectation: with safety stock both ML and baseline reach ≈ 92 % fill rate; the ML policy needs somewhat less excess (≈ 5–13 % less at equal fill rate). Excess is dominated by a few very large orders — the report must say so.

---

# 31. Quarterly Aggregation

No separate quarterly model. Quarter forecast = Σ of the three genuine one-step-ahead monthly forecasts, each generated at its own origin (July at end of June, August at end of July, September at end of August). This preserves the one-step-ahead methodology.

# 32. Quarterly Limitation

The MVP cannot forecast all three months of a quarter at the start of the quarter (would require direct multi-horizon or recursive forecasting — v2). Quarterly aggregation is for back-testing, reporting and rolling aggregation of monthly forecasts.

---

# 33. Product Screens (Streamlit)

1. **Executive Dashboard** — active products, total next-month forecast, total target inventory, champion wMAPE and Bias, hold-out fill rate, stockout and excess units, run id / timestamp / data hash.
2. **Product Forecasts** — table: Product, Description, Last Month, 3M Avg, Forecast, Safety Stock, Target Inventory, Sigma source, ABC; filters (StockCode / description search, ABC, high uncertainty, forecast > 0); **Download CSV**.
3. **Product Detail** — monthly chart (actual, historical back-test forecasts, next-month forecast with ± z·σ band); inventory recommendation block; metadata (age, months since last sale, ABC, active status).
4. **Model Evaluation** — B1, B2, M1, M2, M3, M4: wMAPE, Bias, MAE, RMSE; by month; by ABC; actual-vs-forecast; champion decision trace (which gate each candidate passed).
5. **Inventory Policy Evaluation** — forecast-only vs. forecast + robust safety stock, ML vs. baseline, z slider; fill rate, stockout units, excess units, stockout SKU-month rate.
6. **Pipeline & Data Quality** — cleaning waterfall, duplicate and exclusion counts, contract validation status, last run status, failure message if any (§39), model card and evaluation report rendered.
7. **Data & Insights (EDA)** — the mandatory EDA figures of §35A (monthly demand, seasonality, lifecycle, ABC Pareto, top products, intermittency, outliers, countries) with the matching insights from `insights.md`; download of the EDA tables.

Optional: an "as-of month" selector to replay any historical origin (useful for the demo). Charts use Matplotlib/Seaborn or Streamlit-native charts; optional HTML/CSS theming.

---

# 34. CrewAI Architecture

CrewAI orchestrates the workflow; it does not replace the deterministic Python pipeline. Two crews, each with three agents, each agent equipped with approved Python tools. LLM output is used for narrative artifacts (insights, model card, evaluation narrative) and for reviewing deterministic results — never for computing numbers (§38).

# 35. Data Analyst Crew

| Agent | Responsibilities | Tools (deterministic) | Outputs |
|---|---|---|---|
| **1. Data Quality Analyst** | schema validation, missing values, duplicates, cancellations, returns, abnormal quantities, partial months, non-product codes | `profile_raw()`, `detect_duplicates()`, `list_nonproduct_codes()` | `data_quality_findings.json` (+ section in eda_report) |
| **2. Data Preparation Analyst** | run cleaning pipeline (§10), canonical descriptions, aggregate to Product × Month, zero-fill, partial-month flag | `clean_transactions()`, `build_panel()` | `clean_transactions.parquet`, **`clean_data.csv`** |
| **3. Business & EDA Analyst** | EDA and visual insights (monthly units/revenue, seasonality, product lifecycle, intermittency, ABC concentration, country mix, top products), business summary, dataset contract | `run_eda()`, `write_contract()` | **`eda_report.html`** (charts + tables), **`insights.md`**, **`dataset_contract.json`** |

Required content: `eda_report.html` — data-quality waterfall, monthly demand chart, seasonality (share of Sep–Nov), product lifecycle histogram, zero-share by k, ABC curve, top-20 products, country mix; `insights.md` — 8–12 business insights with numbers; `dataset_contract.json` — schema (name, type, nullability), allowed values / ranges, primary key, grain, date range, cleaning assumptions, exclusion list, k, partial-month rule, leakage rules (Appendix A).

# 35A. Descriptive Analytics & Visualization Requirements (Data Analyst Crew)

The course brief requires the Data Analyst Crew to "perform descriptive analytics and exploratory data analysis" and to "produce visual insights and business summaries". This section makes that requirement concrete: which analyses are mandatory, which figures must exist, where they appear, and the quality bar. All numbers behind the figures are computed by deterministic tools and written to `artifacts/reports/eda_tables/*.csv|json`; the LLM agent writes the narrative (`insights.md`) from those tables only.

## 35A.1 Mandatory analyses and figures

| Id | Analysis (on `clean_transactions.parquet` unless stated) | Figure / table (mandatory) | Why it matters for the product |
|---|---|---|---|
| E1 | Data-quality profile: missing values per column, cancellations, adjustments, zero prices, duplicates, non-product codes, partial month | Cleaning **waterfall table** + horizontal bar chart of rows removed per step; missing-value table | Documents the contract assumptions; feeds Screen 6 |
| E2 | Demand over time: monthly units and revenue; year-over-year comparison (Dec 09–Nov 10 vs Dec 10–Nov 11); share of Sep–Nov | Line chart of monthly units (and revenue on a second panel); bar of monthly seasonal index; YoY table | Seasonality drives the test window choice and the month-of-year feature |
| E3 | Intra-week / intra-day patterns | Bar charts of units by weekday and by hour | Shows operating pattern (no Saturday trading; midday peak) — context, not a feature |
| E4 | Product portfolio: number of products with sales per month; new vs. disappearing products per quarter | Line/stacked bar over time | Product churn motivates the active-product rule |
| E5 | Product lifecycle: months with sales per product | Histogram (share of products sold in ≤ 6 months, ≥ 24 months) | Explains why seasonal-naive fails at SKU level |
| E6 | Concentration (ABC): cumulative revenue vs. cumulative share of products; A/B/C counts | Pareto curve with 80 % / 95 % markers; ABC table | Reporting groups for evaluation and sigma fallback |
| E7 | Top products: top-20 by units and by revenue (canonical descriptions) | Bar charts + table | Business narrative; sanity check of key |
| E8 | Intermittency: distribution of zero-month share per product; zero-target share vs. k (1, 3, 6, 12) | Histogram + small table/plot of zero share and rows per k | Justifies k = 6 |
| E9 | Demand magnitude and outliers: distribution of product-month units (log scale); largest single lines / product-months | Log-scale histogram; table of the 20 largest lines with invoice, product, quantity, customer type | Motivates robust sigma and wMAPE |
| E10 | Order structure: invoice value and lines per invoice; average quantity per line (wholesale signal) | Histograms + summary table | Wholesale vs. retail context for the presentation |
| E11 | Customers and countries (descriptive only): identified vs. anonymous rows/revenue; top-10 countries by units and revenue | Bar chart + table (UK share stated) | Explains why country is not a forecasting dimension and why anonymous rows are kept |
| E12 | Returns: cancellation rate over time; top returned products; returned units share | Line chart + table | Explains the gross-demand definition |
| E13 | Price: distribution of unit prices; price stability per product (CV) | Histogram + table | Supports the `avg_unit_price_lag_1` feature |
| E14 | Panel preview (`clean_data.csv`): rows per month, share of zero rows, partial-month rows | Table | Validates the hand-off artifact before the contract is written |

At least figures E1, E2, E5, E6, E7, E8, E9, E11 must appear in `eda_report.html`; the others may be tables. Each figure is accompanied in `insights.md` by a one-sentence "so what" with the number behind it (e.g. "September–November account for ≈ 35 % of annual units, so the hold-out deliberately includes the peak").

## 35A.2 Visualization standards

- Library: Matplotlib / Seaborn (course requirement); one consistent, colour-blind-safe palette across all figures; ABC classes always use the same three colours.
- Every figure has a title, axis labels with units, a source/footnote line ("Online Retail II, cleaned, Dec 2009–Nov 2011"), readable font sizes, and no more than one message per chart. Log scale is stated in the axis label when used.
- Figures are saved as PNG (≥ 150 dpi) under `artifacts/reports/figures/` with stable names (`E02_monthly_units.png`, …) and embedded (base64) into a **single self-contained** `eda_report.html` generated from a template by code, so the report renders offline and inside the repository.
- Tables shown in the report are the same CSV/JSON tables used by the LLM agent for `insights.md` — no number appears in the narrative that is not in a table.
- Product-level charts in the app (Screen 3) reuse the same style: actual units (bars/line), back-test forecasts, next-month forecast with a ± z·σ band; partial December 2011 drawn hatched and labelled "partial".

## 35A.3 Where the visuals appear

`eda_report.html` (full set); `insights.md` (narrative + links to figures); Streamlit Screen 1 (monthly demand line, seasonal index), Screen 3 (product history), Screen 6 (waterfall) and Screen 7 (Data & Insights: embeds the EDA figures and `insights.md`); presentation slides 3–4.

---

# 36. Data Scientist Crew

| Agent | Responsibilities | Tools | Outputs |
|---|---|---|---|
| **1. Feature Engineering Specialist** | validate `clean_data.csv` against the contract, active-product logic, lags / rolling / calendar features, leakage checks | `validate_contract()`, `build_features()`, `leakage_check()` | **`features.csv`**, `feature_validation.json` |
| **2. Forecasting Model Scientist** | baselines, Linear Regression, GBM variants (Poisson, squared, absolute), tuning on rolling folds, serialisation | `train_models()`, `tune()` | **`model.joblib`** (champion) + `models/*.joblib`, `model_config.yaml` |
| **3. Model Evaluation & Inventory Scientist** | temporal evaluation, rolling-origin back-test, wMAPE / Bias / MAE / RMSE, by month & ABC, champion rules, robust σ, inventory simulation, model card | `evaluate()`, `backtest()`, `robust_sigma()`, `simulate_inventory()` | **`evaluation_report.md`**, **`model_card.md`**, `backtest_predictions.csv`, `latest_forecast.csv`, `inventory_plan.csv` |

**`model_card.md` must contain (course requirement):** model purpose; training-data summary (rows, products, period, k, cleaning summary); metrics (wMAPE, Bias, MAE, RMSE overall / by month / by ABC, vs. baselines); limitations (24 months only, partial Dec 2011, no inventory or lead-time data, cold start, extreme orders, gross demand definition); ethical considerations (§48). **`evaluation_report.md` must contain:** comparison table of all candidates, champion decision trace, back-test consistency, inventory KPIs per policy.

---

# 37. CrewAI Flow

```text
Raw data → 1 Dataset intake → 2 Data Analyst Crew → 3 Contract validation → 4 Data Scientist Crew
→ 5 Feature validation → 6 Temporal training & back-testing → 7 Model evaluation & champion selection
→ 8 Inventory policy calibration → 9 Artifact validation → 10 Publish to Streamlit
```

- **State:** a Pydantic `FlowState` (run_id, timestamp, data hash, artifact paths, validation flags, metrics, champion, errors). Deterministic steps are Flow methods that call Python tools; crews are kicked off with `@start` / `@listen`; a `@router` after each validation routes to `continue` or `fail`.
- **Step 3 — Contract validation (deterministic):** required columns, dtypes, null rules, uniqueness of `(stock_code, month)`, date range, grain, non-negative units, partial-month flag consistency. Fail → STOP.
- **Step 5 — Feature validation:** target present, required features present, no NaN in required features, `forecast_origin < target_month` for every row, leakage test (recompute `lag_1` from `clean_data` for a random sample and compare; permutation test: shuffling future months must not change any feature), rows satisfy the active rule. Fail → STOP.
- **Step 7:** champion rules of §20 executed by code.
- **Step 9:** all required artifacts exist with the exact names of §41 and non-zero size.
- **Logging:** structured log per step (`logs/run_<id>.log` + `run_log.json`): inputs, outputs, row counts, durations, warnings, metrics.
- **No-LLM mode:** `python -m pipeline --no-llm` runs steps 1–9 fully deterministically for CI and reproducibility; the LLM-narrative agents run only when the pipeline succeeded.

# 38. Core Architectural Principle

LLM agents must not compute wMAPE, Bias, aggregations, MAD, safety stock, predictions or splits. They run approved tools, interpret deterministic results, write documentation, review outputs and generate business insights.

# 39. Failure Handling

The Flow fails gracefully: on any failed validation it writes `artifacts/validation_report.json` (step, rule, message, counts), logs the failure, exits with a non-zero code and **does not** produce or overwrite forecasts. Examples: `FLOW STOPPED: Missing required column Quantity`; `FLOW STOPPED: clean_data does not match dataset_contract.json (3 violations)`; `FLOW STOPPED: feature uses information after forecast origin`; `FLOW STOPPED: model.joblib was not generated`. Streamlit never shows stale forecasts as current: it displays *"Forecast data unavailable — latest pipeline run failed (run id …)"* with the failure reason.

# 40. Reproducibility

Every run records: run id, timestamp, dataset hash, cleaning / feature / model / inventory configuration, random seed (single global seed = 42), metrics, champion, versions of Python and key libraries (`requirements.txt` pinned). Determinism test: two runs on the same data must yield identical `clean_data.csv`, `features.csv` and metrics (asserted in `tests/`).

# 41. Artifacts and Repository Structure

```text
data/
    raw/                      # not committed (download script + hash check)
    processed/
        clean_transactions.parquet
        clean_data.csv        # required
        features.csv          # required
artifacts/
    models/model.joblib       # required (champion) + candidates
    forecasts/backtest_predictions.csv, latest_forecast.csv, inventory_plan.csv
    reports/eda_report.html, insights.md, evaluation_report.md, model_card.md   # required
    contracts/dataset_contract.json                                            # required
    validation_report.json, run_log.json
config/
    non_inventory_stockcodes.csv, model_config.yaml, inventory_policy.yaml, cleaning_config.yaml
src/  (pipeline/, crews/, flow/, app/)      tests/      logs/      docs/ (PRD, presentation, demo script)
```

The eight artifacts required by the course brief must exist with these exact names: `clean_data.csv`, `eda_report.html`, `insights.md`, `dataset_contract.json`, `features.csv`, `model.joblib` (or `.pkl`), `evaluation_report.md`, `model_card.md`.

# 42. GitHub and Collaboration

- Raw data is not committed (GitHub warns at 50 MiB, blocks at 100 MiB); a script downloads it from UCI/Kaggle and verifies the hash. `clean_data.csv` (panel) and `features.csv` are a few MB and are committed.
- **Pull-request workflow (course requirement):** `main` protected; feature branches per task; every change merged through a PR with at least one reviewer; PR template (what / why / how tested); CI (GitHub Actions) runs `pytest` and the pipeline in `--no-llm` mode on a sample.
- Repository contains: source, processed artifacts, documentation (README with setup, PRD, presentation, demo script), model artifact, reports, configuration, tests.

# 43. Technology Stack

Required: Python 3.11, Pandas, NumPy, Scikit-learn, CrewAI, Streamlit, Matplotlib / Seaborn, Git + GitHub. Serialization: Joblib. Validation: Pydantic + custom checks. Testing: pytest. Optional (recommended): Streamlit Cloud or Railway for the app; Supabase for storing `clean_data`, forecasts and inventory plan (`run_id` column). Versions pinned in `requirements.txt`.

# 44. Product Success Metrics

**ML:** wMAPE, Bias, stability across months. **Inventory:** fill rate, stockout units, excess units at equal service level vs. baseline. **Product:** for any product a user understands within seconds how many units are expected, how much inventory is recommended, why, and how reliable the forecast is.

# 45. Model Champion Rules

See §20 (gates apply to all candidates including baselines; "no meaningful improvement" is a legitimate, transparently reported outcome).

# 46. Expected Findings (indicative)

SKU-level demand is very noisy (baseline wMAPE ≈ 55 %); many zero product-months; A products are far more predictable than C products (indicative wMAPE ≈ 45 % vs. ≈ 72 %); seasonal-naive fails at SKU level; Linear Regression does not beat the baseline; Gradient Boosting improves wMAPE modestly (≈ 1.5–2.5 points with unbiased losses; ≈ 5 points with absolute error but with −24 % bias); safety stock has a larger business impact than small accuracy gains; extreme orders dominate ordinary standard deviation, so robust σ is essential.

# 47. Risks

Extreme wholesale orders (robust σ, ABC reporting, anomaly diagnostics); intermittent demand (k = 6, global model, explicit zeros); product churn (active rule); **data leakage** (temporal features, temporal split, ABC and σ from past only, automated leakage tests); partial December 2011 (excluded); **left-censoring** — products first seen in December 2009 may be older, so `product_age_months` is a lower bound; only 24 months of history (one prior season for month-of-year effects); LLM non-determinism and cost (numbers only from code; `--no-llm` mode; cached prompts); team / timeline (milestones §54); GitHub size limits (no raw data in repo).

# 48. Ethical Considerations and Licensing

No decisions about individuals; Customer ID is not used; data are anonymised. Residual risks: blind reliance on forecasts, overstock from poor uncertainty estimates, stockouts from biased models, misinterpretation of safety stock. The app exposes metrics, uncertainty, limitations and the statement that recommendations rely only on historical sales in the dataset. Licensing: UCI dataset CC BY 4.0 — cite Chen (2019) / Chen, Sain & Guo (2012); the Kaggle mirror is CC0; derived artifacts keep the attribution.

# 49. MVP Acceptance Criteria

Raw data loads and hashes; cleaning waterfall reproducible; `clean_data.csv` generated; contract generated and validated by code; Product × Month panel with zero months; active rule k = 6; features without leakage (tests pass); baselines B1–B2 (+B3 reference) computed; ≥ 2 model variations trained (M1–M4); temporal hold-out and rolling-origin implemented; wMAPE and Bias reported overall / by month / by ABC; champion selected by the §20 rules; robust σ, safety stock, target inventory computed; inventory simulation for ML and baseline policies; Streamlit shows screens 1–7 and CSV download; `eda_report.html` contains at least the mandatory figures E1, E2, E5, E6, E7, E8, E9, E11 and `insights.md` has 8–12 insights each backed by a number from a computed table (§35A); CrewAI Flow performs the hand-off with validations and fails gracefully; all required artifacts saved with exact names; model card with the five required sections; evaluation report; README; PR-based history in GitHub; presentation (10–12 slides) and demo video (≤ 5 min) delivered.

# 50. Version 2 Roadmap

Quantile regression (P50 / P90 / P95 as direct inventory levels); direct multi-horizon models (t+1, t+2, t+3) for start-of-quarter planning; cold-start rule from the first 14 days; real inventory data (on hand, on order, backorders, lead times) → order quantities; cost optimisation (holding vs. stockout cost, margin); demand capping / wholesale-order flags in evaluation.

# 51. MVP in One Sentence

> A Streamlit application that analyses historical sales, uses one global machine-learning model to predict how many units each active product will sell next month, converts the forecast into a recommended target inventory using safety stock derived from out-of-sample forecast errors, and lets users evaluate product forecasts, model performance and the trade-off between stockouts and excess inventory.

# 52. Core Product Principle

Not "which model has the lowest wMAPE?" but **"which combination of demand forecast and inventory policy gives the best balance between accuracy, stockout risk and excess inventory?"**

---

# 53. Course Deliverables

**53.1 Repository** — as in §41–42.  
**53.2 Business presentation (10–12 slides):** 1 title & team; 2 business problem; 3 data & cleaning waterfall; 4 what happened (seasonality, concentration, lifecycle); 5 research question & two-layer design; 6 CrewAI Flow architecture; 7 model comparison (wMAPE + Bias table); 8 champion decision & by-ABC results; 9 inventory policy & back-test KPIs; 10 product demo screenshots; 11 limitations & ethics; 12 roadmap.  
**53.3 Demo recording (≤ 5 min):** 0:00 problem & data (30 s) → 0:30 run the Flow, show validations and logs (90 s) → 2:00 Streamlit: dashboard, product detail, evaluation, inventory policy (2 min) → 4:00 artifacts in the repo and model card (45 s) → 4:45 close.

# 54. Team Roles and Milestones (up to 5 students)

Roles: (1) Data & EDA lead (Crew 1 tools, contract); (2) ML lead (features, models, evaluation); (3) Flow / CrewAI lead (state, routers, logging, failure paths); (4) App lead (Streamlit, export, optional deployment); (5) PM / QA / docs (PR reviews, tests, model card, presentation, video). Milestones: M1 data pipeline + contract + EDA; M2 features + baselines + leakage tests; M3 models + evaluation + champion; M4 σ + inventory simulation + app; M5 Flow integration + failure handling + docs; M6 presentation + demo. Each milestone ends with a merged PR and a green CI run.

# 55. Testing Strategy

`tests/test_cleaning.py` (waterfall counts, exclusion list, duplicates), `test_panel.py` (zero-fill, no rows before first sale, partial flag), `test_contract.py` (validation passes / fails on corrupted samples), `test_features.py` (lag correctness, active rule, NaN policy), `test_leakage.py` (permutation of future months leaves features unchanged), `test_metrics.py` (wMAPE / Bias on known arrays), `test_inventory.py` (σ, safety stock, target, simulation), `test_flow.py` (graceful failure path produces `validation_report.json`), `test_determinism.py` (two runs identical).

# 56. Configuration Files

`cleaning_config.yaml` (cancellation prefixes, price/quantity rules, duplicate policy, warning thresholds); `model_config.yaml` (k, features list, split dates, models and hyper-parameters, seed, champion gates); `inventory_policy.yaml` (lead time, z, min residuals for σ, ABC thresholds); `non_inventory_stockcodes.csv` (code, reason).

# 57. Glossary

Forecast origin — last month whose data may be used. Active product — ≥ 1 sale in the previous k months. wMAPE — Σ|error| / Σ actual. Bias — Σ(forecast − actual) / Σ actual. Robust σ — 1.4826 × MAD of out-of-sample residuals. Fill rate — fulfilled / actual demand. Champion — the model selected by the §20 gates.

---

# Appendix A — `dataset_contract.json` skeleton

```json
{
  "dataset": "clean_data",
  "version": "1.0",
  "source": "UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci",
  "grain": "one row per stock_code x month, from first observed sale to 2011-11 (2011-12 flagged partial)",
  "primary_key": ["stock_code", "month"],
  "date_range": {"first_month": "2009-12", "last_full_month": "2011-11", "partial_months": ["2011-12"]},
  "columns": {
    "month": {"type": "string", "format": "YYYY-MM", "nullable": false},
    "stock_code": {"type": "string", "pattern": "^\\d{5}[A-Z]{0,2}$|^DCGS", "nullable": false},
    "description": {"type": "string", "nullable": true},
    "units_sold": {"type": "int", "min": 0, "nullable": false},
    "gross_revenue": {"type": "float", "min": 0}, "avg_unit_price": {"type": "float", "min": 0},
    "invoice_count": {"type": "int", "min": 0}, "sale_line_count": {"type": "int", "min": 0},
    "customer_count": {"type": "int", "min": 0}, "max_line_qty": {"type": "int", "min": 0},
    "returned_units": {"type": "int", "min": 0, "note": "EDA only - never a feature"},
    "is_partial_month": {"type": "bool"}
  },
  "cleaning_assumptions": ["cancellations (C/A invoices) removed", "Quantity <= 0 removed", "Price <= 0 removed",
                           "non-inventory stock codes removed (see config/non_inventory_stockcodes.csv)",
                           "exact duplicate rows removed", "rows without Customer ID kept", "gross demand (returns not subtracted)"],
  "active_rule": {"k": 6, "definition": ">= 1 positive sale in the 6 months before the target month"},
  "leakage_rules": ["features use months <= forecast_origin only", "target_month = forecast_origin + 1",
                    "ABC / sigma computed on training window only", "2011-12 never a target"],
  "modeling_split": {"train_targets": "2010-03..2011-05", "test_targets": "2011-06..2011-11"}
}
```

# Appendix B — Changes from v1.0 (review log)

1. **Course-brief coverage added:** deliverables (presentation outline, demo script), team roles and milestones, pull-request workflow and CI, optional deployment (Streamlit Cloud / Railway / Supabase), Matplotlib/Seaborn charts required in the EDA report, model-card mandatory sections, exact names of the eight required artifacts, logging specification, graceful-failure semantics.
2. **Data definitions completed:** source and licence; exact row counts, columns and period; Customer-ID-less rows explicitly kept; `Price ≤ 0` rule added; "A" adjustment invoices; StockCode normalisation; concrete non-inventory code list; duplicate warning threshold; abnormal-quantity policy (flag, do not remove); intermediate `clean_transactions.parquet`; new panel fields (`customer_count`, `max_line_qty`, `returned_units`).
3. **Modelling clarified:** forecast-origin definition and latest operational forecast (Nov 2011 → Dec 2011); NaN/zero policy for early lags; `features.csv` key and columns; added `rolling_std_3`, `rolling_max_6`, `invoice_count_lag_1`, `avg_unit_price_lag_1`; renamed `target_month` feature to `target_month_of_year`; Poisson-loss GBM added as primary (count target, unbiased) with squared error as second variation and absolute error as challenger; hyper-parameter tuning on rolling folds; post-hoc bias correction explicitly excluded from MVP after it proved unstable.
4. **Champion policy tightened:** gates apply to baselines too; per-month bias reporting; "meaningful improvement" threshold (≥ 2 wMAPE points); decision trace stored.
5. **Evaluation and inventory:** minimum residual count for product σ; residual-time ordering; back-test file name; policies compared on identical rows; excess-to-shortage ratio; expectations stated as indicative.
6. **Flow:** Pydantic state, routers, no-LLM mode, validation report artifact, permutation leakage test, determinism test.
7. **Risks added:** left-censoring, 24-month history, LLM non-determinism, GitHub limits. **Ethics:** licensing and attribution. **App:** new Pipeline & Data Quality screen; z slider; optional as-of selector.
8. **Consistency fixes:** scope statement now references the active rule instead of "sufficient history"; example numbers (820 / 116 / 936) aligned across sections; indicative numbers labelled and never hard-coded.
9. **Descriptive analytics & visualization (v1.2):** new §35A specifying 14 mandatory analyses (E1–E14) with required figures/tables, visualization standards (Matplotlib/Seaborn, single self-contained `eda_report.html` with embedded PNGs, tables as the only source of numbers for `insights.md`), where visuals appear (report, app screens 1/3/6/7, slides), a new Streamlit screen 7 "Data & Insights", and matching acceptance criteria.
10. **Final consistency check (v1.3):** non-inventory code list corrected to the 28 codes actually present (incl. `C3`, `GIFT`, `gift_0001_90`, `SP1002`) and the row count in §8 fixed accordingly; left-censoring convention for early lags and truncated rolling windows made explicit in §17; Sep–Nov share confirmed for units (≈ 35 %) as well as revenue.

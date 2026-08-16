# What the data says

*Dataset:* Online Retail II (UCI), 2009-12 to 2011-11 complete, plus the partial month 2011-12 (shown, never scored).

*Run:* `20260816T130244Z-fb02e1` · *Generated:* 2026-08-16T13:03:19.264019+00:00

*Provenance:* every number below was computed by the pipeline and written to a table under `artifacts/reports/eda_tables/`. The narrative may be rewritten by the Business & EDA Analyst agent in LLM mode, but the numbers may only ever come from those tables, and the same guard checks both versions (PRD §38).

*Active-product window:* 6 months. *First target month:* 2010-03.

## Insights

1. **September to November carries 35.5 % of the year's units** — so the hold-out window deliberately covers the peak — a model validated only on quiet months would be scored on data unlike the months that matter (§21). (E2, table `E02_sep_nov_share`)
2. **Between the two complete December-to-November years, units moved -6.7 % while revenue moved 2.5 %** — so volume and value are drifting apart, and a forecast of units cannot be read as a forecast of money. (E2, table `E02_yoy`)
3. **At the configured look-back of 6 months the model sees 72,182 product-months, 25.9 % of which sold nothing; widening the window to 12 months raises that to 32.4 %** — so the active-product window is a trade between coverage and empty targets, and this table is the justification for the configured value (§14). (E8, table `E08_zero_share_by_k`)
4. **Class A is 21.9 % of the catalogue and 80.0 % of revenue** — so accuracy on a small head of products decides the result, and the safety-stock fallback groups products by that same class when a product has too few residuals (§27). (E6, table `E06_abc_table`)
5. **24.4 % of products sell in 6 months or fewer, and the median product sells in 14 months** — so long history is the exception; the feature set leans on short lags rather than assuming two years of it (§18.1). (E5, table `E05_lifecycle_summary`)
6. **The largest single order line is 80,995 units, against a median product-month of 37 units** — so errors are weighted by volume (wMAPE) and the safety stock uses a robust spread measure — a standard deviation would let a few wholesale orders inflate the stock recommendation for every similar product (§23, §26). (E9, table `E09_largest_lines`)
7. **22.6 % of sales rows carry no customer id, and United Kingdom alone accounts for 82.0 % of units** — so anonymous rows are kept — the goods were still sold and still had to be in stock — and the model has no country dimension, because there is not enough of anywhere else to learn from (§9, §6.2). (E11, table `E11_customer_identification`)
8. **490,994 units came back against 11,188,039 sold — 4.4 % of the total** — so returns are visible but never subtracted from the target: the stock had to be on the shelf to fill the original order, so netting them off would under-forecast exactly the products that get returned (§9). (E12, table `E12_cancellation_rate_monthly`)
9. **The best-selling product, 84077 WORLD WAR 2 GLIDERS ASSTD DESIGNS, moved 104,772 units — 1.0 % of all units** — so no single product dominates volume; the ranking by revenue is a different list, which is why both are reported. (E7, table `E07_top20_units`)
10. **The number of products selling in a month ranges from 2,346 to 3,009** — so the catalogue is not stable, and forecasting every code every month would spend the model on products that are not selling — which is what the active-product rule prevents (§14). (E4, table `E04_products_per_month`)
11. **Half of all invoices carry 15 product lines or fewer, and 25.2 % of lines are wholesale-sized** — so this is a wholesaler rather than a shop, which is the context for the outsized lines above. (E10, table `E10_invoice_summary`)
12. **Saturday sees 5,119 units against 2,317,317 on Thursday** — so trading is a weekday business; the pattern is context for the reader and is never a model input, because the forecast is monthly (§35A E3). (E3, table `E03_weekday`)

## Figures

- `figures/E01_waterfall.png`
- `figures/E02_monthly_units.png`
- `figures/E02_seasonal_index.png`
- `figures/E03_hour.png`
- `figures/E03_weekday.png`
- `figures/E04_products_per_month.png`
- `figures/E05_lifecycle_hist.png`
- `figures/E06_pareto.png`
- `figures/E07_top20_revenue.png`
- `figures/E07_top20_units.png`
- `figures/E08_zero_share_by_k.png`
- `figures/E08_zero_share_hist.png`
- `figures/E09_units_hist_log.png`
- `figures/E10_invoice_hist.png`
- `figures/E11_top_countries.png`
- `figures/E12_cancellation_rate.png`
- `figures/E13_price_hist.png`

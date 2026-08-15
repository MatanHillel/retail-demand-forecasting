"""Clean Online Retail II with standard rules and test the feasibility of candidate research questions."""
import pandas as pd, numpy as np, json

df = pd.read_parquet('data/online_retail_II_raw.parquet')
R = {}

# ---------- cleaning ----------
is_cancel = df['Invoice'].str.upper().str.startswith('C')
is_product = df['StockCode'].str.match(r'^\d{5}[A-Za-z]{0,2}$') | df['StockCode'].str.startswith('DCGS')
steps = []
def log(name, mask_keep):
    global df
    before = len(df)
    df = df[mask_keep].copy()
    steps.append({'step': name, 'removed': before - len(df), 'remaining': len(df)})

log('drop cancellations (Invoice starts with C)', ~is_cancel.loc[df.index])
log('drop Quantity <= 0', df['Quantity'] > 0)
log('drop Price <= 0', df['Price'] > 0)
log('drop non-product stock codes (POST, M, D, DOT, C2, BANK CHARGES, ...)', is_product.loc[df.index])
log('drop exact duplicate rows', ~df.duplicated())
sales_all = df.copy()          # all sales incl. anonymous (for revenue analysis)
log('drop rows without CustomerID (for customer-level modelling)', df['CustomerID'].notna())
df['CustomerID'] = df['CustomerID'].astype(int)
df['LineTotal'] = df['Quantity'] * df['Price']
sales_all['LineTotal'] = sales_all['Quantity'] * sales_all['Price']
R['cleaning_steps'] = steps
R['clean_rows_identified'] = len(df)
R['clean_rows_all_sales'] = len(sales_all)
R['clean_customers'] = int(df['CustomerID'].nunique())
R['clean_invoices'] = int(df['Invoice'].nunique())
R['clean_products'] = int(df['StockCode'].nunique())
R['clean_revenue_identified_gbp'] = round(float(df['LineTotal'].sum()), 0)
R['clean_revenue_all_gbp'] = round(float(sales_all['LineTotal'].sum()), 0)
df.to_parquet('data/clean_identified.parquet', index=False)
sales_all.to_parquet('data/clean_all_sales.parquet', index=False)

# ---------- monthly seasonality ----------
m = sales_all.set_index('InvoiceDate').resample('MS').agg(revenue=('LineTotal', 'sum'), invoices=('Invoice', 'nunique'))
m['revenue'] = m['revenue'].round(0)
R['monthly'] = {str(k.date()): {'revenue': float(v.revenue), 'invoices': int(v.invoices)} for k, v in m.iterrows()}
# YoY on the 12 fully overlapping months Dec2010-Nov2011 vs Dec2009-Nov2010
y1 = m.loc['2009-12-01':'2010-11-01', 'revenue'].sum(); y2 = m.loc['2010-12-01':'2011-11-01', 'revenue'].sum()
R['yoy_revenue_growth_pct'] = round((y2 / y1 - 1) * 100, 1)
q4 = m[m.index.month.isin([9, 10, 11])]['revenue'].sum() / m['revenue'].sum()
R['sep_oct_nov_revenue_share_pct'] = round(q4 * 100, 1)
wk = sales_all.set_index('InvoiceDate').resample('W-SUN')['LineTotal'].sum()
R['n_weeks'] = int(len(wk))

# ---------- customer base ----------
cust = df.groupby('CustomerID').agg(first=('InvoiceDate', 'min'), last=('InvoiceDate', 'max'),
                                    n_inv=('Invoice', 'nunique'), revenue=('LineTotal', 'sum'),
                                    n_lines=('StockCode', 'size'), n_products=('StockCode', 'nunique'),
                                    qty=('Quantity', 'sum'), country=('Country', 'first'))
cust['tenure_days'] = (cust['last'] - cust['first']).dt.days
R['customers'] = int(len(cust))
R['one_time_customers_pct'] = round((cust['n_inv'] == 1).mean() * 100, 1)
R['orders_per_customer_describe'] = cust['n_inv'].describe(percentiles=[.25, .5, .75, .9, .99]).round(1).to_dict()
R['revenue_per_customer_describe'] = cust['revenue'].describe(percentiles=[.25, .5, .75, .9, .99]).round(0).to_dict()
srt = cust['revenue'].sort_values(ascending=False).cumsum() / cust['revenue'].sum()
R['top10pct_customers_revenue_share_pct'] = round(float(srt.iloc[int(len(srt) * 0.10) - 1]) * 100, 1)
R['top20pct_customers_revenue_share_pct'] = round(float(srt.iloc[int(len(srt) * 0.20) - 1]) * 100, 1)
R['nonuk_customers_pct'] = round((cust['country'] != 'United Kingdom').mean() * 100, 1)
R['nonuk_revenue_pct'] = round(cust.loc[cust['country'] != 'United Kingdom', 'revenue'].sum() / cust['revenue'].sum() * 100, 1)
avg_qty_line = (cust['qty'] / cust['n_lines'])
R['customers_avg_qty_per_line_gt_12_pct'] = round((avg_qty_line > 12).mean() * 100, 1)
# inter-purchase gap
inv_dates = df.groupby(['CustomerID', 'Invoice'])['InvoiceDate'].min().reset_index().sort_values(['CustomerID', 'InvoiceDate'])
inv_dates['gap'] = inv_dates.groupby('CustomerID')['InvoiceDate'].diff().dt.days
gaps = inv_dates['gap'].dropna()
R['interpurchase_gap_days_describe'] = gaps.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict()
# customers per country
R['customers_by_country_top10'] = cust['country'].value_counts().head(10).to_dict()

# ---------- RQ A: repurchase within N days after cut-off ----------
inv_c = inv_dates[['CustomerID', 'Invoice', 'InvoiceDate']].copy()
inv_c['date'] = inv_c['InvoiceDate'].dt.normalize()
inv_val = df.groupby('Invoice')['LineTotal'].sum()
inv_c['value'] = inv_c['Invoice'].map(inv_val)
END = pd.Timestamp('2011-12-09')
cutoffs = pd.date_range('2010-06-01', '2011-09-01', freq='3MS')  # quarterly cut-offs

def snapshot(cutoff, active_days, horizon):
    obs = inv_c[(inv_c['date'] < cutoff)]
    active = obs[obs['date'] >= cutoff - pd.Timedelta(days=active_days)]['CustomerID'].unique()
    fut = inv_c[(inv_c['date'] >= cutoff) & (inv_c['date'] < cutoff + pd.Timedelta(days=horizon))]
    buyers = set(fut['CustomerID'].unique())
    lab = pd.Series([c in buyers for c in active], index=active)
    spend = fut.groupby('CustomerID')['value'].sum().reindex(active).fillna(0)
    return lab, spend

rqa = []
for co in cutoffs:
    for active_days in (365, 180):
        for horizon in (60, 90, 180):
            if co + pd.Timedelta(days=horizon) > END + pd.Timedelta(days=1):
                continue
            lab, spend = snapshot(co, active_days, horizon)
            rqa.append({'cutoff': str(co.date()), 'active_window_days': active_days, 'horizon_days': horizon,
                        'n_active_customers': int(len(lab)), 'repurchase_rate_pct': round(lab.mean() * 100, 1),
                        'spend_zero_pct': round((spend == 0).mean() * 100, 1),
                        'spend_median_gbp': round(float(spend[spend > 0].median()), 0) if (spend > 0).any() else None,
                        'spend_p90_gbp': round(float(spend[spend > 0].quantile(.9)), 0) if (spend > 0).any() else None,
                        'spend_max_gbp': round(float(spend.max()), 0)})
R['rqA_repurchase_snapshots'] = rqa
# aggregate: total rows for the recommended design (active 365, horizon 90)
sel = [r for r in rqa if r['active_window_days'] == 365 and r['horizon_days'] == 90]
R['rqA_design_365_90_total_rows'] = int(sum(r['n_active_customers'] for r in sel))
R['rqA_design_365_90_mean_rate_pct'] = round(float(np.mean([r['repurchase_rate_pct'] for r in sel])), 1)

# ---------- RQ B: first-order -> second order (new customer conversion) ----------
first = inv_c.groupby('CustomerID')['date'].min()
second = inv_c.groupby('CustomerID')['date'].nth(1) if False else None
ordered = inv_c.sort_values(['CustomerID', 'date']).drop_duplicates(['CustomerID', 'date'])  # one per day
sec = ordered.groupby('CustomerID')['date'].nth(1)
sec = sec.reset_index().set_index('CustomerID')['date'] if isinstance(sec, pd.DataFrame) else sec
sec = ordered.groupby('CustomerID')['date'].apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
rqb = []
for horizon in (90, 180, 365):
    eligible = first[first <= END - pd.Timedelta(days=horizon)]
    conv = ((sec.reindex(eligible.index) - eligible).dt.days < horizon).fillna(False)
    rqb.append({'horizon_days': horizon, 'n_new_customers_eligible': int(len(eligible)),
                'converted_pct': round(conv.mean() * 100, 1)})
R['rqB_first_to_second_order'] = rqb
# customers whose first order is in the data (all, since data starts Dec 2009 - but Dec 2009 "first" may be pre-existing customers)
R['note_first_order'] = 'Customers first seen in Dec-2009 may be pre-existing customers (left-censoring); consider a 3-month burn-in.'
first_after_burnin = first[first >= pd.Timestamp('2010-03-01')]
elig = first_after_burnin[first_after_burnin <= END - pd.Timedelta(days=90)]
conv = ((sec.reindex(elig.index) - elig).dt.days < 90).fillna(False)
R['rqB_burnin_new_customers_90d'] = {'n': int(len(elig)), 'converted_pct': round(conv.mean() * 100, 1)}
# what does the first order look like (features available)
first_inv = inv_c.sort_values(['CustomerID', 'InvoiceDate']).groupby('CustomerID').head(1)
fi = df[df['Invoice'].isin(first_inv['Invoice'])].groupby('Invoice').agg(lines=('StockCode', 'size'), qty=('Quantity', 'sum'), value=('LineTotal', 'sum'))
R['first_order_value_describe'] = fi['value'].describe(percentiles=[.25, .5, .75, .9]).round(0).to_dict()
R['first_order_lines_describe'] = fi['lines'].describe(percentiles=[.25, .5, .75, .9]).round(0).to_dict()

# ---------- RQ C: demand forecasting feasibility ----------
prod_month = sales_all.assign(month=sales_all['InvoiceDate'].dt.to_period('M')).groupby(['StockCode', 'month'])['Quantity'].sum().unstack(fill_value=0)
n_months = prod_month.shape[1]
active_months = (prod_month > 0).sum(axis=1)
R['rqC_n_months'] = int(n_months)
R['rqC_products_total'] = int(len(active_months))
R['rqC_products_sold_in_ge_20_months'] = int((active_months >= 20).sum())
R['rqC_products_sold_in_ge_24_months'] = int((active_months >= 24).sum())
R['rqC_products_sold_in_le_6_months'] = int((active_months <= 6).sum())
top50 = sales_all.groupby('StockCode')['LineTotal'].sum().nlargest(50)
R['rqC_top50_products_revenue_share_pct'] = round(top50.sum() / sales_all['LineTotal'].sum() * 100, 1)
top500 = sales_all.groupby('StockCode')['LineTotal'].sum().nlargest(500)
R['rqC_top500_products_revenue_share_pct'] = round(top500.sum() / sales_all['LineTotal'].sum() * 100, 1)
# weekly total series stats
R['rqC_weekly_revenue_describe'] = wk.describe().round(0).to_dict()
# product-month coefficient of variation for top-50 products
cv = prod_month.loc[top50.index].apply(lambda r: r.std() / r.mean() if r.mean() > 0 else np.nan, axis=1)
R['rqC_top50_monthly_cv_median'] = round(float(cv.median()), 2)

# ---------- RQ D: returns / cancellations ----------
raw = pd.read_parquet('data/online_retail_II_raw.parquet')
canc = raw[raw['Invoice'].str.upper().str.startswith('C') & raw['CustomerID'].notna()].copy()
canc['CustomerID'] = canc['CustomerID'].astype(int)
R['rqD_cancel_rows_with_customer'] = int(len(canc))
R['rqD_customers_with_any_cancellation_pct'] = round(canc['CustomerID'].nunique() / df['CustomerID'].nunique() * 100, 1)
R['rqD_cancel_invoices_with_customer'] = int(canc['Invoice'].nunique())
R['rqD_cancel_invoices_per_100_purchase_invoices'] = round(canc['Invoice'].nunique() / df['Invoice'].nunique() * 100, 1)
R['rqD_returned_qty_share_of_sold_pct'] = round(-canc['Quantity'].sum() / df['Quantity'].sum() * 100, 2)
R['rqD_returned_value_share_pct'] = round(-(canc['Quantity'] * canc['Price']).sum() / df['LineTotal'].sum() * 100, 2)
# share of purchase invoices followed by ANY cancellation by the same customer within 30 days
c_dates = canc.groupby(['CustomerID', 'Invoice'])['InvoiceDate'].min().reset_index()
p = inv_c[['CustomerID', 'Invoice', 'InvoiceDate']].copy()
merged = p.merge(c_dates, on='CustomerID', suffixes=('', '_c'))
merged = merged[(merged['InvoiceDate_c'] >= merged['InvoiceDate']) & (merged['InvoiceDate_c'] <= merged['InvoiceDate'] + pd.Timedelta(days=30))]
R['rqD_purchase_invoices_followed_by_cancel_30d_pct'] = round(merged['Invoice'].nunique() / p['Invoice'].nunique() * 100, 1)
# line-level: purchase lines that have a matching cancellation (same customer, same stockcode) later
canc_keys = set(zip(canc['CustomerID'], canc['StockCode']))
R['rqD_purchase_lines_with_same_customer_product_cancel_pct'] = round(np.mean([k in canc_keys for k in zip(df['CustomerID'], df['StockCode'])]) * 100, 1)

# ---------- RQ E: basket / invoice value ----------
iv = df.groupby('Invoice').agg(value=('LineTotal', 'sum'), lines=('StockCode', 'size'), cust=('CustomerID', 'first'), date=('InvoiceDate', 'min'))
R['rqE_invoice_value_describe'] = iv['value'].describe(percentiles=[.25, .5, .75, .9, .99]).round(0).to_dict()
R['rqE_invoices_ge_1000_pct'] = round((iv['value'] >= 1000).mean() * 100, 1)

# ---------- RFM snapshot at end for descriptive crew ----------
ref = END + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(recency=('InvoiceDate', lambda s: (ref - s.max()).days), frequency=('Invoice', 'nunique'), monetary=('LineTotal', 'sum'))
R['rfm_describe'] = rfm.describe(percentiles=[.25, .5, .75]).round(1).to_dict()
R['customers_inactive_gt_180d_at_end_pct'] = round((rfm['recency'] > 180).mean() * 100, 1)
R['customers_inactive_gt_365d_at_end_pct'] = round((rfm['recency'] > 365).mean() * 100, 1)

# ---------- artifact size estimates ----------
import os
sales_all.to_csv('out/clean_all_sales_tmp.csv', index=False); df.to_csv('out/clean_identified_tmp.csv', index=False)
R['size_clean_all_sales_csv_mb'] = round(os.path.getsize('out/clean_all_sales_tmp.csv') / 1e6, 1)
R['size_clean_identified_csv_mb'] = round(os.path.getsize('out/clean_identified_tmp.csv') / 1e6, 1)
import gzip, shutil
with open('out/clean_identified_tmp.csv', 'rb') as f_in, gzip.open('out/clean_identified_tmp.csv.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
R['size_clean_identified_csv_gz_mb'] = round(os.path.getsize('out/clean_identified_tmp.csv.gz') / 1e6, 1)
df.to_parquet('out/clean_identified_tmp.parquet', index=False)
R['size_clean_identified_parquet_mb'] = round(os.path.getsize('out/clean_identified_tmp.parquet') / 1e6, 1)
for f in ['out/clean_all_sales_tmp.csv', 'out/clean_identified_tmp.csv', 'out/clean_identified_tmp.csv.gz', 'out/clean_identified_tmp.parquet']:
    os.remove(f)

json.dump(R, open('out/feasibility.json', 'w'), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))

"""Inventory-oriented feasibility checks: stock-adjustment rows, ABC/XYZ, SKU 'dead stock' base rate,
per-SKU monthly demand forecasting with a global model vs naive baselines, new-product initial demand."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

raw = pd.read_parquet('data/online_retail_II_raw.parquet')
sales = pd.read_parquet('data/clean_all_sales.parquet')
OUT = {}

# ---------- 1. stock-adjustment / shrinkage rows (not cancellations, no customer, qty<0 or price==0 with note) ----------
is_cancel = raw['Invoice'].str.upper().str.startswith('C')
adj = raw[~is_cancel & raw['CustomerID'].isna() & ((raw['Quantity'] < 0) | (raw['Price'] == 0))].copy()
adj['desc'] = adj['Description'].fillna('').str.lower().str.strip()
OUT['adjustment_rows'] = int(len(adj))
OUT['adjustment_rows_negative_qty'] = int((adj['Quantity'] < 0).sum())
OUT['adjustment_units_removed'] = int(-adj.loc[adj['Quantity'] < 0, 'Quantity'].sum())
OUT['adjustment_units_added'] = int(adj.loc[adj['Quantity'] > 0, 'Quantity'].sum())
kw = {'damaged/damages/smashed/crushed/wet': r'damag|smash|crush|wet|broken|faulty', 'missing/lost': r'missing|lost', 'found': r'found',
      'check/counted/adjust': r'check|count|adjust', 'thrown away/given away': r'thrown|given away|dumped', 'sold as set / dotcom / amazon / ebay': r'sold as|dotcom|amazon|ebay',
      'empty description': r'^$', 'question mark': r'\?'}
OUT['adjustment_reasons'] = {k: int(adj['desc'].str.contains(v, regex=True).sum()) for k, v in kw.items()}
OUT['adjustment_top_products_by_units_removed'] = (-adj[adj['Quantity'] < 0].groupby('StockCode')['Quantity'].sum()).nlargest(10).astype(int).to_dict()
OUT['adjustment_by_month'] = {str(k.date()): int(v) for k, v in adj.set_index('InvoiceDate').resample('QS')['Quantity'].count().items()}

# ---------- 2. ABC / XYZ ----------
prod = sales.groupby('StockCode').agg(revenue=('LineTotal', 'sum'), qty=('Quantity', 'sum'))
prod = prod.sort_values('revenue', ascending=False)
prod['cum_share'] = prod['revenue'].cumsum() / prod['revenue'].sum()
prod['ABC'] = np.where(prod['cum_share'] <= 0.8, 'A', np.where(prod['cum_share'] <= 0.95, 'B', 'C'))
pm = sales.assign(month=sales['InvoiceDate'].dt.to_period('M')).groupby(['StockCode', 'month'])['Quantity'].sum().unstack(fill_value=0)
pm = pm.loc[:, pm.columns.sort_values()]
full = pm.iloc[:, 1:-1]  # drop partial Dec-2009? Dec 2009 is full month; drop Dec 2011 (partial)
full = pm.loc[:, [c for c in pm.columns if str(c) != '2011-12']]
cv = full.std(axis=1) / full.mean(axis=1).replace(0, np.nan)
prod['cv'] = cv.reindex(prod.index)
prod['XYZ'] = pd.cut(prod['cv'], bins=[-1, 0.5, 1.0, np.inf], labels=['X', 'Y', 'Z'])
OUT['abc_counts'] = prod['ABC'].value_counts().to_dict()
OUT['abc_xyz_matrix'] = prod.groupby(['ABC', 'XYZ']).size().unstack(fill_value=0).astype(int).to_dict()
OUT['A_items_share_of_skus_pct'] = round((prod['ABC'] == 'A').mean() * 100, 1)

# ---------- 3. SKU "dead stock" / discontinuation base rate ----------
pq = sales.assign(q=sales['InvoiceDate'].dt.to_period('Q')).groupby(['StockCode', 'q'])['Quantity'].sum().unstack(fill_value=0)
qs = list(pq.columns)
rows = []
for i in range(1, len(qs) - 1):  # need previous & next quarter; skip first and last (partial) quarter
    cur, nxt, prev = qs[i], qs[i + 1], qs[i - 1]
    active = pq[pq[cur] > 0]
    dead = (active[nxt] == 0)
    low = (active[nxt] < 0.25 * active[cur])
    rows.append({'quarter': str(cur), 'active_skus': int(len(active)), 'zero_next_q_pct': round(dead.mean() * 100, 1), 'drop_75pct_next_q_pct': round(low.mean() * 100, 1)})
OUT['sku_dead_stock_by_quarter'] = rows
OUT['sku_dead_stock_total_rows'] = int(sum(r['active_skus'] for r in rows))
OUT['sku_dead_stock_mean_zero_pct'] = round(float(np.mean([r['zero_next_q_pct'] for r in rows])), 1)

# ---------- 4. per-SKU monthly demand forecasting: global model vs naive ----------
months = [c for c in pm.columns if str(c) != '2011-12']
M = pm[months]
top = prod[prod['ABC'].isin(['A', 'B'])].index  # A+B items
M = M.loc[M.index.intersection(top)]
recs = []
for sku, r in M.iterrows():
    v = r.values.astype(float)
    for t in range(12, len(v)):
        recs.append({'sku': sku, 't': t, 'y': v[t], 'lag1': v[t - 1], 'lag2': v[t - 2], 'lag3': v[t - 3], 'lag12': v[t - 12],
                     'ma3': v[t - 3:t].mean(), 'ma6': v[t - 6:t].mean(), 'max6': v[t - 6:t].max(), 'nz6': (v[t - 6:t] > 0).sum(),
                     'month': months[t].month})
D = pd.DataFrame(recs)
D = D[D['ma6'] > 0]  # only SKUs alive in the last 6 months
test_t = sorted(D['t'].unique())[-6:]  # last 6 months as test (Jun-Nov 2011)
tr, te = D[~D['t'].isin(test_t)], D[D['t'].isin(test_t)]
feats = ['lag1', 'lag2', 'lag3', 'lag12', 'ma3', 'ma6', 'max6', 'nz6', 'month']
OUT['sku_forecast_train_rows'] = int(len(tr)); OUT['sku_forecast_test_rows'] = int(len(te)); OUT['sku_forecast_n_skus'] = int(D['sku'].nunique())
def wmape(y, p): return round(float(np.abs(y - p).sum() / y.sum() * 100), 1)
res = {'naive_last_month_wMAPE': wmape(te['y'], te['lag1']), 'naive_ma3_wMAPE': wmape(te['y'], te['ma3']),
       'seasonal_naive_lag12_wMAPE': wmape(te['y'], te['lag12'])}
ridge = Ridge(alpha=1.0).fit(tr[feats], tr['y']); res['ridge_wMAPE'] = wmape(te['y'], np.clip(ridge.predict(te[feats]), 0, None))
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42).fit(tr[feats], np.log1p(tr['y']))
res['hgb_log_wMAPE'] = wmape(te['y'], np.expm1(hgb.predict(te[feats])).clip(min=0))
hgb2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, loss='absolute_error', random_state=42).fit(tr[feats], tr['y'])
res['hgb_mae_wMAPE'] = wmape(te['y'], hgb2.predict(te[feats]).clip(min=0))
OUT['sku_forecast'] = res

# ---------- 5. new-product initial demand: predict first-90-day units from first-14-day units ----------
first_sale = sales.groupby('StockCode')['InvoiceDate'].min()
new = first_sale[(first_sale >= '2010-03-01') & (first_sale <= pd.Timestamp('2011-12-09') - pd.Timedelta(days=90))]
s2 = sales[sales['StockCode'].isin(new.index)].copy()
s2['age'] = (s2['InvoiceDate'] - s2['StockCode'].map(first_sale)).dt.days
u14 = s2[s2['age'] < 14].groupby('StockCode')['Quantity'].sum().reindex(new.index).fillna(0)
u90 = s2[s2['age'] < 90].groupby('StockCode')['Quantity'].sum().reindex(new.index).fillna(0)
OUT['new_products_n'] = int(len(new))
OUT['new_products_first90_units_describe'] = u90.describe(percentiles=[.25, .5, .75, .9]).round(0).to_dict()
OUT['new_products_corr_log_u14_u90'] = round(float(np.corrcoef(np.log1p(u14), np.log1p(u90))[0, 1]), 3)
hit = (u90 >= u90.quantile(0.75)).astype(int)
from sklearn.metrics import roc_auc_score
OUT['new_products_auc_top_quartile_from_u14'] = round(roc_auc_score(hit, u14), 3)

json.dump(OUT, open('out/inventory.json', 'w'), indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))

"""EDA for the MVP Product x Month table: universe rule (active if >=1 sale in last k months),
row counts, zero-target share, naive baselines, and a first Naive -> Linear -> GBM comparison
on the exact table spec proposed for the project. Also per-product sigma for the safety-stock rule."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sales = pd.read_parquet('data/clean_all_sales.parquet')      # all positive sales incl. anonymous customers
sales['month'] = sales['InvoiceDate'].dt.to_period('M')
months = [m for m in sorted(sales['month'].unique()) if str(m) != '2011-12']   # 24 full months Dec-2009..Nov-2011
OUT = {'n_full_months': len(months), 'first_month': str(months[0]), 'last_month': str(months[-1])}

# product x month aggregates
g = sales[sales['month'].isin(months)].groupby(['StockCode', 'month'])
agg = g.agg(units=('Quantity', 'sum'), revenue=('LineTotal', 'sum'), n_invoices=('Invoice', 'nunique'),
            n_customers=('CustomerID', lambda s: s.dropna().nunique()), n_lines=('Quantity', 'size'),
            price_median=('Price', 'median'), max_line_qty=('Quantity', 'max')).reset_index()
units = agg.pivot(index='StockCode', columns='month', values='units').reindex(columns=months).fillna(0)
inv = agg.pivot(index='StockCode', columns='month', values='n_invoices').reindex(columns=months).fillna(0)
cust = agg.pivot(index='StockCode', columns='month', values='n_customers').reindex(columns=months).fillna(0)
price = agg.pivot(index='StockCode', columns='month', values='price_median').reindex(columns=months)
first_month = units.apply(lambda r: r[r > 0].index[0], axis=1)
OUT['n_products'] = int(len(units))
OUT['product_month_cells'] = int(units.size)
OUT['nonzero_cells_pct'] = round(float((units > 0).mean().mean() * 100), 1)

# outliers in monthly units
OUT['largest_single_line_qty'] = int(sales['Quantity'].max())
OUT['product_months_units_gt_10000'] = int((units > 10000).sum().sum())
OUT['units_describe_nonzero'] = units[units > 0].stack().describe(percentiles=[.5, .9, .99]).round(0).to_dict()

# ---------- build the Product x Month feature table for a given k ----------
def build(k):
    rows = []
    U, I, C, P = units.values, inv.values, cust.values, price.values
    idx = units.index
    for j in range(3, len(months) - 1):          # need 3 lags; target = month j+1
        active = (U[:, max(0, j - k + 1):j + 1] > 0).any(axis=1)     # >=1 sale in last k months (incl. month j)
        sel = np.where(active)[0]
        for i in sel:
            u = U[i]
            rows.append({'StockCode': idx[i], 'month': str(months[j]), 'target_month': str(months[j + 1]),
                         'lag1': u[j], 'lag2': u[j - 1], 'lag3': u[j - 2], 'ma3': u[j - 2:j + 1].mean(),
                         'ma6': u[max(0, j - 5):j + 1].mean(), 'std3': u[j - 2:j + 1].std(), 'max3': u[j - 2:j + 1].max(),
                         'months_with_sales_6': (u[max(0, j - 5):j + 1] > 0).sum(),
                         'inv_lag1': I[i, j], 'cust_lag1': C[i, j],
                         'price_lag1': P[i, j] if not np.isnan(P[i, j]) else np.nanmedian(P[i, :j + 1]),
                         'age_months': (months[j] - first_month[idx[i]]).n,
                         'month_of_year': months[j + 1].month, 'quarter': months[j + 1].quarter,
                         'target': u[j + 1]})
    return pd.DataFrame(rows)

def wmape(y, p): return round(float(np.abs(np.asarray(y) - np.asarray(p)).sum() / np.asarray(y).sum() * 100), 1)
def bias(y, p): return round(float((np.asarray(p) - np.asarray(y)).sum() / np.asarray(y).sum() * 100), 1)

feats = ['lag1', 'lag2', 'lag3', 'ma3', 'ma6', 'std3', 'max3', 'months_with_sales_6', 'inv_lag1', 'cust_lag1', 'price_lag1', 'age_months', 'month_of_year', 'quarter']
TEST_FROM = '2011-06'   # test targets Jun-2011 .. Nov-2011 (6 months incl. peak season)
res_k = {}
for k in (1, 3, 6, 12):
    D = build(k)
    tr, te = D[D['target_month'] < TEST_FROM], D[D['target_month'] >= TEST_FROM]
    r = {'rows_total': len(D), 'rows_train': len(tr), 'rows_test': len(te), 'products': int(D['StockCode'].nunique()),
         'zero_target_pct': round(float((D['target'] == 0).mean() * 100), 1),
         'naive_last_month_wMAPE': wmape(te['target'], te['lag1']), 'naive_ma3_wMAPE': wmape(te['target'], te['ma3'])}
    res_k[f'k={k}'] = r
OUT['universe_rule_by_k'] = res_k

# ---------- full comparison on k=6 (recommended MVP universe) ----------
D = build(6)
D.to_csv('out/features_product_month_k6.csv', index=False)
tr, te = D[D['target_month'] < TEST_FROM], D[D['target_month'] >= TEST_FROM]
cmp = {'rows_train': len(tr), 'rows_test': len(te)}
cmp['naive_last_month'] = {'wMAPE': wmape(te['target'], te['lag1']), 'bias_pct': bias(te['target'], te['lag1'])}
cmp['naive_ma3'] = {'wMAPE': wmape(te['target'], te['ma3']), 'bias_pct': bias(te['target'], te['ma3'])}
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[feats], tr['target'])
p = lin.predict(te[feats]).clip(min=0); cmp['linear_regression_clipped'] = {'wMAPE': wmape(te['target'], p), 'bias_pct': bias(te['target'], p)}
p_raw = lin.predict(te[feats]); cmp['linear_regression_negative_preds_pct'] = round(float((p_raw < 0).mean() * 100), 1)
pois = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)).fit(tr[feats], tr['target'])
p = pois.predict(te[feats]); cmp['poisson_regression'] = {'wMAPE': wmape(te['target'], p), 'bias_pct': bias(te['target'], p)}
for name, m in {'hgb_absolute_error': HistGradientBoostingRegressor(loss='absolute_error', max_iter=400, learning_rate=0.05, random_state=42),
                'hgb_poisson': HistGradientBoostingRegressor(loss='poisson', max_iter=400, learning_rate=0.05, random_state=42),
                'hgb_squared_error': HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42)}.items():
    m.fit(tr[feats], tr['target']); p = m.predict(te[feats]).clip(min=0)
    cmp[name] = {'wMAPE': wmape(te['target'], p), 'bias_pct': bias(te['target'], p)}
# error by month (peak season vs rest) for best HGB
m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=400, learning_rate=0.05, random_state=42).fit(tr[feats], tr['target'])
te = te.assign(pred=m.predict(te[feats]).clip(min=0))
cmp['hgb_abs_by_target_month'] = {mo: {'wMAPE': wmape(d['target'], d['pred']), 'naive_ma3': wmape(d['target'], d['ma3'])} for mo, d in te.groupby('target_month')}
# error by volume band (ABC on training revenue)
rev = sales.groupby('StockCode')['LineTotal'].sum().sort_values(ascending=False); cum = rev.cumsum() / rev.sum()
abc = pd.Series(np.where(cum <= 0.8, 'A', np.where(cum <= 0.95, 'B', 'C')), index=rev.index)
te = te.assign(abc=te['StockCode'].map(abc))
cmp['hgb_abs_by_abc'] = {a: {'rows': len(d), 'wMAPE': wmape(d['target'], d['pred']), 'naive_ma3': wmape(d['target'], d['ma3'])} for a, d in te.groupby('abc')}
OUT['comparison_k6'] = cmp

# ---------- safety-stock inputs: per-product sigma of out-of-sample error (from a rolling backtest on train months) ----------
# simple version: sigma_p = std of (target - ma3) over the training rows of product p (>=4 rows), used with z=1.645
sig = tr.groupby('StockCode').apply(lambda d: (d['target'] - d['ma3']).std() if len(d) >= 4 else np.nan)
OUT['sigma_available_products'] = int(sig.notna().sum())
# backtest of inventory policy on test months: stock = forecast + z*sigma ; shortage = max(actual - stock, 0); excess = max(stock - actual, 0)
z = 1.645
te = te.assign(sigma=te['StockCode'].map(sig))
bt = te.dropna(subset=['sigma']).copy()
def policy(pred_col):
    stock = bt[pred_col] + z * bt['sigma']
    short = (bt['target'] - stock).clip(lower=0); exc = (stock - bt['target']).clip(lower=0)
    return {'rows': len(bt), 'shortage_units': int(short.sum()), 'excess_units': int(exc.sum()),
            'fill_rate_pct': round(float(1 - short.sum() / bt['target'].sum()) * 100, 1),
            'stockout_events_pct': round(float((short > 0).mean() * 100), 1)}
OUT['inventory_backtest_z1645'] = {'ml_policy': policy('pred'), 'naive_ma3_policy': policy('ma3'), 'naive_last_month_policy': policy('lag1')}
def policy_nosafety(pred_col):
    stock = bt[pred_col]
    short = (bt['target'] - stock).clip(lower=0); exc = (stock - bt['target']).clip(lower=0)
    return {'shortage_units': int(short.sum()), 'excess_units': int(exc.sum()), 'fill_rate_pct': round(float(1 - short.sum() / bt['target'].sum()) * 100, 1)}
OUT['inventory_backtest_no_safety_stock'] = {'ml_policy': policy_nosafety('pred'), 'naive_ma3_policy': policy_nosafety('ma3')}

json.dump(OUT, open('out/mvp_table_eda.json', 'w'), indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))

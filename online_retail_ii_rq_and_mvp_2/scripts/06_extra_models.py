"""Extra checks on the MVP table (out/features_product_month_k6.csv):
log-linear regression variants, HGB on log target, quantile-regression inventory policies,
and naive + robust (MAD) safety stock — used for the implementation tips in mvp_handoff.md."""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

D = pd.read_csv('out/features_product_month_k6.csv')
feats = ['lag1', 'lag2', 'lag3', 'ma3', 'ma6', 'std3', 'max3', 'months_with_sales_6', 'inv_lag1', 'cust_lag1', 'price_lag1', 'age_months', 'month_of_year', 'quarter']
tr, te = D[D['target_month'] < '2011-06'], D[D['target_month'] >= '2011-06']
def wmape(y, p): return round(float(np.abs(np.asarray(y) - np.asarray(p)).sum() / np.asarray(y).sum() * 100), 1)
def bias(y, p): return round(float((np.asarray(p) - np.asarray(y)).sum() / np.asarray(y).sum() * 100), 1)
OUT = {}

def logf(X):
    X = X.copy()
    for c in ['lag1', 'lag2', 'lag3', 'ma3', 'ma6', 'std3', 'max3', 'inv_lag1', 'cust_lag1', 'price_lag1']:
        X[c] = np.log1p(X[c])
    return X

lin = make_pipeline(StandardScaler(), LinearRegression()).fit(logf(tr[feats]), np.log1p(tr['target']))
p = np.expm1(lin.predict(logf(te[feats]))).clip(0, None)
OUT['linreg_log_log'] = {'wMAPE': wmape(te.target, p), 'bias': bias(te.target, p)}
resid = np.log1p(tr['target']) - lin.predict(logf(tr[feats])); smear = float(np.mean(np.exp(resid)))
p3 = (np.expm1(lin.predict(logf(te[feats]))) * smear).clip(0, None)
OUT['linreg_log_log_smeared'] = {'wMAPE': wmape(te.target, p3), 'bias': bias(te.target, p3), 'smear': round(smear, 2)}
rd = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(logf(tr[feats]), np.log1p(tr['target']))
p4 = np.expm1(rd.predict(logf(te[feats]))).clip(0, None)
OUT['ridge_log_log'] = {'wMAPE': wmape(te.target, p4), 'bias': bias(te.target, p4)}
h = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42).fit(tr[feats], np.log1p(tr['target']))
p5 = np.expm1(h.predict(te[feats])).clip(0, None)
OUT['hgb_log_target'] = {'wMAPE': wmape(te.target, p5), 'bias': bias(te.target, p5)}

for q in (0.8, 0.9, 0.95):
    hq = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=400, learning_rate=0.05, random_state=42).fit(tr[feats], tr['target'])
    stock = hq.predict(te[feats]).clip(min=0)
    short = (te['target'] - stock).clip(lower=0); exc = (stock - te['target']).clip(lower=0)
    OUT[f'quantile_{q}_policy'] = {'shortage_units': int(short.sum()), 'excess_units': int(exc.sum()),
                                   'fill_rate_pct': round(float(1 - short.sum() / te['target'].sum()) * 100, 1),
                                   'stockout_events_pct': round(float((short > 0).mean() * 100), 1)}

sig = tr.groupby('StockCode').apply(lambda d: 1.4826 * np.median(np.abs((d['target'] - d['ma3']) - np.median(d['target'] - d['ma3']))) if len(d) >= 4 else np.nan)
t2 = te.assign(sig=te['StockCode'].map(sig)).dropna(subset=['sig'])
for z in (1.645, 2.5):
    stock = t2['ma3'] + z * t2['sig']; short = (t2['target'] - stock).clip(lower=0); exc = (stock - t2['target']).clip(lower=0)
    OUT[f'ma3_plus_{z}_robust_sigma'] = {'rows': len(t2), 'shortage_units': int(short.sum()), 'excess_units': int(exc.sum()),
                                         'fill_rate_pct': round(float(1 - short.sum() / t2['target'].sum()) * 100, 1)}
json.dump(OUT, open('out/mvp_extra_models.json', 'w'), indent=2)
print(json.dumps(OUT, indent=2))

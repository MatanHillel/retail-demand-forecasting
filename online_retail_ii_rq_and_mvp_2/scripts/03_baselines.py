"""Quick baseline models to show the candidate research questions are answerable (signal exists).
RQ1: 90-day repurchase for customers active in the last 365 days (temporal split by cut-off).
RQ2: new-customer conversion (2nd order within 90 days) from first-order features only.
RQ3: next-180-day spend regression (log1p) for active customers.
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_parquet('data/clean_identified.parquet')
df['date'] = df['InvoiceDate'].dt.normalize()
raw = pd.read_parquet('data/online_retail_II_raw.parquet')
canc = raw[raw['Invoice'].str.upper().str.startswith('C') & raw['CustomerID'].notna()].copy()
canc['CustomerID'] = canc['CustomerID'].astype(int); canc['date'] = canc['InvoiceDate'].dt.normalize()
END = pd.Timestamp('2011-12-09')
OUT = {}

inv = df.groupby(['CustomerID', 'Invoice']).agg(date=('date', 'min'), value=('LineTotal', 'sum'), lines=('StockCode', 'size'),
                                                qty=('Quantity', 'sum'), nprod=('StockCode', 'nunique'), country=('Country', 'first')).reset_index()

def features_at(cutoff, active_days=365):
    obs = inv[inv['date'] < cutoff]
    act_ids = obs[obs['date'] >= cutoff - pd.Timedelta(days=active_days)]['CustomerID'].unique()
    o = obs[obs['CustomerID'].isin(act_ids)]
    g = o.groupby('CustomerID')
    f = pd.DataFrame({
        'recency': (cutoff - g['date'].max()).dt.days,
        'tenure': (cutoff - g['date'].min()).dt.days,
        'frequency': g['Invoice'].nunique(),
        'monetary': g['value'].sum(),
        'avg_order_value': g['value'].mean(),
        'avg_lines': g['lines'].mean(),
        'avg_qty_per_line': g['qty'].sum() / g['lines'].sum(),
        'n_products': g['nprod'].sum(),
        'last90_orders': o[o['date'] >= cutoff - pd.Timedelta(days=90)].groupby('CustomerID')['Invoice'].nunique().reindex(act_ids).fillna(0).values,
        'last90_spend': o[o['date'] >= cutoff - pd.Timedelta(days=90)].groupby('CustomerID')['value'].sum().reindex(act_ids).fillna(0).values,
        'is_uk': (g['country'].first() == 'United Kingdom').astype(int),
        'q4_share': o.assign(q4=o['date'].dt.month.isin([9, 10, 11])).groupby('CustomerID')['q4'].mean(),
    })
    f['freq_per_month'] = f['frequency'] / (f['tenure'].clip(lower=30) / 30)
    c = canc[(canc['date'] < cutoff)].groupby('CustomerID')['Invoice'].nunique().reindex(f.index).fillna(0)
    f['n_cancel_invoices'] = c.values
    f['cutoff_month'] = cutoff.month
    return f

def label_at(cutoff, ids, horizon=90):
    fut = inv[(inv['date'] >= cutoff) & (inv['date'] < cutoff + pd.Timedelta(days=horizon))]
    buyers = set(fut['CustomerID'])
    y = pd.Series([int(i in buyers) for i in ids], index=ids)
    spend = fut.groupby('CustomerID')['value'].sum().reindex(ids).fillna(0)
    return y, spend

cutoffs = pd.date_range('2010-06-01', '2011-09-01', freq='3MS')
frames = []
for co in cutoffs:
    f = features_at(co)
    y, s = label_at(co, f.index, 90)
    y180, s180 = label_at(co, f.index, 180) if co + pd.Timedelta(days=180) <= END + pd.Timedelta(days=1) else (None, None)
    f['y90'] = y.values; f['spend90'] = s.values
    f['spend180'] = s180.values if s180 is not None else np.nan
    f['cutoff'] = co
    frames.append(f)
D = pd.concat(frames)
feat_cols = ['recency', 'tenure', 'frequency', 'monetary', 'avg_order_value', 'avg_lines', 'avg_qty_per_line', 'n_products',
             'last90_orders', 'last90_spend', 'is_uk', 'q4_share', 'freq_per_month', 'n_cancel_invoices', 'cutoff_month']
train = D[D['cutoff'] <= '2011-03-01']; test = D[D['cutoff'] >= '2011-06-01']
OUT['rq1_train_rows'] = int(len(train)); OUT['rq1_test_rows'] = int(len(test))
OUT['rq1_train_rate'] = round(train['y90'].mean(), 3); OUT['rq1_test_rate'] = round(test['y90'].mean(), 3)
models = {
    'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42),
}
res = {}
for name, m in models.items():
    m.fit(train[feat_cols], train['y90'])
    p = m.predict_proba(test[feat_cols])[:, 1]
    # precision at top 20% most likely to churn (lowest p) -> churn precision
    k = int(len(p) * 0.2); idx = np.argsort(p)[:k]
    churn_prec = 1 - test['y90'].values[idx].mean()
    res[name] = {'roc_auc': round(roc_auc_score(test['y90'], p), 3), 'pr_auc_repurchase': round(average_precision_score(test['y90'], p), 3),
                 'churn_precision_at_top20pct': round(churn_prec, 3), 'base_churn_rate': round(1 - test['y90'].mean(), 3)}
    if name == 'hist_gradient_boosting':
        pi = permutation_importance(m, test[feat_cols], test['y90'], scoring='roc_auc', n_repeats=5, random_state=0)
        res[name]['top_features'] = dict(sorted(zip(feat_cols, pi.importances_mean.round(4)), key=lambda x: -x[1])[:8])
OUT['rq1_models'] = res
# recency-only heuristic baseline
p_h = -test['recency'].values
OUT['rq1_recency_only_auc'] = round(roc_auc_score(test['y90'], p_h), 3)

# ---------- RQ3: 180-day spend regression (log1p) among active customers ----------
tr = train.dropna(subset=['spend180']); te = test.dropna(subset=['spend180'])
OUT['rq3_train_rows'] = int(len(tr)); OUT['rq3_test_rows'] = int(len(te))
reg = {'ridge_log': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
       'hgb_log': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)}
r3 = {}
for name, m in reg.items():
    m.fit(tr[feat_cols], np.log1p(tr['spend180']))
    pred = np.expm1(m.predict(te[feat_cols])).clip(min=0)
    r3[name] = {'MAE_gbp': round(mean_absolute_error(te['spend180'], pred), 0),
                'R2_log_scale': round(r2_score(np.log1p(te['spend180']), np.log1p(pred)), 3),
                'spearman': round(pd.Series(pred).corr(te['spend180'].reset_index(drop=True), method='spearman'), 3)}
r3['naive_last180_spend_MAE_gbp'] = round(mean_absolute_error(te['spend180'], te['last90_spend'] * 2), 0)
r3['spend180_test_describe'] = te['spend180'].describe(percentiles=[.5, .9, .99]).round(0).to_dict()
OUT['rq3_models'] = r3

# ---------- RQ2: new-customer conversion from first order only ----------
first_inv = inv.sort_values(['CustomerID', 'date']).groupby('CustomerID').head(1).set_index('CustomerID')
first_lines = df.merge(first_inv[['Invoice']].reset_index(), on=['CustomerID', 'Invoice'])
g = first_lines.groupby('CustomerID')
F2 = pd.DataFrame({
    'first_value': g['LineTotal'].sum(), 'first_lines': g['StockCode'].size(), 'first_qty': g['Quantity'].sum(),
    'first_nprod': g['StockCode'].nunique(), 'first_avg_price': g['Price'].mean(), 'first_max_qty_line': g['Quantity'].max(),
    'first_month': g['InvoiceDate'].min().dt.month, 'first_dow': g['InvoiceDate'].min().dt.dayofweek, 'first_hour': g['InvoiceDate'].min().dt.hour,
    'is_uk': (g['Country'].first() == 'United Kingdom').astype(int),
})
F2['first_date'] = first_inv['date']
sec = inv.sort_values(['CustomerID', 'date']).drop_duplicates(['CustomerID', 'date']).groupby('CustomerID')['date'].apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
F2['second_date'] = sec
F2 = F2[(F2['first_date'] >= '2010-03-01') & (F2['first_date'] <= END - pd.Timedelta(days=90))]  # burn-in + full horizon
F2['y'] = ((F2['second_date'] - F2['first_date']).dt.days < 90).fillna(False).astype(int)
f2cols = ['first_value', 'first_lines', 'first_qty', 'first_nprod', 'first_avg_price', 'first_max_qty_line', 'first_month', 'first_dow', 'first_hour', 'is_uk']
tr2 = F2[F2['first_date'] < '2011-03-01']; te2 = F2[F2['first_date'] >= '2011-03-01']
OUT['rq2_train_rows'] = int(len(tr2)); OUT['rq2_test_rows'] = int(len(te2)); OUT['rq2_rate_train'] = round(tr2['y'].mean(), 3); OUT['rq2_rate_test'] = round(te2['y'].mean(), 3)
r2 = {}
for name, m in {'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)}.items():
    m.fit(tr2[f2cols], tr2['y']); p = m.predict_proba(te2[f2cols])[:, 1]
    r2[name] = {'roc_auc': round(roc_auc_score(te2['y'], p), 3), 'pr_auc': round(average_precision_score(te2['y'], p), 3)}
    if name == 'hist_gradient_boosting':
        pi = permutation_importance(m, te2[f2cols], te2['y'], scoring='roc_auc', n_repeats=5, random_state=0)
        r2[name]['top_features'] = dict(sorted(zip(f2cols, pi.importances_mean.round(4)), key=lambda x: -x[1])[:6])
OUT['rq2_models'] = r2

# ---------- RQ4: returns - customer-level "any cancellation in next 90 days" among active customers ----------
def cancel_label(cutoff, ids, horizon=90):
    fut = canc[(canc['date'] >= cutoff) & (canc['date'] < cutoff + pd.Timedelta(days=horizon))]
    s = set(fut['CustomerID']); return pd.Series([int(i in s) for i in ids], index=ids)
D['y_cancel90'] = np.concatenate([cancel_label(co, D[D['cutoff'] == co].index).values for co in cutoffs])
train = D[D['cutoff'] <= '2011-03-01']; test = D[D['cutoff'] >= '2011-06-01']
OUT['rq4_customer_cancel90_rate_train'] = round(train['y_cancel90'].mean(), 3)
OUT['rq4_customer_cancel90_rate_test'] = round(test['y_cancel90'].mean(), 3)
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42).fit(train[feat_cols], train['y_cancel90'])
p = m.predict_proba(test[feat_cols])[:, 1]
OUT['rq4_hgb_auc'] = round(roc_auc_score(test['y_cancel90'], p), 3); OUT['rq4_hgb_pr_auc'] = round(average_precision_score(test['y_cancel90'], p), 3)

# ---------- RQ5: weekly revenue forecasting quick check ----------
sales_all = pd.read_parquet('data/clean_all_sales.parquet')
wk = sales_all.set_index('InvoiceDate').resample('W-SUN')['LineTotal'].sum()
wk = wk.iloc[1:-1]  # drop partial first/last weeks
s = wk.copy()
X = pd.DataFrame({'lag1': s.shift(1), 'lag2': s.shift(2), 'lag4': s.shift(4), 'lag52': s.shift(52), 'ma4': s.shift(1).rolling(4).mean(),
                  'weekofyear': s.index.isocalendar().week.values.astype(int)}).dropna()
y = s.loc[X.index]
split = X.index[int(len(X) * 0.75)]
Xtr, Xte, ytr, yte = X[X.index < split], X[X.index >= split], y[y.index < split], y[y.index >= split]
OUT['rq5_weeks_usable_with_lag52'] = int(len(X)); OUT['rq5_test_weeks'] = int(len(Xte))
r5 = {}
r5['seasonal_naive_lag52_MAPE'] = round(float((abs(Xte['lag52'] - yte) / yte).mean() * 100), 1)
r5['naive_lag1_MAPE'] = round(float((abs(Xte['lag1'] - yte) / yte).mean() * 100), 1)
for name, m in {'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)), 'hgb': HistGradientBoostingRegressor(max_iter=200, random_state=42)}.items():
    m.fit(Xtr, ytr); pred = m.predict(Xte)
    r5[name + '_MAPE'] = round(float((abs(pred - yte) / yte).mean() * 100), 1)
OUT['rq5_models'] = r5

json.dump(OUT, open('out/baselines.json', 'w'), indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))

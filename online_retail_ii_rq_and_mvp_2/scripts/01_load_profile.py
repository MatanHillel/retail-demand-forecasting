"""Load Online Retail II (both sheets), normalise dtypes, save parquet, and print a data profile."""
import pandas as pd, numpy as np, json, time

t = time.time()
sheets = pd.read_excel('data/online_retail_II.xlsx', sheet_name=None, engine='calamine')
df = pd.concat([v.assign(sheet=k) for k, v in sheets.items()], ignore_index=True)
df.columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'CustomerID', 'Country', 'sheet']
for c in ['Invoice', 'StockCode', 'Description', 'Country']:
    df[c] = df[c].astype('string')
df['Description'] = df['Description'].str.strip()
df['StockCode'] = df['StockCode'].str.strip()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.to_parquet('data/online_retail_II_raw.parquet', index=False)
print(f'loaded {df.shape} in {time.time()-t:.1f}s')

P = {}
P['rows'] = len(df)
P['rows_by_sheet'] = df['sheet'].value_counts().to_dict()
P['date_min'] = str(df['InvoiceDate'].min())
P['date_max'] = str(df['InvoiceDate'].max())
P['missing'] = df.isna().sum().to_dict()
P['missing_pct'] = (df.isna().mean() * 100).round(2).to_dict()
P['exact_duplicate_rows'] = int(df.duplicated().sum())
P['n_invoices'] = int(df['Invoice'].nunique())
P['n_stockcodes'] = int(df['StockCode'].nunique())
P['n_descriptions'] = int(df['Description'].nunique())
P['n_customers'] = int(df['CustomerID'].nunique())
P['n_countries'] = int(df['Country'].nunique())

# cancellations
is_cancel = df['Invoice'].str.upper().str.startswith('C')
P['cancel_rows'] = int(is_cancel.sum())
P['cancel_invoices'] = int(df.loc[is_cancel, 'Invoice'].nunique())
P['cancel_invoice_share_pct'] = round(P['cancel_invoices'] / P['n_invoices'] * 100, 2)
P['neg_quantity_rows'] = int((df['Quantity'] < 0).sum())
P['neg_quantity_not_cancel_rows'] = int(((df['Quantity'] < 0) & ~is_cancel).sum())
P['zero_price_rows'] = int((df['Price'] == 0).sum())
P['neg_price_rows'] = int((df['Price'] < 0).sum())
P['price_gt_1000_rows'] = int((df['Price'] > 1000).sum())
P['quantity_gt_1000_rows'] = int((df['Quantity'] > 1000).sum())

# invoice prefixes other than digits/C
pref = df['Invoice'].str.extract(r'^([A-Za-z]+)')[0].fillna('')
P['invoice_prefixes'] = pref.value_counts().to_dict()

# stock code patterns
sc = df['StockCode']
is_std = sc.str.match(r'^\d{5}[A-Za-z]{0,2}$')
P['stockcode_standard_rows'] = int(is_std.sum())
nonstd = df.loc[~is_std, 'StockCode'].value_counts()
P['stockcode_nonstandard_top30'] = nonstd.head(30).to_dict()
P['stockcode_nonstandard_rows'] = int((~is_std).sum())
P['stockcode_nonstandard_unique'] = int(nonstd.shape[0])

# descriptions that look like notes rather than products (lowercase / contains ? / damaged etc.)
desc = df['Description'].fillna('')
lower_desc = desc[(desc != '') & (desc == desc.str.lower())]
P['lowercase_description_rows'] = int(lower_desc.shape[0])
P['lowercase_description_examples'] = lower_desc.value_counts().head(20).to_dict()

# missing customer share and where
P['rows_no_customer'] = int(df['CustomerID'].isna().sum())
P['rows_no_customer_pct'] = round(df['CustomerID'].isna().mean() * 100, 2)
rev = df['Quantity'] * df['Price']
P['revenue_share_no_customer_pct'] = round(rev[df['CustomerID'].isna() & ~is_cancel & (df['Quantity'] > 0)].sum() /
                                          rev[~is_cancel & (df['Quantity'] > 0)].sum() * 100, 2)

# countries
P['country_top15_rows'] = df['Country'].value_counts().head(15).to_dict()
P['uk_row_share_pct'] = round((df['Country'] == 'United Kingdom').mean() * 100, 2)

# time
P['saturday_rows'] = int((df['InvoiceDate'].dt.dayofweek == 5).sum())
P['rows_by_weekday'] = df['InvoiceDate'].dt.day_name().value_counts().to_dict()
P['rows_by_hour'] = df['InvoiceDate'].dt.hour.value_counts().sort_index().to_dict()

# stock code -> multiple descriptions
sc_desc = df.dropna(subset=['Description']).groupby('StockCode')['Description'].nunique()
P['stockcodes_with_multiple_descriptions'] = int((sc_desc > 1).sum())
P['stockcodes_with_multiple_descriptions_pct'] = round((sc_desc > 1).mean() * 100, 2)

# per-invoice stats
inv = df[~is_cancel & (df['Quantity'] > 0) & (df['Price'] > 0)].groupby('Invoice').agg(
    lines=('StockCode', 'size'), qty=('Quantity', 'sum'), value=('Quantity', lambda s: 0))
inv_val = df[~is_cancel & (df['Quantity'] > 0) & (df['Price'] > 0)].assign(v=lambda d: d.Quantity * d.Price).groupby('Invoice')['v'].sum()
P['invoice_value_describe'] = inv_val.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(2).to_dict()
P['invoice_lines_describe'] = inv.lines.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(2).to_dict()

json.dump(P, open('out/profile_raw.json', 'w'), indent=2, default=str)
print(json.dumps(P, indent=2, default=str))

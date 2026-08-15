# Online Retail II — research-question feasibility + MVP handoff

Files
- research_questions.md  : candidate research questions with data-backed feasibility (Hebrew + English)
- mvp_handoff.md         : locked RQ, data facts, Product x Month table spec, feasibility results, implementation tips, contract skeleton
- features_product_month_k6.csv : starter Product x Month table (k=6 universe rule), 70,678 rows
- scripts/               : 01 profile, 02 clean+feasibility, 03 baselines, 04 inventory angle, 05 MVP table EDA, 06 extra models
- out/                   : JSON outputs every number in the docs was taken from

Reproduce
1. put online_retail_II.xlsx (UCI, both sheets) in data/
2. pip install pandas scikit-learn pyarrow python-calamine
3. run scripts in order (01 -> 06)

# Data licence and attribution

**Licence** — this project's code is MIT-licensed (see [`LICENSE`](LICENSE)). The dataset it
analyses, and every artifact derived from it, is licensed and attributed separately, as required by
the dataset's own terms (PRD §48).

## Source

**Online Retail II**, UCI Machine Learning Repository, dataset 502.

* **Citation:** Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  `https://doi.org/10.24432/C5CG6D`. See also Chen, Sain & Guo (2012), *Data mining for the online
  retail industry: A case study of RFM model-based customer segmentation using data mining*,
  Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208.
* **Licence:** CC BY 4.0 (Creative Commons Attribution 4.0 International) — reuse is permitted for
  any purpose, including commercially, provided attribution is given.
* **Kaggle mirror:** `mashlyn/online-retail-ii-uci`, listed there as CC0. This project treats the
  UCI source as authoritative and keeps the CC BY 4.0 attribution regardless of which mirror a
  contributor downloads from (`config/data_sources.yaml`).

## What this means in practice

* **Raw data is never committed** to this repository (`data/raw/` is git-ignored). A script
  downloads it and verifies a SHA-256 hash — see the Quick start section of the
  [README](README.md#quick-start).
* **Derived artifacts keep the attribution.** `data/processed/clean_data.csv`,
  `data/processed/features.csv` and every report under `artifacts/reports/` are committed, are
  derived works of the CC BY 4.0 dataset, and carry this same attribution requirement — this file,
  and the citation line on the README's title, are that attribution.
* **Anonymisation.** `Customer ID` is used only to count distinct customers per product-month
  (`customer_count`, a diagnostic column, never a model feature); no individual-level decision is
  made and no personally identifying field is displayed by the application (PRD §48).

## `CITATION.cff`

[`CITATION.cff`](CITATION.cff) carries the same citation in the machine-readable
[Citation File Format](https://citation-file-format.github.io/), so GitHub's "Cite this repository"
button and citation managers pick it up automatically.

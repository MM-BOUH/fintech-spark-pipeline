# Fintech Fraud Detection — Spark Pipeline

A PySpark MLlib pipeline that ingests 6.36M synthetic mobile-money transactions, engineers fraud-detection features at scale, and trains a gradient-boosted tree classifier that improves recall over the dataset's built-in rule by ~200x. Outputs are served via a Streamlit dashboard.

Built to demonstrate distributed-data engineering and honest ML evaluation on a dataset too large for standard tools (Excel and Numbers cap at 1M rows).

## Live dashboard

[fintech-spark-pipeline.streamlit.app](#) — interactive exploration of the dataset, model performance, and ablation analysis.

## What the pipeline does

| Stage | Script | What happens |
|---|---|---|
| 1. Ingest | `src/01_explore_transactions.py` | Read 6.36M-row CSV, profile by transaction type, write partitioned Parquet |
| 2. Customer features | `src/02_customer_features.py` | Per-origin aggregations; finds that PaySim has near-zero customer recurrence |
| 3. Destination + window features | `src/03_destination_features.py` | 24-hour rolling velocity per destination, balance reconciliation features |
| 4. Model | `src/04_fraud_model.py` | GBT classifier, time-based split, class weighting, evaluation vs baseline rule |
| 5. Ablation | `src/05_ablation.py` | Retrains on reduced feature sets to identify which signals carry the model |
| 6. Dashboard | `dashboard.py` | Streamlit + Plotly UI over the Parquet outputs |

## Key findings

- **100% of fraud is in TRANSFER and CASH_OUT** — 57% of records can be skipped from scoring scope
- **The rule-based baseline catches 0.19% of fraud** (16 of 8,213 cases)
- **The GBT model catches 99.94% on the time-based test split** (1,599 of 1,600)
- **Ablation shows balance features carry the model**. Removing them drops recall to 73.88% and precision to 10.42% — a known PaySim synthetic-data property that wouldn't transfer cleanly to real banking data without equivalent reconciliation features

## Tech stack

PySpark 3.5 · MLlib · Parquet (Snappy compression, partitioned by transaction type) · Plotly · Streamlit · gradient-boosted trees · time-based evaluation · ablation analysis

## Running locally

Prerequisites: Python 3.11+, Java 11 or 17.

```bash
git clone https://github.com/<your-username>/fintech-spark-pipeline.git
cd fintech-spark-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pyspark==3.5.3  # only needed to regenerate the data
```

Download the PaySim dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), rename to `data/raw/paysim.csv`, then run scripts 01 → 05 in order. After that:

```bash
streamlit run dashboard.py
```

## Methodology notes

- **Time-based train/test split.** Train on steps 1–600, test on steps 601–744. Random splits leak future patterns into training data and produce optimistic evaluation scores.
- **Class weighting, not resampling.** 0.3% positive rate is handled by weighting fraud examples ~411x heavier in the loss function. No data is thrown away and no synthetic samples are created.
- **Ablation over single-number metrics.** Reporting 99.94% recall without investigating *why* would be misleading. The ablation table makes the model's dependence on balance features explicit.

## What I'd do differently

- Re-evaluate on a dataset with real per-customer history (Sparkov, IEEE-CIS)
- Add Spark Structured Streaming for live transaction scoring
- Tune the decision threshold against a defined business cost ratio
- Compare GBT against logistic regression and random forest baselines
- Serve the model behind a FastAPI endpoint with Docker

## Author

Mohamed Mehfoud Bouh — PhD, Information Science, Kyushu University. Portfolio at [mmbouh.netlify.app](https://mmbouh.netlify.app).

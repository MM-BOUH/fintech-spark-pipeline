"""
Streamlit dashboard for the fintech Spark fraud detection pipeline.
Reads precomputed Parquet outputs — no Spark needed at dashboard time.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA = Path(__file__).parent / "data" / "processed"
if not DATA.exists():
    DATA = Path(__file__).parent / "sample_data"

st.set_page_config(
    page_title="Fintech Fraud Detection — Spark Pipeline",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Fintech Fraud Detection — Spark Pipeline")
st.caption(
    "PySpark MLlib pipeline on 6.36M financial transactions. "
    "Mohamed Mehfoud Bouh — "
    "[GitHub](https://github.com/MM-BOUH/fintech-spark-pipeline)"
)

# Sample-data notice (only shows in the hosted demo, not locally)
if DATA.name == "sample_data":
    st.info(
        "**About this demo.** The hosted version runs on a stratified 500K-row sample "
        "(all 8,213 fraud transactions plus 491K randomly-sampled legit transactions) "
        "rather than the full 6.36M-row dataset. This is a Streamlit Community Cloud "
        "memory constraint — the free tier provides ~1GB RAM, and loading the full "
        "dataset into pandas uses ~3GB. The full pipeline runs locally against all "
        "6.36M rows; the scripts, methodology, and model artifacts are in the "
        "[GitHub repository](https://github.com/MM-BOUH/fintech-spark-pipeline). "
        "All model evaluation numbers (sections 3 and 4) come from the full 6.36M-row "
        "training run."
    )

# ---------- Data loading ----------
@st.cache_data
def load_transactions():
    p = DATA / "transactions"
    if p.is_dir():
        return pd.read_parquet(p)
    return pd.read_parquet(p.with_suffix(".parquet"))

@st.cache_data
def load_predictions():
    p = DATA / "predictions"
    if p.is_dir():
        return pd.read_parquet(p)
    return pd.read_parquet(p.with_suffix(".parquet"))


with st.spinner("Loading transactions..."):
    tx = load_transactions()
    preds = load_predictions()

# ---------- Sidebar ----------
st.sidebar.header("Filters")
st.sidebar.caption(
    "Filters apply to sections 1 and 2 (dataset and exploratory findings). "
    "Sections 3–5 describe the model and pipeline as built, so they ignore filters."
)

all_types = sorted(tx["type"].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Transaction types", all_types, default=all_types,
    help="Leave all selected to see the full dataset."
)
# Defensive: if user clears all, treat as 'all selected' rather than 'none'
if not selected_types:
    selected_types = all_types

step_min, step_max = int(tx["step"].min()), int(tx["step"].max())
step_range = st.sidebar.slider("Time range (step)", step_min, step_max, (step_min, step_max))

fraud_only = st.sidebar.checkbox("Show only fraud transactions")

tx_view = tx[tx["type"].isin(selected_types)]
tx_view = tx_view[(tx_view["step"] >= step_range[0]) & (tx_view["step"] <= step_range[1])]
if fraud_only:
    tx_view = tx_view[tx_view["isFraud"] == 1]

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filter", f"{len(tx_view):,}")
st.sidebar.metric("Fraud after filter", f"{int(tx_view['isFraud'].sum()):,}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "1. CSV ingestion, partitioned Parquet output\n"
    "2. Feature engineering on TRANSFER + CASH_OUT\n"
    "3. 24-hour rolling window per destination\n"
    "4. Balance reconciliation features\n"
    "5. GBT classifier with ablation\n"
    "6. This dashboard"
)

# Show an explicit warning if filters reduce the data to nothing
if len(tx_view) == 0:
    st.warning("No rows match the current filters. Widen the time range or uncheck 'show only fraud'.")
    st.stop()

# ===============================================================
# 1. DATASET
# ===============================================================
st.header("1. Dataset")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Transactions", f"{len(tx_view):,}")
col2.metric("Fraud cases", f"{int(tx_view['isFraud'].sum()):,}")
fraud_pct = tx_view['isFraud'].mean()*100 if len(tx_view) else 0
col3.metric("Fraud rate", f"{fraud_pct:.3f}%")
col4.metric("Origins", f"{tx_view['nameOrig'].nunique():,}")
col5.metric("Destinations", f"{tx_view['nameDest'].nunique():,}")

st.markdown(
    "PaySim is a synthetic mobile-money dataset of 6.36M transactions across roughly "
    "6.35M unique origin accounts and 2.72M destinations. Almost every origin appears "
    "exactly once, which rules out per-customer history features and pushes the "
    "modelling toward transaction-level and destination-level signals."
)

st.subheader("Activity over time")
ts = tx_view.groupby("step").agg(
    transactions=("amount", "count"),
    fraud=("isFraud", "sum"),
).reset_index()

col1, col2 = st.columns(2)
with col1:
    fig = px.line(ts, x="step", y="transactions",
                  labels={"step": "Step (hour)", "transactions": "Transactions"})
    fig.update_layout(title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Transactions per hour in the filtered range.")

with col2:
    fig = px.line(ts, x="step", y="fraud",
                  labels={"step": "Step (hour)", "fraud": "Fraud cases"},
                  color_discrete_sequence=["#E63946"])
    fig.update_layout(title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Fraud is sparse early in the simulation and ramps up later — about 0.24% "
               "of training-period transactions are fraud vs 3.67% in the test period.")

st.subheader("Where the fraud amounts sit")
fraud_amounts = tx_view[tx_view["isFraud"] == 1]["amount"]
legit_pool = tx_view[tx_view["isFraud"] == 0]["amount"]

if len(fraud_amounts) > 0 and len(legit_pool) > 0:
    legit_sample = legit_pool.sample(min(50000, len(legit_pool)), random_state=42)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.log10(legit_sample + 1), name="Legit",
                               opacity=0.6, marker_color="#2E86AB", nbinsx=60))
    fig.add_trace(go.Histogram(x=np.log10(fraud_amounts + 1), name="Fraud",
                               opacity=0.7, marker_color="#E63946", nbinsx=60))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="log10(amount + 1)",
        yaxis_title="Count",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Fraud amounts cluster higher than legit. The peak around 4–6 on the log "
               "scale corresponds to roughly $10K–$1M per transaction.")
else:
    st.caption("Need both fraud and legit transactions in the filter to render this comparison.")

# ===============================================================
# 2. EXPLORATORY FINDINGS
# ===============================================================
st.header("2. Exploratory findings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Volume by type")
    vol = tx_view.groupby("type").agg(
        tx_count=("amount", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
    ).reset_index().sort_values("total_amount", ascending=False)
    fig = px.bar(vol, x="type", y="total_amount",
                 labels={"total_amount": "Total amount ($)", "type": "Type"},
                 color="total_amount", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(vol, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Fraud rate by type")
    fraud_rate = tx_view.groupby("type").agg(
        total=("isFraud", "count"),
        fraud_count=("isFraud", "sum"),
    ).reset_index()
    fraud_rate["fraud_rate_pct"] = fraud_rate.apply(
        lambda r: r["fraud_count"] / r["total"] * 100 if r["total"] else 0, axis=1
    )
    fig = px.bar(fraud_rate.sort_values("fraud_rate_pct", ascending=False),
                 x="type", y="fraud_rate_pct",
                 labels={"fraud_rate_pct": "Fraud rate (%)", "type": "Type"},
                 color="fraud_rate_pct", color_continuous_scale="Reds")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fraud_rate, hide_index=True, use_container_width=True)

st.markdown(
    "All fraud lives in TRANSFER and CASH_OUT. PAYMENT, DEBIT and CASH_IN contain none. "
    "Fraudsters move money out of compromised accounts and extract it — they don't deposit. "
    "The downstream model only scores these two types, which removes about 57% of records "
    "from scope."
)

st.subheader("Balance reconciliation")
# These come from the full dataset analysis (script 03) — they describe a property
# of the dataset as a whole, not the filtered slice.
recon = pd.DataFrame({
    "Class": ["Fraud", "Legit"],
    "Origin balance mismatch (%)": [0.55, 90.10],
    "Destination balance mismatch (%)": [51.58, 9.61],
})

col1, col2 = st.columns([1, 1])

with col1:
    fig = px.bar(recon.melt(id_vars="Class", var_name="Side", value_name="Mismatch %"),
                 x="Class", y="Mismatch %", color="Side", barmode="group",
                 color_discrete_sequence=["#E63946", "#F4A261"])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(recon, hide_index=True, use_container_width=True)
    st.markdown(
        "A balance mismatch means `oldbalance − newbalance − amount` is not zero — "
        "the math doesn't add up. Counterintuitively, legitimate transactions in "
        "PaySim fail this check 90% of the time at the origin, while fraud reconciles "
        "almost perfectly. The destination side flips: fraud destinations mismatch "
        "52%, legit ones 10%. This is a synthetic-data property, not a real fraud "
        "signal — but the model latches onto it strongly. (Values computed once over "
        "the full TRANSFER+CASH_OUT subset; not affected by filters.)"
    )

st.subheader("Most active destinations")
top_dest_source = tx_view[tx_view["type"].isin(["TRANSFER", "CASH_OUT"])]
if len(top_dest_source):
    top_dest = (top_dest_source
                .groupby("nameDest")
                .agg(times_received=("amount", "count"),
                     total_received=("amount", "sum"),
                     fraud_received=("isFraud", "sum"))
                .reset_index()
                .sort_values("times_received", ascending=False)
                .head(10))
    st.dataframe(top_dest, hide_index=True, use_container_width=True)
    st.caption(
        "Top destinations by frequency within the filtered slice. In the full dataset, "
        "the most popular destinations received 45–56 transactions but only 1–2 fraud "
        "cases each — fraud doesn't concentrate at popular destinations, which is the "
        "opposite of typical mule-account behavior."
    )
else:
    st.caption("No TRANSFER or CASH_OUT transactions in the current filter.")

# ===============================================================
# 3. MODEL  (uses full prediction set — filters don't apply)
# ===============================================================
st.header("3. Model")
st.caption("Section 3 shows the model as trained and evaluated on the full test split — sidebar filters do not apply here.")

st.markdown(
    "A gradient-boosted tree classifier (50 trees, depth 5) trained on steps 1–600 "
    "and evaluated on steps 601–744. The split is time-based rather than random "
    "because random splits leak future patterns into training and produce optimistic "
    "scores. Class weights compensate for the 0.3% positive rate."
)

tp = int(((preds["isFraud"] == 1) & (preds["prediction"] == 1)).sum())
fn = int(((preds["isFraud"] == 1) & (preds["prediction"] == 0)).sum())
fp = int(((preds["isFraud"] == 0) & (preds["prediction"] == 1)).sum())
tn = int(((preds["isFraud"] == 0) & (preds["prediction"] == 0)).sum())

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precision", f"{precision*100:.2f}%")
col2.metric("Recall", f"{recall*100:.2f}%")
col3.metric("F1", f"{f1*100:.2f}%")
col4.metric("AUC-ROC", "1.0000")
col5.metric("vs baseline rule", "200x", "+99.75 pp")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion matrix")
    cm = [[tn, fp], [fn, tp]]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        text=cm, texttemplate="%{text:,}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(height=400, title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Test set: steps 601–744. One missed fraud, one false alarm out of 43,550 transactions.")

with col2:
    st.subheader("Recall vs rule-based baseline")
    comparison = pd.DataFrame({
        "Method": ["isFlaggedFraud (rule)", "GBT model"],
        "Recall (%)": [0.500, recall * 100],
    })
    fig = px.bar(comparison, x="Method", y="Recall (%)",
                 color="Method", color_discrete_sequence=["#888", "#2E86AB"],
                 text="Recall (%)")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False, height=400, yaxis_range=[0, 110])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dataset's built-in rule caught 16 of 8,213 fraud cases (0.19%) overall. "
               "The model catches 1,599 of 1,600 on the test split.")

st.subheader("Model confidence on the test set")
preds["fraud_prob"] = preds["probability"].apply(
    lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x
)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=preds[preds["isFraud"] == 0]["fraud_prob"],
    name="Actual Legit", opacity=0.7, marker_color="#2E86AB", nbinsx=40,
))
fig.add_trace(go.Histogram(
    x=preds[preds["isFraud"] == 1]["fraud_prob"],
    name="Actual Fraud", opacity=0.7, marker_color="#E63946", nbinsx=40,
))
fig.update_layout(
    barmode="overlay",
    xaxis_title="Predicted P(fraud)",
    yaxis_title="Count (log scale)",
    yaxis_type="log",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud and legit transactions pile up at opposite ends of the probability axis. "
           "The y-axis is log-scaled because legit transactions outnumber fraud by ~26x in this split.")

st.subheader("Feature importance")
fi = pd.DataFrame({
    "Feature": [
        "orig_balance_diff", "newbalanceOrig", "dest_balance_diff", "oldbalanceOrg",
        "amount", "oldbalanceDest", "dest_tx_last_24h", "newbalanceDest",
        "dest_amount_last_24h", "orig_balance_mismatch", "type_idx", "dest_balance_mismatch",
    ],
    "Importance": [0.8084, 0.1485, 0.0271, 0.0099, 0.0024, 0.0015,
                   0.0011, 0.0007, 0.0002, 0.0001, 0.0001, 0.0000],
})
fig = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature",
             orientation="h", color="Importance", color_continuous_scale="Viridis")
fig.update_layout(height=500, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("Eighty percent of the model's decisions trace back to one feature: "
           "the origin balance reconciliation error. Amount, type, and velocity together "
           "contribute less than 1%.")

# ===============================================================
# 4. ABLATION  (uses model results — filters don't apply)
# ===============================================================
st.header("4. Ablation")
st.caption("Section 4 reports a fixed experiment over the model's training/test setup — sidebar filters do not apply.")

st.markdown(
    "A 99.94% recall score is suspicious on its own, so I retrained the model on "
    "reduced feature sets to see what was actually doing the work. Three configurations:"
)

st.markdown(
    "- **Full** keeps every feature, including the four engineered balance-difference and "
    "mismatch features computed in script 03.\n"
    "- **Raw balances only** removes those four engineered features but keeps the raw "
    "`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` columns. "
    "This asks whether the model needed the shortcuts or could reconstruct the pattern itself.\n"
    "- **No balance information** strips all balance columns entirely. This asks how much "
    "of the signal lives in amount, type, and velocity alone."
)

ablation = pd.DataFrame({
    "Feature set": [
        "Full (all 12)",
        "Raw balances only",
        "No balance information",
    ],
    "What's excluded": [
        "nothing",
        "orig_balance_diff, dest_balance_diff, mismatch flags",
        "all balance columns + engineered diffs",
    ],
    "AUC": [1.0000, 0.9982, 0.8122],
    "Recall (%)": [99.94, 99.69, 73.88],
    "Precision (%)": [99.94, 57.56, 10.42],
    "F1 (%)": [99.94, 72.98, 18.26],
})

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(ablation, hide_index=True, use_container_width=True)

with col2:
    fig = px.bar(ablation, x="Feature set", y=["Recall (%)", "Precision (%)"],
                 barmode="group",
                 color_discrete_sequence=["#2E86AB", "#E63946"])
    fig.update_layout(yaxis_title="Score (%)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**What this tells us.** Removing the four engineered difference features barely hurts recall
(99.94 → 99.69) but precision collapses from 99.94 to 57.56. The engineered features weren't
the source of the fraud signal — they were what kept false positives down. Stripping balance
information entirely (third row) collapses both metrics: recall to 73.88, precision to 10.42.

The PaySim balance-reconciliation quirk is what's carrying the model. On real banking data
without this artifact, performance is likely closer to the third row unless equivalent
reconciliation features can be engineered from production transaction and balance logs.
That's a worthwhile thing to know up front rather than after deployment.
"""
)

st.subheader("Looking at the two errors")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Missed fraud (false negative)**")
    fn_rows = preds[(preds["isFraud"] == 1) & (preds["prediction"] == 0)]
    st.dataframe(fn_rows[["step", "type", "amount", "fraud_prob"]].head(10),
                 hide_index=True, use_container_width=True)
    st.caption("A $341 CASH_OUT — small enough to look like a routine purchase. "
               "Small-dollar fraud is a known weak spot of models trained mostly on big-ticket fraud.")

with col2:
    st.markdown("**False alarm (false positive)**")
    fp_rows = preds[(preds["isFraud"] == 0) & (preds["prediction"] == 1)]
    st.dataframe(fp_rows[["step", "type", "amount", "fraud_prob"]].head(10),
                 hide_index=True, use_container_width=True)
    st.caption("A $488K CASH_OUT scored 0.525 — barely above the 0.5 cutoff. "
               "Raising the threshold to 0.6 would have caught it without losing real fraud.")

# ===============================================================
# 5. ENGINEERING NOTES
# ===============================================================
st.header("5. Engineering notes")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Storage")
    storage = pd.DataFrame({
        "Format": ["CSV", "Parquet"],
        "Size (MB)": [477, 256],
    })
    fig = px.bar(storage, x="Format", y="Size (MB)",
                 color="Format", color_discrete_sequence=["#888", "#2E86AB"])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Partitioned Parquet output is 46% smaller than the raw CSV. The win is "
               "modest here because account IDs are high-cardinality strings that don't "
               "compress well — drop those columns and Parquet shrinks to under 50MB.")

with col2:
    st.subheader("Spark features used")
    st.markdown(
        "- DataFrame API and Spark SQL\n"
        "- Partitioned writes for partition pruning\n"
        "- Window functions for 24-hour velocity\n"
        "- Conditional aggregation\n"
        "- Time-based train/test split\n"
        "- MLlib Pipeline: Indexer, VectorAssembler, GBT\n"
        "- Class weighting for imbalance\n"
        "- Lazy evaluation, cached training data"
    )

with col3:
    st.subheader("What I'd do differently")
    st.markdown(
        "- Re-run on a dataset with real per-customer history (Sparkov, IEEE-CIS)\n"
        "- Add structured streaming for live scoring\n"
        "- Tune the decision threshold against a defined business cost ratio\n"
        "- Stratify evaluation by transaction-amount band\n"
        "- Serve the model behind a FastAPI endpoint\n"
        "- Compare GBT against logistic regression and random forest as baselines"
    )

st.markdown("---")
st.caption(
    "PySpark · MLlib · Parquet · Plotly · Streamlit · gradient-boosted trees · "
    "time-based evaluation · ablation analysis"
)
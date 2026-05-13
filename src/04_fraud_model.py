"""
Train a fraud detection model with Spark MLlib on enriched transactions.

Pipeline:
  1. Load enriched data from script 03
  2. Time-based train/test split (no future leakage)
  3. Feature assembly + class weighting
  4. Train GBTClassifier
  5. Evaluate against the 0.19% rule-based baseline
  6. Save model and predictions to disk
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from config import DATA_PROCESSED

spark = (SparkSession.builder
         .appName("FintechPipeline-FraudModel")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# =============================================================
# 1. LOAD DATA
# =============================================================
df = spark.read.parquet(str(DATA_PROCESSED / "enriched_transactions"))
print(f"\n{'='*60}\nLOADED ENRICHED TRANSACTIONS\n{'='*60}")
print(f"Total rows: {df.count():,}")
print(f"Fraud rate: {df.filter(F.col('isFraud') == 1).count() / df.count() * 100:.3f}%")

# =============================================================
# 2. TIME-BASED TRAIN/TEST SPLIT
# =============================================================
# PaySim has 744 steps (hours). Train on first 600, test on last 144.
# This mimics production: a model never sees the future at training time.
SPLIT_STEP = 600
train = df.filter(F.col("step") <= SPLIT_STEP)
test = df.filter(F.col("step") > SPLIT_STEP)

train_total = train.count()
train_fraud = train.filter(F.col("isFraud") == 1).count()
test_total = test.count()
test_fraud = test.filter(F.col("isFraud") == 1).count()

print(f"\n{'='*60}\nTRAIN/TEST SPLIT (time-based, step={SPLIT_STEP})\n{'='*60}")
print(f"Train: {train_total:,} rows | {train_fraud:,} fraud ({train_fraud/train_total*100:.3f}%)")
print(f"Test:  {test_total:,} rows | {test_fraud:,} fraud  ({test_fraud/test_total*100:.3f}%)")

# =============================================================
# 3. CLASS WEIGHTING
# =============================================================
# 0.3% fraud rate means a naive model predicts "legit always" and gets
# 99.7% accuracy with 0% recall. We weight fraud samples ~300x heavier
# so the model is forced to learn the minority class.
fraud_weight = (train_total - train_fraud) / train_fraud
train = train.withColumn(
    "class_weight",
    F.when(F.col("isFraud") == 1, fraud_weight).otherwise(1.0)
)
print(f"\nFraud class weight: {fraud_weight:.2f}")

# =============================================================
# 4. FEATURE PIPELINE
# =============================================================
# Encode transaction type as a number (TRANSFER=0, CASH_OUT=1, etc.)
type_indexer = StringIndexer(inputCol="type", outputCol="type_idx", handleInvalid="keep")

# Numeric features we engineered + raw numeric columns
feature_cols = [
    "type_idx",
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_diff", "dest_balance_diff",
    "orig_balance_mismatch", "dest_balance_mismatch",
    "dest_tx_last_24h", "dest_amount_last_24h",
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

# Gradient-boosted trees: ~50 trees, depth 5 — standard tabular config
gbt = GBTClassifier(
    labelCol="isFraud",
    featuresCol="features",
    weightCol="class_weight",
    maxIter=50,
    maxDepth=5,
    seed=42,
)

pipeline = Pipeline(stages=[type_indexer, assembler, gbt])

# =============================================================
# 5. TRAIN
# =============================================================
print(f"\n{'='*60}\nTRAINING GBT CLASSIFIER\n{'='*60}")
print("This will take 2-5 minutes on a laptop...")
model = pipeline.fit(train)
print("Training complete.")

# =============================================================
# 6. EVALUATE ON TEST SET
# =============================================================
predictions = model.transform(test)

# AUC — threshold-independent quality measure
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="isFraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
)
auc = auc_evaluator.evaluate(predictions)

# Confusion matrix at the default 0.5 threshold
tp = predictions.filter((F.col("isFraud") == 1) & (F.col("prediction") == 1)).count()
fn = predictions.filter((F.col("isFraud") == 1) & (F.col("prediction") == 0)).count()
fp = predictions.filter((F.col("isFraud") == 0) & (F.col("prediction") == 1)).count()
tn = predictions.filter((F.col("isFraud") == 0) & (F.col("prediction") == 0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n{'='*60}\nMODEL PERFORMANCE ON TEST SET\n{'='*60}")
print(f"AUC-ROC:    {auc:.4f}")
print(f"\nConfusion matrix (threshold = 0.5):")
print(f"                 Predicted Legit | Predicted Fraud")
print(f"  Actual Legit:  {tn:>14,} | {fp:>14,}")
print(f"  Actual Fraud:  {fn:>14,} | {tp:>14,}")
print(f"\nPrecision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1:        {f1*100:.2f}%")

# =============================================================
# 7. COMPARE TO RULE-BASED BASELINE
# =============================================================
baseline_tp = test.filter((F.col("isFraud") == 1) & (F.col("isFlaggedFraud") == 1)).count()
baseline_recall = baseline_tp / test_fraud if test_fraud > 0 else 0.0

print(f"\n{'='*60}\nBASELINE COMPARISON\n{'='*60}")
print(f"Rule-based 'isFlaggedFraud' recall: {baseline_recall*100:.3f}%")
print(f"GBT model recall:                   {recall*100:.3f}%")
if baseline_recall > 0:
    print(f"Improvement factor:                 {recall/baseline_recall:.1f}x")
else:
    print(f"Improvement: model caught {tp} cases vs baseline caught {baseline_tp}")

# =============================================================
# 8. FEATURE IMPORTANCE
# =============================================================
gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()
print(f"\n{'='*60}\nFEATURE IMPORTANCE\n{'='*60}")
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {name:<30} {imp:.4f}")

# =============================================================
# 9. SAVE MODEL AND PREDICTIONS
# =============================================================
model_path = DATA_PROCESSED / "fraud_model"
pred_path = DATA_PROCESSED / "predictions"
model.write().overwrite().save(str(model_path))
predictions.select("step", "type", "amount", "isFraud", "prediction", "probability") \
    .write.mode("overwrite").parquet(str(pred_path))

print(f"\nModel saved to:       {model_path}")
print(f"Predictions saved to: {pred_path}")

spark.stop()
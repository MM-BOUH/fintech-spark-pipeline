"""
Ablation study: how much of the model's performance comes from
the suspected-artifact features (balance-reconciliation) vs
the rest? We retrain on a reduced feature set and compare.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from config import DATA_PROCESSED

spark = (SparkSession.builder
         .appName("Ablation")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(str(DATA_PROCESSED / "enriched_transactions"))
train = df.filter(F.col("step") <= 600)
test = df.filter(F.col("step") > 600)

train_total = train.count()
train_fraud = train.filter(F.col("isFraud") == 1).count()
fraud_weight = (train_total - train_fraud) / train_fraud
train = train.withColumn(
    "class_weight",
    F.when(F.col("isFraud") == 1, fraud_weight).otherwise(1.0)
)

# Define feature sets to compare
FEATURE_SETS = {
    "full":          ["type_idx", "amount", "oldbalanceOrg", "newbalanceOrig",
                      "oldbalanceDest", "newbalanceDest",
                      "orig_balance_diff", "dest_balance_diff",
                      "orig_balance_mismatch", "dest_balance_mismatch",
                      "dest_tx_last_24h", "dest_amount_last_24h"],
    "no_balance_diffs": ["type_idx", "amount", "oldbalanceOrg", "newbalanceOrig",
                         "oldbalanceDest", "newbalanceDest",
                         "dest_tx_last_24h", "dest_amount_last_24h"],
    "no_balance_at_all": ["type_idx", "amount",
                          "dest_tx_last_24h", "dest_amount_last_24h"],
}

evaluator = BinaryClassificationEvaluator(
    labelCol="isFraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

print(f"\n{'='*70}\nABLATION RESULTS\n{'='*70}")
print(f"{'Feature set':<22} {'AUC':>8} {'Recall':>10} {'Precision':>12} {'F1':>8}")
print("-" * 70)

for name, cols in FEATURE_SETS.items():
    indexer = StringIndexer(inputCol="type", outputCol="type_idx", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    gbt = GBTClassifier(labelCol="isFraud", featuresCol="features",
                        weightCol="class_weight", maxIter=50, maxDepth=5, seed=42)
    pipeline = Pipeline(stages=[indexer, assembler, gbt])
    
    model = pipeline.fit(train)
    preds = model.transform(test)
    
    auc = evaluator.evaluate(preds)
    tp = preds.filter((F.col("isFraud") == 1) & (F.col("prediction") == 1)).count()
    fn = preds.filter((F.col("isFraud") == 1) & (F.col("prediction") == 0)).count()
    fp = preds.filter((F.col("isFraud") == 0) & (F.col("prediction") == 1)).count()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    
    print(f"{name:<22} {auc:>8.4f} {recall*100:>9.2f}% {precision*100:>11.2f}% {f1*100:>7.2f}%")

spark.stop()
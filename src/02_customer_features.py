"""
Build customer-level behavioral features from transactions.
Reads from Parquet (faster than CSV) and aggregates per account.
Introduces:
  - reading partitioned Parquet
  - filtering with partition pruning
  - groupBy aggregations
  - conditional aggregations (sum-when pattern)
  - writing a derived dataset
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from config import DATA_PROCESSED

spark = (SparkSession.builder
         .appName("FintechPipeline-CustomerFeatures")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# --- Load from Parquet (much faster than CSV) ---
tx = spark.read.parquet(str(DATA_PROCESSED / "transactions"))

# Focus on the types where fraud actually happens.
# Spark's partition pruning means it reads ONLY these folders.
risky_tx = tx.filter(F.col("type").isin("TRANSFER", "CASH_OUT"))

print(f"\n{'='*60}\nBUILDING CUSTOMER FEATURES\n{'='*60}")
print(f"Risky transactions (TRANSFER + CASH_OUT): {risky_tx.count():,}")

# --- Per-origin-customer aggregations ---
# This is the "sum-when" pattern: count fraud only where isFraud=1
customer_features = (risky_tx.groupBy("nameOrig")
    .agg(
        F.count("*").alias("tx_count"),
        F.round(F.sum("amount"), 2).alias("total_sent"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.max("amount"), 2).alias("max_amount"),
        F.round(F.stddev("amount"), 2).alias("amount_stddev"),
        F.sum(F.when(F.col("type") == "TRANSFER", 1).otherwise(0)).alias("transfer_count"),
        F.sum(F.when(F.col("type") == "CASH_OUT", 1).otherwise(0)).alias("cashout_count"),
        F.sum("isFraud").alias("fraud_count"),
        F.round(F.avg("oldbalanceOrg"), 2).alias("avg_balance_before"),
        # behavioral signal: did they drain their account?
        F.sum(F.when(F.col("newbalanceOrig") == 0, 1).otherwise(0)).alias("zero_balance_count"),
    )
)

# Add a fraud label at the customer level
customer_features = customer_features.withColumn(
    "is_fraud_customer", (F.col("fraud_count") > 0).cast("integer")
)

print(f"\nTotal customers in risky transactions: {customer_features.count():,}")

# --- Top customers by volume ---
print(f"\n{'='*60}\nTOP 10 CUSTOMERS BY VOLUME SENT\n{'='*60}")
customer_features.orderBy(F.desc("total_sent")).show(10, truncate=False)

# --- Compare fraud vs legit customer profiles ---
print(f"\n{'='*60}\nAVG BEHAVIOR: FRAUD CUSTOMERS vs LEGIT\n{'='*60}")
(customer_features.groupBy("is_fraud_customer")
    .agg(
        F.count("*").alias("customer_count"),
        F.round(F.avg("tx_count"), 2).alias("avg_tx_count"),
        F.round(F.avg("total_sent"), 2).alias("avg_total_sent"),
        F.round(F.avg("avg_amount"), 2).alias("avg_tx_amount"),
        F.round(F.avg("zero_balance_count"), 2).alias("avg_account_drains"),
    ).show(truncate=False))

# --- Write the feature table ---
output_path = DATA_PROCESSED / "customer_features"
(customer_features.write
   .mode("overwrite")
   .parquet(str(output_path)))
print(f"\nWritten to: {output_path}")

spark.stop()
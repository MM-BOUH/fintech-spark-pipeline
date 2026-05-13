"""
Investigate destination accounts and build time-window features.
Per-customer history is unavailable (each origin appears once),
so we shift to destination-side aggregation and global time windows.

Introduces:
  - more groupBy patterns
  - window functions (rolling counts in time)
  - balance reconciliation features
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from config import DATA_PROCESSED

spark = (SparkSession.builder
         .appName("FintechPipeline-DestFeatures")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

tx = spark.read.parquet(str(DATA_PROCESSED / "transactions"))
risky = tx.filter(F.col("type").isin("TRANSFER", "CASH_OUT"))

print(f"\n{'='*60}\nINVESTIGATING DESTINATION ACCOUNTS\n{'='*60}")

# --- How often is each destination reused? ---
dest_freq = (risky.groupBy("nameDest")
    .agg(
        F.count("*").alias("times_received"),
        F.round(F.sum("amount"), 2).alias("total_received"),
        F.sum("isFraud").alias("fraud_received"),
    ))

print(f"Unique destinations in risky tx: {dest_freq.count():,}")
print(f"\nDistribution of how often a destination is reused:")
(dest_freq.groupBy("times_received")
    .count()
    .orderBy(F.desc("count"))
    .show(10, truncate=False))

print(f"\nDestinations that received fraud, sorted by frequency:")
(dest_freq.filter(F.col("fraud_received") > 0)
    .orderBy(F.desc("times_received"))
    .show(10, truncate=False))

# --- Time-window features: "transactions in the last 24 steps" ---
# The 'step' column is hours of simulation. Per destination, compute a rolling
# count and sum of incoming transactions over a 24-hour lookback window.
print(f"\n{'='*60}\nTIME-WINDOW FEATURES (per destination, 24h rolling)\n{'='*60}")

window_spec = (Window
    .partitionBy("nameDest")
    .orderBy("step")
    .rangeBetween(-24, 0))

risky_with_velocity = (risky
    .withColumn("dest_tx_last_24h", F.count("*").over(window_spec))
    .withColumn("dest_amount_last_24h", F.sum("amount").over(window_spec)))

print(f"\nVelocity feature: avg incoming-tx count to destination in prior 24h:")
(risky_with_velocity.groupBy("isFraud")
    .agg(
        F.count("*").alias("tx_count"),
        F.round(F.avg("dest_tx_last_24h"), 2).alias("avg_dest_velocity_24h"),
        F.round(F.avg("dest_amount_last_24h"), 2).alias("avg_dest_amount_24h"),
    ).show(truncate=False))

# --- Balance reconciliation: classic PaySim signal ---
# For fraud, origin balance often doesn't decrement and destination balance
# often doesn't increment to match the transaction amount.
print(f"\n{'='*60}\nBALANCE RECONCILIATION CHECK\n{'='*60}")

reconciled = risky_with_velocity.withColumn(
    "orig_balance_diff",
    F.round(F.col("oldbalanceOrg") - F.col("newbalanceOrig") - F.col("amount"), 2)
).withColumn(
    "dest_balance_diff",
    F.round(F.col("newbalanceDest") - F.col("oldbalanceDest") - F.col("amount"), 2)
).withColumn(
    "orig_balance_mismatch", (F.abs(F.col("orig_balance_diff")) > 0.01).cast("integer")
).withColumn(
    "dest_balance_mismatch", (F.abs(F.col("dest_balance_diff")) > 0.01).cast("integer")
)

print(f"\nBalance mismatch rate, fraud vs legit:")
(reconciled.groupBy("isFraud")
    .agg(
        F.count("*").alias("tx_count"),
        F.round(F.avg("orig_balance_mismatch") * 100, 2).alias("orig_mismatch_pct"),
        F.round(F.avg("dest_balance_mismatch") * 100, 2).alias("dest_mismatch_pct"),
    ).show(truncate=False))

# --- Write enriched transactions for the ML step ---
output_path = DATA_PROCESSED / "enriched_transactions"
(reconciled.write
   .mode("overwrite")
   .parquet(str(output_path)))
print(f"\nWritten to: {output_path}")

spark.stop()
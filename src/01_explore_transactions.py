"""
Explore PaySim transactions: schema, volume, fraud rates by type.
Persist to partitioned Parquet for fast downstream queries.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from config import DATA_RAW, DATA_PROCESSED

spark = (SparkSession.builder
         .appName("FintechPipeline-Explore")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# --- Load ---
df = spark.read.csv(str(DATA_RAW / "paysim.csv"), header=True, inferSchema=True)

print(f"\n{'='*60}\nDATASET OVERVIEW\n{'='*60}")
print(f"Total transactions: {df.count():,}")
print(f"Unique origin accounts:  {df.select('nameOrig').distinct().count():,}")
print(f"Unique destination accounts: {df.select('nameDest').distinct().count():,}")
df.printSchema()

# --- Transaction volume by type ---
print(f"\n{'='*60}\nVOLUME BY TRANSACTION TYPE\n{'='*60}")
(df.groupBy("type")
   .agg(F.count("*").alias("tx_count"),
        F.round(F.sum("amount"), 2).alias("total_amount"),
        F.round(F.avg("amount"), 2).alias("avg_amount"))
   .orderBy(F.desc("total_amount"))
   .show(truncate=False))

# --- Fraud rate by type ---
print(f"\n{'='*60}\nFRAUD RATE BY TRANSACTION TYPE\n{'='*60}")
(df.groupBy("type")
   .agg(F.count("*").alias("total"),
        F.sum("isFraud").alias("fraud_count"),
        F.round(F.sum("isFraud") / F.count("*") * 100, 4).alias("fraud_rate_pct"))
   .orderBy(F.desc("fraud_rate_pct"))
   .show(truncate=False))

# --- How well does the existing isFlaggedFraud catch fraud? ---
print(f"\n{'='*60}\nEXISTING FRAUD-FLAG SYSTEM PERFORMANCE\n{'='*60}")
total_fraud = df.filter(F.col("isFraud") == 1).count()
flagged_fraud = df.filter((F.col("isFraud") == 1) & (F.col("isFlaggedFraud") == 1)).count()
print(f"Actual fraud cases:   {total_fraud:,}")
print(f"Caught by flag:       {flagged_fraud:,}")
print(f"Recall of flag rule:  {flagged_fraud / total_fraud * 100:.2f}%")

# --- Persist as partitioned Parquet ---
print(f"\n{'='*60}\nWRITING TO PARQUET\n{'='*60}")
output_path = DATA_PROCESSED / "transactions"
(df.write
   .mode("overwrite")
   .partitionBy("type")
   .parquet(str(output_path)))
print(f"Written to: {output_path}")

spark.stop()
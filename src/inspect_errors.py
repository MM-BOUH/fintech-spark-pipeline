from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from config import DATA_PROCESSED

spark = SparkSession.builder.appName("Inspect").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

p = spark.read.parquet(str(DATA_PROCESSED / "predictions"))

print("\nMISSED FRAUD (false negative):")
p.filter((F.col("isFraud") == 1) & (F.col("prediction") == 0)).show(truncate=False)

print("\nFALSE ALARM (false positive):")
p.filter((F.col("isFraud") == 0) & (F.col("prediction") == 1)).show(truncate=False)

spark.stop()

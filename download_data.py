from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

input_data = spark.read.table('dave_carlson_databricks_com_db.gartner_2020_featurized').toPandas()
input_data.to_csv('data/train.csv', index=False)


code_lookup_df = spark.read.table("gartner.descriptions").toPandas()
code_lookup_df.to_csv('data/descriptions.csv', index=False)

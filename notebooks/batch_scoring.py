# Databricks notebook source
import mlflow

model_uri='models:/income_adults_model_prod/Production'

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri)

columns = mlflow.pyfunc.load_model(model_uri).metadata.get_input_schema().input_names()

display(spark.table('msh.adults_test').withColumn('predictions', loaded_model(*columns)))

# COMMAND ----------

spark.udf.register('model', loaded_model)

# COMMAND ----------

# MAGIC %sql
# MAGIC select model(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country ) from msh.adults_test

# COMMAND ----------



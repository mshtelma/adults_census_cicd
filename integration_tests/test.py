# Databricks notebook source
# MAGIC %run ../notebooks/train_model

# COMMAND ----------

# MAGIC %run ../notebooks/evaluate_models

# COMMAND ----------

# MAGIC %md
# MAGIC # Tests

# COMMAND ----------


cnt = spark.read.format("mlflow-experiment").load(experimentID).where("tags.candidate='true'").count()
assert cnt == 0

# COMMAND ----------



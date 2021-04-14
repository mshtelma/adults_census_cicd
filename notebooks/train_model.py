# Databricks notebook source
# MAGIC %md  
# MAGIC # MLflow in Action & Model Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC # Train model

# COMMAND ----------

# DBTITLE 1,Let's install packages we are going to use
# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

# DBTITLE 1,Import needed packages
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import databricks.automl

# COMMAND ----------

import mlflow
import mlflow.lightgbm
from  mlflow.tracking import MlflowClient

username = 'michael.shtelma@databricks.com'
experimentPath = "/Users/" + username + "/adults_census_exp1"
mlflow.set_experiment(experimentPath)
experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
print(experimentID)

model_name = "income_adults_model"

# COMMAND ----------

# DBTITLE 1,Read dataset into Spark DataFrame
df = spark.read.table('msh.adults') 
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's train our own model!

# COMMAND ----------

# DBTITLE 1,Let's turn on autologging!
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.shap
mlflow.lightgbm.autolog(log_input_examples=True)

# COMMAND ----------

# DBTITLE 1,Handling categorical columns
from sklearn.model_selection import train_test_split, cross_val_score

target = "income"

pdDf = df.toPandas()

for col in pdDf.columns:
  if pdDf.dtypes[col]=='object':
    pdDf[col] =  pdDf[col].astype('category').cat.codes
  if pdDf.dtypes[col]=='int8':
    pdDf[col] =  pdDf[col].astype('float')
  pdDf[col] = pdDf[col].fillna(0)
    
X_train, X_test, Y_train, Y_test = train_test_split(pdDf.drop([target], axis=1), pdDf[target], test_size=0.2)

# COMMAND ----------

luate_

# COMMAND ----------

# DBTITLE 1,Training LightGBM Model
from mlflow.models.signature import infer_signature
import lightgbm as lgb
import sklearn

number_of_explained_examples = 6

with mlflow.start_run(run_name="lightgbm") as run:

  params = {"num_leaves": 32, "objective": "regression", "metric": "rmse", "num_rounds":100}
  num_rounds = 100
  
  train_lgb_dataset = lgb.Dataset(X_train, label=Y_train.values)
  test_lgb_dataset = lgb.Dataset(X_test, label=Y_test.values)

  model = lgb.train(
    params, train_lgb_dataset, num_rounds
  )
  
  val_pred_proba = model.predict(X_test)
  val_pred = np.array([1 if x>=0.5 else 0 for  x in val_pred_proba])

  val_metrics = {
      "val_precision_score": sklearn.metrics.precision_score(Y_test.values, val_pred, average="weighted"),
      "val_recall_score": sklearn.metrics.recall_score(Y_test, val_pred, average="weighted"),
      "val_f1_score": sklearn.metrics.f1_score(Y_test, val_pred, average="weighted"),
      "val_accuracy_score": sklearn.metrics.accuracy_score(Y_test, val_pred, normalize=True),
      "val_log_loss": sklearn.metrics.log_loss(Y_test, val_pred_proba),
  }

  val_metrics["val_roc_auc_score"] = sklearn.metrics.roc_auc_score(
      Y_test,
      val_pred_proba,
      average="weighted",
      )
  mlflow.log_metrics(val_metrics)
  display(pd.DataFrame(val_metrics, index=[0]))
  
  # Let's log explanations
  #mlflow.shap.log_explanation(lambda X: model.predict(X), X_test[0:number_of_explained_examples])
  
  run_id = run.info.run_uuid

# COMMAND ----------

# DBTITLE 1,Let's deploy LightGBM model to production
#model_name = "income_adults_model"
#model_version = mlflow.register_model("runs:/{}/model".format(run_id), model_name)
#model_version

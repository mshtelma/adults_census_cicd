# Databricks notebook source
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

df = spark.read.table('msh.adults') 
display(df)

# COMMAND ----------

from sklearn.model_selection import train_test_split, cross_val_score

target = "income"

pdDf = df.toPandas()

for col in pdDf.columns:
  if pdDf.dtypes[col]=='object':
    pdDf[col] =  pdDf[col].astype('category').cat.codes
  if pdDf.dtypes[col]=='int8':
    pdDf[col] =  pdDf[col].astype('float')
  pdDf[col] = pdDf[col].fillna(0)
    
x_train, x_test, y_train, y_test = train_test_split(pdDf.drop([target], axis=1), pdDf[target], test_size=0.2)

# COMMAND ----------

from sklearn.metrics import roc_auc_score
def evaluate_model(run_id, X, Y):
    model = mlflow.lightgbm.load_model('runs:/{}/model'.format(run_id))
    probs = model.predict(X, verbose=0)
    roc_auc_score_ = roc_auc_score(y_test, probs, average="weighted")
    return roc_auc_score_

# COMMAND ----------

from mlflow.exceptions import RestException

def evaluate_all_candidate_models():
    mlflow_client = MlflowClient()

    cand_run_ids = get_candidate_models()
    best_cand_metric, best_cand_run_id = get_best_model(cand_run_ids, x_test, y_test)
    print('Best ROC AUC (candidate models): ', best_cand_metric)

    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=['Production'])
        prod_run_ids = [v.run_id for v in versions]
        best_prod_metric, best_prod_run_id = get_best_model(prod_run_ids, x_test, y_test)
    except RestException:
        best_prod_metric = -1
    print('ROC AUC (production models): ', best_prod_metric)

    if best_cand_metric >= best_prod_metric:
        # deploy new model
        model_version = mlflow.register_model("runs:/" + best_cand_run_id + "/model", model_name)
        time.sleep(15)
        mlflow_client.transition_model_version_stage(name=model_name, version=model_version.version,
                                                     stage="Production")
        print('Deployed version: ', model_version.version)
    # remove candidate tags
    for run_id in cand_run_ids:
        mlflow_client.set_tag(run_id, 'candidate', 'false')

def get_best_model(run_ids, X, Y):
    best_metric = -1
    best_run_id = None
    for run_id in run_ids:
        metric = evaluate_model(run_id, X, Y)
        print('Evaluated model with metric: ', metric)
        if metric > best_metric:
            best_metric = metric
            best_run_id = run_id
    return best_metric, best_run_id

def get_candidate_models():
    spark_df = spark.read.format("mlflow-experiment").load(experimentID)
    pdf = spark_df.where("tags.candidate='true'").select("run_id").toPandas()
    return pdf['run_id'].values

evaluate_all_candidate_models()

# COMMAND ----------



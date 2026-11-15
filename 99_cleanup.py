# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Clean up
# MAGIC Removes everything **you** created in the workshop: endpoint, monitor, online table, models and your schema.
# MAGIC
# MAGIC Tick `remove_online_store` only when **everyone** in the workspace has finished, because the online store is shared.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("remove_online_store", "no", ["no", "yes"], "remove_online_store")

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

w, fe = WorkspaceClient(), FeatureEngineeringClient()
client = MlflowClient(registry_uri="databricks-uc")


def attempt(label, fn):
    try:
        fn()
        print("✅ removed", label)
    except Exception as e:
        print("·  skipped", label, "-", str(e)[:100])


attempt(f"endpoint {ENDPOINT}", lambda: w.serving_endpoints.delete(ENDPOINT))
for j in w.jobs.list(name=f"kyx-deployment-job-{USER}"):
    attempt(f"deployment job {j.job_id}", lambda j=j: w.jobs.delete(j.job_id))
attempt("monitor", lambda: w.quality_monitors.delete(table_name=fq(PRED_LOG)))
attempt("online table", lambda: fe.drop_online_table(fq(ONLINE_TABLE)) if hasattr(fe, "drop_online_table")
        else w.tables.delete(fq(ONLINE_TABLE)))
for m in (MODEL, MODEL + "_direct"):
    attempt(f"model {m}", lambda m=m: client.delete_registered_model(fq(m)))
attempt(f"schema {CATALOG}.{SCHEMA}", lambda: spark.sql(f"DROP SCHEMA IF EXISTS `{CATALOG}`.`{SCHEMA}` CASCADE"))
attempt("experiment", lambda: w.experiments.delete_experiment(
    __import__("mlflow").get_experiment_by_name(EXPERIMENT).experiment_id))
if dbutils.widgets.get("remove_online_store") == "yes":
    attempt(f"online store {ONLINE_STORE}", lambda: fe.delete_online_store(name=ONLINE_STORE))

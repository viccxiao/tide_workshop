# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · Task 2: Approval check
# MAGIC Stops the job until a person approves this model version.
# MAGIC
# MAGIC **How approval works:** this task fails on purpose until the model version has the tag `Approval_Check = Approved`.
# MAGIC The approver opens the model version in Catalog Explorer, reviews the `test_...` tags and clicks **Approve**.
# MAGIC Databricks then sets the tag and resumes the job from here.

# COMMAND ----------

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

# COMMAND ----------

from mlflow.tracking import MlflowClient

MODEL_NAME = dbutils.widgets.get("model_name")
VERSION = dbutils.widgets.get("model_version")
TAG = "Approval_Check"   # must match this task's name in the job

tags = MlflowClient(registry_uri="databricks-uc").get_model_version(MODEL_NAME, VERSION).tags
if tags.get(TAG, "").lower() != "approved":
    raise Exception(f"⏸ v{VERSION} is waiting for approval. Open the model version and click Approve.")
print(f"✅ v{VERSION} approved")

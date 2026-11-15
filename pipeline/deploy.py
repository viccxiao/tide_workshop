# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · Task 3: Deploy
# MAGIC Makes the approved version the **champion**, keeps the old one as **previous** (for a one-step rollback), and points the endpoint at the new version.

# COMMAND ----------

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

# COMMAND ----------

# MAGIC %run ../_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from mlflow.tracking import MlflowClient

MODEL_NAME = dbutils.widgets.get("model_name") or fq(MODEL)
VERSION = dbutils.widgets.get("model_version")
client = MlflowClient(registry_uri="databricks-uc")
w = WorkspaceClient()

old = client.get_model_version_by_alias(MODEL_NAME, "champion")
if str(old.version) != str(VERSION):
    client.set_registered_model_alias(MODEL_NAME, "previous", old.version)
client.set_registered_model_alias(MODEL_NAME, "champion", VERSION)
print(f"champion: v{old.version} -> v{VERSION} (v{old.version} kept as 'previous')")

new = client.get_model_version(MODEL_NAME, VERSION)
save_config(champion_version=str(VERSION), review_threshold=float(new.tags.get("review_threshold", 0.5)))

# COMMAND ----------

# DBTITLE 1,Point the endpoint at the new champion
endpoint = load_config().get("endpoint", ENDPOINT)
if any(e.name == endpoint for e in w.serving_endpoints.list()):
    w.serving_endpoints.update_config(name=endpoint, served_entities=[ServedEntityInput(
        entity_name=MODEL_NAME, entity_version=str(VERSION), workload_size="Small", scale_to_zero_enabled=True)])
    print(f"✅ Endpoint {endpoint} is switching to v{VERSION}. It keeps serving the old version until the new one is ready.")
else:
    print(f"No endpoint called {endpoint} yet. Run 03_deploy first. The champion alias has still been updated.")

# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Deploy the model
# MAGIC
# MAGIC **Goal:** serve the champion model from **one managed endpoint** that any service can call over HTTP.
# MAGIC
# MAGIC | Today (DIY) | With Model Serving |
# MAGIC |---|---|
# MAGIC | Docker sidecar next to every service pod | One endpoint, called over HTTP |
# MAGIC | Same model copied into ~12 containers across environments | One registered model, served by alias |
# MAGIC | Sidecar and service can't scale separately | Endpoint scales on its own, down to zero when idle |
# MAGIC | Backend prepares the model inputs | Endpoint looks features up from the online store |
# MAGIC | Requests not recorded | Every request and response logged to a table |
# MAGIC
# MAGIC **This notebook has two parts:**
# MAGIC - **Part A:** start the endpoint (about 1 min to run, then 5-15 min to become ready). **Then go to 04_monitoring.**
# MAGIC - **Part B:** come back after 04 and call the endpoint.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering mlflow==3.16.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A · Start the endpoint

# COMMAND ----------

# DBTITLE 1,Create (or update) the endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (AiGatewayConfig, AiGatewayInferenceTableConfig,
                                            EndpointCoreConfigInput, ServedEntityInput)
from mlflow.tracking import MlflowClient

w = WorkspaceClient()
client = MlflowClient(registry_uri="databricks-uc")

ONLINE_OK = load_config().get("online_ok", False)
SERVED_MODEL = fq(MODEL) if ONLINE_OK else fq(MODEL + "_direct")
champ = client.get_model_version_by_alias(SERVED_MODEL, "champion")
print(f"Serving {SERVED_MODEL} version {champ.version} (champion)")

served = ServedEntityInput(entity_name=SERVED_MODEL, entity_version=champ.version,
                           workload_size="Small", scale_to_zero_enabled=True)
logging = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(
    catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix="kyx_endpoint", enabled=True))

if any(e.name == ENDPOINT for e in w.serving_endpoints.list()):
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served])
    print("Updating endpoint", ENDPOINT)
else:
    try:
        w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]),
                                   ai_gateway=logging)
    except Exception as e:
        print("Request logging not available, creating endpoint without it:", str(e)[:120])
        w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]))
    print("Creating endpoint", ENDPOINT)

save_config(endpoint=ENDPOINT, served_model=SERVED_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 👉 While it starts
# MAGIC 1. Open the endpoint (link below) and watch the state go from **Not ready** to **Ready**.
# MAGIC 2. Note what's on the page: served model and version, compute size, scale to zero, and the **Logs** and **Metrics** tabs.
# MAGIC 3. **Now go to `04_monitoring`** and come back here when it's done.

# COMMAND ----------

link(f"Open endpoint {ENDPOINT}", f"/ml/endpoints/{ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B · Call the endpoint (after 04)

# COMMAND ----------

# DBTITLE 1,Wait until the endpoint is ready
import time
for i in range(90):
    state = w.serving_endpoints.get(ENDPOINT).state
    ready, update = str(state.ready), str(state.config_update)
    if ready.endswith("READY") and not ready.endswith("NOT_READY") and "IN_PROGRESS" not in update:
        print("✅ Endpoint is ready")
        break
    if i % 4 == 0:
        print(f"still starting... ({ready}, {update})")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call it like your onboarding service would
# MAGIC With the online store, the request only contains **customer IDs**. The endpoint looks up the rest.
# MAGIC
# MAGIC *If the endpoint has been idle it scales to zero to save cost, and the first call wakes it up. The cell retries until it answers.*

# COMMAND ----------

# DBTITLE 1,Score three new customers
import pandas as pd

threshold = float(load_config()["review_threshold"])
sample_ids = [r.customer_id for r in spark.table(CUSTOMERS).where("cohort = 'live'")
              .orderBy("signup_ts", ascending=False).limit(3).collect()]

if ONLINE_OK:
    records = [{"customer_id": cid} for cid in sample_ids]
else:  # fallback: send the features in the request
    records = (spark.table(FEATURE_TABLE).where(f"customer_id IN ({','.join(repr(c) for c in sample_ids)})")
               .drop("customer_id").toPandas().to_dict("records"))

def query_with_wakeup(records, tries=12):
    """Scale to zero: after idle time the first request wakes the endpoint up, which can take a few minutes."""
    for i in range(tries):
        try:
            return w.serving_endpoints.query(name=ENDPOINT, dataframe_records=records)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"Endpoint is waking up from scale to zero, retrying in 30s... ({str(e)[:80]})")
            time.sleep(30)


response = query_with_wakeup(records)
result = pd.DataFrame({"customer_id": sample_ids, "risk_score": [round(float(p), 3) for p in response.predictions]})
result["decision"] = ["send to review" if s >= threshold else "auto-approve" for s in result.risk_score]
print("Request body:", {"dataframe_records": records})
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The same call over plain HTTP
# MAGIC This is all a backend service needs. No sidecar, no model files, no feature code.

# COMMAND ----------

print(f"""curl -X POST {host()}/serving-endpoints/{ENDPOINT}/invocations \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"dataframe_records": [{{"customer_id": "{sample_ids[0]}"}}]}}'""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 👉 Try it in the UI
# MAGIC 1. Open the endpoint (link above) → **Query endpoint** (top right).
# MAGIC 2. Paste the request printed below and click **Send request**.
# MAGIC 3. Open the **Metrics** tab to see request counts and latency.
# MAGIC 4. A few minutes after your calls, the request log table `kyx_endpoint_payload` appears in your schema. Every request and response is recorded there.

# COMMAND ----------

import json
print(json.dumps({"dataframe_records": records[:1]}, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote a new model without redeploying containers
# MAGIC To promote the challenger, you move the **champion** alias in the registry and point the endpoint at it.
# MAGIC Dev, staging and prod use the same registered model, not separate copies.
# MAGIC
# MAGIC ✅ **Done.** You've gone from raw data to a live, monitored model endpoint.

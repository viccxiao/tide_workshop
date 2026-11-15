# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Monitor the model
# MAGIC
# MAGIC **Goal:** catch drift *before* it hurts. Set up a monitor that compares each day of live predictions with the training data.
# MAGIC
# MAGIC **The scenario:** the model has been live for 30 days. Around day 15, a new kind of sign-up starts appearing that the model rarely saw in training. Can you spot it?
# MAGIC
# MAGIC | Today (DIY) | With data quality monitoring |
# MAGIC |---|---|
# MAGIC | Drift found by accident, or after a refresh | Drift measured every day, per feature |
# MAGIC | Custom scripts for each model | One monitor per table, generated dashboard |
# MAGIC | Hard to see *which* customers changed | Sliced by business type and country |
# MAGIC
# MAGIC **What you do:** click **Run all**, then follow the 👉 steps. *(About 10 minutes, mostly waiting for the first monitor run.)*

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering mlflow==3.16.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create 30 days of prediction history
# MAGIC We score the last 30 days of sign-ups with the champion model in **batch**. The same model and the same feature lookups as the endpoint, in one call.

# COMMAND ----------

# DBTITLE 1,Batch-score live sign-ups and the training baseline
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

fe = FeatureEngineeringClient()
client = MlflowClient(registry_uri="databricks-uc")
champ = client.get_model_version_by_alias(fq(MODEL), "champion")
threshold = float(champ.tags["review_threshold"])
model_uri = f"models:/{fq(MODEL)}@champion"


def score(cohort_df):
    return (fe.score_batch(model_uri=model_uri, df=cohort_df, result_type="double")
            .withColumnRenamed("prediction", "risk_score")
            .withColumn("prediction", (F.col("risk_score") >= threshold).cast("long"))
            .withColumn("model_version", F.lit(str(champ.version))))


cust = spark.table(CUSTOMERS)
labels = spark.table(LABELS).select("customer_id", F.col("is_high_risk").cast("long"))

live = cust.where("cohort = 'live'").select("customer_id", F.col("signup_ts").alias("scored_at"))
(score(live).join(labels, "customer_id")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PRED_LOG))

baseline = cust.where("cohort = 'train'").sample(fraction=0.35, seed=1).select("customer_id")
(score(baseline).join(labels, "customer_id")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BASELINE))

print(f"{PRED_LOG}: {spark.table(PRED_LOG).count():,} predictions over 30 days")
print(f"{BASELINE}: {spark.table(BASELINE).count():,} rows from training")
display(spark.table(PRED_LOG).orderBy(F.desc("scored_at")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Create the monitor
# MAGIC The monitor treats the table as an **inference log**: timestamp, model version, prediction and true label.
# MAGIC It's compared with the training baseline, day by day, and split by business type and country.

# COMMAND ----------

# DBTITLE 1,Create monitor
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()
try:
    w.quality_monitors.delete(table_name=fq(PRED_LOG))
    print("Removed the previous monitor")
    time.sleep(10)
except Exception:
    pass

monitor = w.quality_monitors.create(
    table_name=fq(PRED_LOG),
    assets_dir=MONITOR_ASSETS,
    output_schema_name=f"{CATALOG}.{SCHEMA}",
    baseline_table_name=fq(BASELINE),
    slicing_exprs=["business_type", "country"],
    inference_log=MonitorInferenceLog(
        timestamp_col="scored_at",
        granularities=["1 day"],
        model_id_col="model_version",
        prediction_col="prediction",
        label_col="is_high_risk",
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    ),
)
print("Monitor created. First run is starting.")

# COMMAND ----------

# DBTITLE 1,Wait for the first monitor run (about 5-10 min)
state = "PENDING"
for i in range(60):
    try:
        refreshes = w.quality_monitors.list_refreshes(table_name=fq(PRED_LOG)).refreshes or []
    except Exception:
        refreshes = []
    state = str(refreshes[0].state) if refreshes else "PENDING"
    if any(k in state for k in ("SUCCESS", "FAILED", "CANCELED")):
        break
    if i % 3 == 0:
        print(f"monitor running... ({state})")
    time.sleep(20)
print("✅ Monitor run finished" if "SUCCESS" in state else f"Monitor state: {state}. Re-run this cell in a minute.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Which features drifted?
# MAGIC The monitor writes its results to two tables in your schema: `predictions_log_profile_metrics` and `predictions_log_drift_metrics`.
# MAGIC This chart reads the drift table: how different each day is from training, per feature (0 = same, 1 = completely different).
# MAGIC
# MAGIC *Tip: if the chart shows as a table, click **+** → **Visualization** → **Line**, with `day` on X, `drift` on Y, grouped by `feature`.*

# COMMAND ----------

drift = spark.sql(f"""
  SELECT CAST(window.start AS DATE) AS day, column_name AS feature, ROUND(js_distance, 3) AS drift
  FROM {fq(PRED_LOG)}_drift_metrics
  WHERE drift_type = 'BASELINE' AND slice_key IS NULL AND js_distance IS NOT NULL
    AND column_name NOT IN ('scored_at', 'model_version', 'customer_id', 'prediction', 'is_high_risk', ':table')
""")
top = [r.feature for r in drift.groupBy("feature").agg(F.max("drift").alias("m")).orderBy(F.desc("m")).limit(5).collect()]
print("Most drifted features:", ", ".join(top))
display(drift.where(F.col("feature").isin(top)).orderBy("day", "feature"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · 👉 Explore the monitor (click through)
# MAGIC 1. Open the **monitor dashboard** (first link). It was generated for you.
# MAGIC    - Look at **drift over time**: when does it start?
# MAGIC    - Use the **slice** filter: pick `business_type` / `country`. Which group changed?
# MAGIC    - Look at **model performance** over time (precision, recall).
# MAGIC 2. Open the predictions table (second link) → **Quality** tab. This is where the monitor lives in Catalog Explorer.
# MAGIC
# MAGIC **Discuss:** what would you do next? *(Retrain on recent data, check the new segment with the risk team, add a feature?)*
# MAGIC In production this monitor runs on a schedule and can trigger an alert or a retraining job.

# COMMAND ----------

dashboard_id = w.quality_monitors.get(table_name=fq(PRED_LOG)).dashboard_id   # created after the first run
if dashboard_id:
    link("Open the monitor dashboard", f"/sql/dashboardsv3/{dashboard_id}")
else:
    print("Dashboard is still being created. Re-run this cell in a minute.")
link(f"Open {PRED_LOG} → Quality tab", f"/explore/data/{CATALOG}/{SCHEMA}/{PRED_LOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Done.** Now go back to **03_deploy → Part B** and call your endpoint.

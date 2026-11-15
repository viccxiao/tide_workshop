# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · Task 1: Evaluate
# MAGIC Runs automatically when a new model version is registered. Tests the new version and records every result on the model version as a tag, so the approver can see them.
# MAGIC If any test fails, the job stops here and the model never reaches the endpoint.

# COMMAND ----------

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

# COMMAND ----------

# MAGIC %run ../_config

# COMMAND ----------

import os, sys
sys.path.append(os.getcwd())
from checks import (check_beats_champion, check_input_data, check_minimum_quality,
                    check_sanctions_hit_raises_risk)

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

MODEL_NAME = dbutils.widgets.get("model_name") or fq(MODEL)
VERSION = dbutils.widgets.get("model_version")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")
fe = FeatureEngineeringClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"Evaluating {MODEL_NAME} v{VERSION} against champion v{champion.version}")


def score(uri, spine):
    return (fe.score_batch(model_uri=uri, df=spine, result_type="double")
            .select("customer_id", "prediction").toPandas().set_index("customer_id")["prediction"])


labels = spark.table(LABELS).select("customer_id", "is_high_risk")
cust = spark.table(CUSTOMERS)

# Holdout: the same fixed customers the models were never trained on
holdout = (cust.where("cohort = 'train'").where(F.crc32("customer_id") % 4 == 0)
           .select("customer_id").join(labels, "customer_id"))
# Recent: the last 7 days of sign-ups, where the new risky pattern is strongest
recent = (cust.where("cohort = 'live'").where(F.col("signup_ts") >= F.date_sub(F.current_date(), 7))
          .select("customer_id").join(labels, "customer_id"))

hold_pdf, recent_pdf = holdout.toPandas().set_index("customer_id"), recent.toPandas().set_index("customer_id")
cand_uri, champ_uri = f"models:/{MODEL_NAME}/{VERSION}", f"models:/{MODEL_NAME}/{champion.version}"
spine_h, spine_r = holdout.select("customer_id"), recent.select("customer_id")
hold_pdf["candidate"], hold_pdf["champion"] = score(cand_uri, spine_h), score(champ_uri, spine_h)
recent_pdf["candidate"], recent_pdf["champion"] = score(cand_uri, spine_r), score(champ_uri, spine_r)

# COMMAND ----------

# DBTITLE 1,Run the checks
results = []

# 1. Data contract on the feature table
results.append(check_input_data(spark.table(FEATURE_TABLE).toPandas()))

# 2. Absolute quality floor
results.append(check_minimum_quality(hold_pdf.is_high_risk, hold_pdf.candidate))

# 3. Ranks at least as well as the champion on the holdout (small tolerance) and on recent data (PR-AUC)
results.append(check_beats_champion("beats_champion_holdout", hold_pdf.is_high_risk,
                                    hold_pdf.candidate, hold_pdf.champion, tolerance=0.02))
results.append(check_beats_champion("beats_champion_recent", recent_pdf.is_high_risk,
                                    recent_pdf.candidate, recent_pdf.champion, tolerance=0.01))

# 4. Behaviour: give 300 clean customers a strong sanctions hit. Their score must not go down.
clean = (spark.table(FEATURE_TABLE).where("sanctions_hits = 0")
         .join(holdout.select("customer_id"), "customer_id").limit(300))
before = score(cand_uri, clean.select("customer_id"))
with_hit = clean.withColumn("sanctions_hits", F.lit(1)).withColumn("max_screening_match_score", F.lit(0.95))
after = (fe.score_batch(model_uri=cand_uri, df=with_hit, result_type="double")
         .select("customer_id", "prediction").toPandas().set_index("customer_id")["prediction"])
results.append(check_sanctions_hit_raises_risk(before.sort_index(), after.sort_index()))

display(pd.DataFrame([(r.name, "✅" if r.passed else "❌", r.detail) for r in results],
                     columns=["check", "result", "detail"]))

# COMMAND ----------

# DBTITLE 1,Record results on the model version
import numpy as np

passed = all(r.passed for r in results)
threshold = float(np.quantile(hold_pdf.candidate, 0.95))
for r in results:
    client.set_model_version_tag(MODEL_NAME, VERSION, f"test_{r.name}", r.as_tag())
client.set_model_version_tag(MODEL_NAME, VERSION, "test_result", "PASSED" if passed else "FAILED")
client.set_model_version_tag(MODEL_NAME, VERSION, "review_threshold", f"{threshold:.4f}")

mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"evaluate_v{VERSION}"):
    mlflow.log_params({"model_name": MODEL_NAME, "model_version": VERSION, "champion_version": champion.version})
    mlflow.log_metrics({f"{r.name}_passed": float(r.passed) for r in results})
    mlflow.log_text("\n".join(f"{r.name}: {r.as_tag()}" for r in results), "test_results.txt")

if not passed:
    failed = ", ".join(r.name for r in results if not r.passed)
    raise Exception(f"❌ v{VERSION} failed: {failed}. It will not be deployed.")
print(f"✅ v{VERSION} passed all checks. Waiting for approval.")

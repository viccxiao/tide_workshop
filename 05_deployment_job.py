# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Automated, approved deployments
# MAGIC
# MAGIC **The problem from 04:** the model is drifting. We need a new version, but we don't want anything going live without being tested and approved.
# MAGIC
# MAGIC **Goal:** a **deployment job** linked to the model. Every time a new version is registered, it runs automatically:
# MAGIC
# MAGIC **Evaluation** (tests) → **Approval_Check** (a person approves) → **Deployment** (champion alias + endpoint updated)
# MAGIC
# MAGIC | Today (DIY) | With a deployment job |
# MAGIC |---|---|
# MAGIC | Someone remembers to run the steps | Runs by itself when a new version is registered |
# MAGIC | Testing is manual, or skipped | The same checks run every time, with results saved on the version |
# MAGIC | Approval happens in chat | An **Approve** button on the model version, with an audit trail |
# MAGIC | New container built and rolled out | Alias moves, endpoint switches with no downtime |
# MAGIC
# MAGIC **What you do:** run each section in order and follow the 👉 steps. Don't use Run all here, because you'll approve in between.
# MAGIC
# MAGIC *Code CI/CD (tests and deploys when code changes) lives in GitHub. See the `cicd/` folder. This notebook covers the model side, which runs in Databricks.*

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering mlflow==3.16.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create the deployment job and link it to your model

# COMMAND ----------

# DBTITLE 1,Create the job
import os
from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

w = WorkspaceClient()
client = MlflowClient(registry_uri="databricks-uc")
here = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
JOB_NAME = f"kyx-deployment-job-{USER}"


def task(key, notebook, after=None):
    t = {"task_key": key, "notebook_task": {"notebook_path": f"{here}/pipeline/{notebook}"},
         "environment_key": "ml", "max_retries": 0}
    if after:
        t["depends_on"] = [{"task_key": after}]
    return t


settings = {
    "name": JOB_NAME,
    "max_concurrent_runs": 1,
    "queue": {"enabled": True},
    "parameters": [{"name": "model_name", "default": fq(MODEL)}, {"name": "model_version", "default": ""}],
    "environments": [{"environment_key": "ml", "spec": {
        "client": "3", "dependencies": ["databricks-feature-engineering", "mlflow==3.16.0"]}}],
    "tasks": [task("Evaluation", "evaluate"),
              task("Approval_Check", "approval_check", after="Evaluation"),   # name must start with "approval"
              task("Deployment", "deploy", after="Approval_Check")],
}

existing = [j.job_id for j in w.jobs.list(name=JOB_NAME)]
if existing:
    JOB_ID = existing[0]
    w.api_client.do("POST", "/api/2.2/jobs/reset", body={"job_id": JOB_ID, "new_settings": settings})
    print("Updated job", JOB_NAME)
else:
    JOB_ID = w.api_client.do("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
    print("Created job", JOB_NAME)

# Link: from now on, every new version of the model starts this job
current = getattr(client.get_registered_model(fq(MODEL)), "deployment_job_id", None)
if str(current or "") != str(JOB_ID):
    if current:
        client.update_registered_model(name=fq(MODEL), deployment_job_id="")   # clear the old link first
    client.update_registered_model(name=fq(MODEL), deployment_job_id=str(JOB_ID))
save_config(deployment_job_id=JOB_ID)
print(f"Linked {fq(MODEL)} -> job {JOB_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **Look at it:**
# MAGIC 1. Open the job (first link) → **Tasks**: three tasks in a row.
# MAGIC 2. Open the model (second link) → **Overview**: the **Deployment job** section shows it's linked.

# COMMAND ----------

link("Open the deployment job", f"/jobs/{JOB_ID}")
link(f"Open the model {MODEL}", f"/explore/data/models/{CATALOG}/{SCHEMA}/{MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Retrain on recent data
# MAGIC The new version also learns from the first three weeks of live sign-ups, so it has seen the new risky pattern.
# MAGIC **Registering it starts the deployment job automatically.** There's no "deploy" button to press.

# COMMAND ----------

# DBTITLE 1,Train and register a new version
import mlflow
import numpy as np
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
fe = FeatureEngineeringClient()


class RiskScorer(BaseEstimator):
    def __init__(self, pipe=None):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict_proba(X)[:, 1]


def train_and_register(run_name, spine, shuffle_labels=False):
    ts = fe.create_training_set(df=spine, label="is_high_risk", exclude_columns=["customer_id"],
                                feature_lookups=[FeatureLookup(table_name=fq(FEATURE_TABLE), lookup_key="customer_id")])
    pdf = ts.load_df().toPandas()
    y = pdf.pop("is_high_risk").astype(int)
    if shuffle_labels:
        y = y.sample(frac=1, random_state=1).reset_index(drop=True)
    cat = ["business_type", "country", "channel"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                             ("num", StandardScaler(), [c for c in pdf.columns if c not in cat])])
    pipe = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6, random_state=7))]).fit(pdf, y)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("training_rows", len(pdf))
        fe.log_model(model=RiskScorer(pipe), artifact_path="model", flavor=mlflow.sklearn, training_set=ts,
                     registered_model_name=fq(MODEL), infer_input_example=False, serialization_format="cloudpickle")
    v = next(m.version for m in client.search_model_versions(f"name='{fq(MODEL)}'") if m.run_id == run.info.run_id)
    print(f"Registered {MODEL} v{v}. The deployment job is starting.")
    return v


cust = spark.table(CUSTOMERS)
labels = spark.table(LABELS).select("customer_id", "is_high_risk")
recent_cutoff = F.date_sub(F.current_date(), 7)   # last 7 days are kept back for testing
spine = (cust.where(((F.col("cohort") == "train") & (F.crc32("customer_id") % 4 != 0)) |
                    ((F.col("cohort") == "live") & (F.col("signup_ts") < recent_cutoff)))
         .select("customer_id").join(labels, "customer_id"))
NEW_VERSION = train_and_register("v_retrained_on_recent_data", spine)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · 👉 Watch the tests
# MAGIC Run the cell below. It follows the job and stops when it's waiting for you (about 3-5 minutes).
# MAGIC Meanwhile, open the **job run** link and click the **Evaluation** task to see the test table.
# MAGIC
# MAGIC | Check | What it means |
# MAGIC |---|---|
# MAGIC | `data_contract` | No missing or duplicate customer IDs, scores and counts in valid ranges |
# MAGIC | `minimum_quality` | Catches at least 50% of high-risk customers at a 5% review rate |
# MAGIC | `beats_champion_holdout` | Ranks risky customers at least as well as the champion on the fixed test customers (PR-AUC) |
# MAGIC | `beats_champion_recent` | Same comparison on the last 7 days, the drifted period |
# MAGIC | `sanctions_hit_raises_risk` | Adding a sanctions hit raises the score for at least 95% of customers, and on average |

# COMMAND ----------

# DBTITLE 1,Follow the job run
import time


def follow(version, until="Approval_Check", done=("SUCCESS", "FAILED", "UPSTREAM_FAILED")):
    """Print the job's task states for this model version until `until` has finished."""
    run = None
    for _ in range(60):
        runs = [r for r in w.jobs.list_runs(job_id=JOB_ID, limit=10)
                if any(p.name == "model_version" and str(p.value) == str(version) for p in (r.job_parameters or []))]
        if runs:
            run = w.jobs.get_run(runs[0].run_id)
            states = {t.task_key: str(t.state.result_state or t.state.life_cycle_state).split(".")[-1]
                      for t in run.tasks}
            print(" | ".join(f"{k}: {v}" for k, v in states.items()))
            if states.get(until) in done:
                break
        else:
            print(f"waiting for the job to start for v{version}...")
        time.sleep(15)
    if run:
        link(f"Open the job run for v{version}", f"/jobs/{JOB_ID}/runs/{run.run_id}")
    return run


run = follow(NEW_VERSION)
for k, v in sorted(client.get_model_version(fq(MODEL), NEW_VERSION).tags.items()):
    if k.startswith("test_"):
        print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · 👉 Approve it
# MAGIC The job is now stopped at **Approval_Check**. That's on purpose: nothing goes live without a person saying yes.
# MAGIC 1. Open the new model version (link below).
# MAGIC 2. Look at the **tags**: every `test_...` result is recorded there, plus `test_result = PASSED`.
# MAGIC 3. Click **Approve** in the deployment job section. The job carries on by itself.
# MAGIC
# MAGIC *No Approve button? Run the next cell instead. It does the same thing (sets the approval tag and resumes the job).*

# COMMAND ----------

link(f"Open {MODEL} version {NEW_VERSION}", f"/explore/data/models/{CATALOG}/{SCHEMA}/{MODEL}/version/{NEW_VERSION}")

# COMMAND ----------

# DBTITLE 1,(Only if there's no Approve button) Approve from code
tags = client.get_model_version(fq(MODEL), NEW_VERSION).tags
assert tags.get("test_result") == "PASSED", f"v{NEW_VERSION} didn't pass its tests, so it can't be approved. See the Evaluation task."
client.set_model_version_tag(fq(MODEL), NEW_VERSION, "Approval_Check", "Approved")
w.jobs.repair_run(run_id=run.run_id, rerun_tasks=["Approval_Check", "Deployment"],
                  job_parameters={"model_name": fq(MODEL), "model_version": str(NEW_VERSION)})
print("Approved. The job is resuming.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · 👉 It goes live
# MAGIC Run the cell below to follow the job to the end. Then open your endpoint:
# MAGIC - It now serves the new version
# MAGIC - The old version is kept as the **previous** alias, so rollback is one alias move
# MAGIC - The endpoint kept answering requests during the switch

# COMMAND ----------

time.sleep(20)   # give the resumed run a moment to start
follow(NEW_VERSION, until="Deployment", done=("SUCCESS", "FAILED"))
champion_now = client.get_model_version_by_alias(fq(MODEL), "champion").version
assert str(champion_now) == str(NEW_VERSION), f"Champion is still v{champion_now}. Check the Deployment task."
print(f"✅ champion is now v{champion_now}")
link(f"Open endpoint {ENDPOINT}", f"/ml/endpoints/{ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Try to ship a bad model
# MAGIC This model is trained on **shuffled labels**, so it's basically guessing. Register it and watch what happens.
# MAGIC
# MAGIC *You can run this while the endpoint from Step 5 is still switching over.*

# COMMAND ----------

BAD_VERSION = train_and_register("v_bad_model_shuffled_labels", spine, shuffle_labels=True)
follow(BAD_VERSION)
bad_tags = client.get_model_version(fq(MODEL), BAD_VERSION).tags
print("Test results for the bad model:")
for k, v in sorted(bad_tags.items()):
    if k.startswith("test_"):
        print(f"  {k}: {v}")
print("champion is still v" + client.get_model_version_by_alias(fq(MODEL), "champion").version)

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 Open the job run above: **Evaluation failed**, so **Approval_Check** and **Deployment** never ran.
# MAGIC Open the bad version in Catalog: the `test_...` tags show exactly which checks failed. The endpoint is still on the good model.
# MAGIC
# MAGIC ✅ **Done.** New models now go live only when they pass the tests and someone approves them.
# MAGIC
# MAGIC **Next:** the `cicd/` folder shows the code side. It's the same setup defined as code, deployed from GitHub to dev, staging and prod.

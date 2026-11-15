# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train, track and register
# MAGIC
# MAGIC **Goal:** train two candidate models, compare them in MLflow, register them in Unity Catalog, and label the better one **champion**.
# MAGIC
# MAGIC | Today (DIY) | With MLflow + Unity Catalog |
# MAGIC |---|---|
# MAGIC | Results scattered across notebooks | Every run tracked: parameters, metrics, model files |
# MAGIC | "Which model is in prod?" | A `champion` alias points to the version in use |
# MAGIC | Model copied into a container per environment | One registered model, promoted by changing the alias |
# MAGIC | Features rebuilt for each model | Features looked up from the feature table automatically |
# MAGIC
# MAGIC **What you do:** click **Run all**, then follow the 👉 steps.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering mlflow==3.16.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Build the training set
# MAGIC We only provide **customer IDs + labels**. The features are looked up from the feature table, and that link is saved with the model.

# COMMAND ----------

# DBTITLE 1,Training set from the feature table
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
fe = FeatureEngineeringClient()

from pyspark.sql import functions as F

labels = (spark.table(LABELS).join(spark.table(CUSTOMERS).where("cohort = 'train'"), "customer_id")
          .select("customer_id", "is_high_risk"))

training_set = fe.create_training_set(
    df=labels,
    feature_lookups=[FeatureLookup(table_name=fq(FEATURE_TABLE), lookup_key="customer_id")],
    label="is_high_risk",
    exclude_columns=[],   # keep customer_id so we can pick the fixed holdout. It's removed before training.
)
train_pdf = training_set.load_df().toPandas()
print(f"{len(train_pdf):,} training rows, high-risk rate {train_pdf.is_high_risk.mean():.1%}")
display(train_pdf.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Train two candidates
# MAGIC - **v1:** logistic regression (simple baseline)
# MAGIC - **v2:** gradient boosted trees
# MAGIC
# MAGIC Both are logged to MLflow and registered in Unity Catalog with their feature lookups.
# MAGIC The business metric is **recall at a 5% review rate**: if analysts review the riskiest 5% of sign-ups, what share of high-risk customers do we catch?

# COMMAND ----------

# DBTITLE 1,Train, log and register
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RiskScorer(BaseEstimator):
    """Wraps a pipeline so the served model returns the high-risk PROBABILITY (0-1)."""
    def __init__(self, pipe=None):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict_proba(X)[:, 1]


# Fixed holdout: every 4th customer (by ID hash) is kept for testing, so every model is compared on the same customers
import zlib
holdout = train_pdf.pop("customer_id").map(lambda c: zlib.crc32(c.encode()) % 4 == 0)
y = train_pdf.pop("is_high_risk").astype(int)
X = train_pdf
CAT = ["business_type", "country", "channel"]
NUM = [c for c in X.columns if c not in CAT]
X_train, X_test, y_train, y_test = X[~holdout], X[holdout], y[~holdout], y[holdout]

candidates = {
    "v1_logistic_regression": LogisticRegression(max_iter=2000),
    "v2_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6, random_state=7),
}

runs, pipes = {}, {}
for run_name, clf in candidates.items():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM),
    ])
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
    pipes[run_name] = pipe
    proba = pipe.predict_proba(X_test)[:, 1]
    threshold = float(np.quantile(proba, 0.95))             # top 5% of sign-ups go to review
    recall = float((proba[y_test.values == 1] >= threshold).mean())
    metrics = {"roc_auc": roc_auc_score(y_test, proba),
               "pr_auc": average_precision_score(y_test, proba),
               "recall_at_5pct_review": recall}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_type": type(clf).__name__, **{k: v for k, v in clf.get_params().items()
                                                                  if isinstance(v, (int, float, str))}})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("review_threshold", threshold)
        fe.log_model(model=RiskScorer(pipe), artifact_path="model", flavor=mlflow.sklearn,
                     training_set=training_set, registered_model_name=fq(MODEL),
                     infer_input_example=False, serialization_format="cloudpickle")
    runs[run_name] = {**metrics, "threshold": threshold, "run_id": run.info.run_id}
    print(f"{run_name:26s} AUC={metrics['roc_auc']:.3f}  recall@5%review={recall:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Pick the champion
# MAGIC The better model by recall (then AUC, if tied) gets the **champion** alias, and the other one becomes **challenger**.
# MAGIC Anything that uses the model (endpoint, batch job) asks for `@champion`, so promoting a new model is just moving the alias. No new containers.

# COMMAND ----------

# DBTITLE 1,Set aliases
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")

versions = {v.run_id: v.version for v in client.search_model_versions(f"name='{fq(MODEL)}'")}
ranked = sorted(runs.items(), key=lambda kv: (kv[1]["recall_at_5pct_review"], kv[1]["roc_auc"]), reverse=True)
(champ_name, champ), (chall_name, chall) = ranked[0], ranked[1]

for alias, (name, r) in [("champion", (champ_name, champ)), ("challenger", (chall_name, chall))]:
    v = versions[r["run_id"]]
    client.set_registered_model_alias(fq(MODEL), alias, v)
    client.set_model_version_tag(fq(MODEL), v, "review_threshold", f"{r['threshold']:.4f}")
    client.update_model_version(fq(MODEL), v, description=f"{name}: AUC {r['roc_auc']:.3f}, "
                                f"recall at 5% review rate {r['recall_at_5pct_review']:.1%}")
    print(f"{alias:10s} -> version {v} ({name})")

client.update_registered_model(fq(MODEL), description="KYC onboarding risk model. Returns the probability that a new "
                               "business customer is high risk. Features are looked up from customer_features.")
save_config(champion_version=versions[champ["run_id"]], review_threshold=champ["threshold"])

# COMMAND ----------

# DBTITLE 1,Fallback (only if there's no online store)
# Without an online store, the endpoint can't look features up by itself. In that case we also register
# a copy of the champion that takes the features in the request. Skipped when the online store exists.
if not load_config().get("online_ok", False):
    from mlflow.models import infer_signature
    best = RiskScorer(pipes[champ_name])
    with mlflow.start_run(run_name=f"{champ_name}_features_in_request"):
        mlflow.sklearn.log_model(best, name="model", registered_model_name=fq(MODEL + "_direct"),
                                 signature=infer_signature(X_test.head(50), best.predict(X_test.head(50))),
                                 input_example=X_test.head(2), serialization_format="cloudpickle")
    v = max(int(m.version) for m in client.search_model_versions(f"name='{fq(MODEL + '_direct')}'"))
    client.set_registered_model_alias(fq(MODEL + "_direct"), "champion", v)
    print(f"Registered {MODEL}_direct v{v} (features sent in the request)")
else:
    print("Online store available. No fallback needed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · 👉 Explore (click through)
# MAGIC **In the MLflow experiment** (first link):
# MAGIC 1. Tick both runs → **Compare**. See metrics side by side.
# MAGIC 2. Open a run: parameters, metrics and the logged model are all saved.
# MAGIC
# MAGIC **In the registered model** (second link):
# MAGIC 3. See both versions with their **champion** / **challenger** aliases and descriptions.
# MAGIC 4. Open a version → **Lineage**: the model links back to `customer_features`, which links back to the raw tables.
# MAGIC
# MAGIC **Bonus:** go back to the feature table's lineage graph from notebook 01. The model now shows up downstream.

# COMMAND ----------

exp = mlflow.get_experiment_by_name(EXPERIMENT)
link("Open the MLflow experiment", f"/ml/experiments/{exp.experiment_id}")
link(f"Open the registered model {MODEL}", f"/explore/data/models/{CATALOG}/{SCHEMA}/{MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Done.** Next, open **03_deploy** and start the endpoint. It takes a few minutes, so we'll do monitoring while it starts.

# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC **What this does**
# MAGIC 1. Creates your own workshop schema, so nothing you do touches anyone else's work
# MAGIC 2. Generates fake KYC onboarding data (customers, document checks, screening hits, risk labels)
# MAGIC 3. Checks your workspace has everything the workshop needs, and shows ✅ / ❌
# MAGIC 4. Starts creating the online feature store (it takes 10-20 minutes, so we start it now)
# MAGIC
# MAGIC **What you do**
# MAGIC 1. In the **catalog** box at the top, type a catalog you can create schemas in (e.g. a dev or sandbox catalog)
# MAGIC 2. Click **Run all**
# MAGIC 3. Check the summary at the bottom is all ✅
# MAGIC
# MAGIC *Compute: Serverless. Takes about 3 minutes.*

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "", "catalog")

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# DBTITLE 1,Create your schema and save your settings
CATALOG = dbutils.widgets.get("catalog").strip() or spark.catalog.currentCatalog()
results = []  # (check, ok, detail)

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}` "
              "COMMENT 'KYX MLOps workshop (synthetic data only)'")
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"USE SCHEMA `{SCHEMA}`")
    save_config(catalog=CATALOG)
    results.append(("Create schema", True, f"{CATALOG}.{SCHEMA}"))
except Exception as e:
    results.append(("Create schema", False, str(e)[:200]))
    raise Exception(f"Can't create a schema in catalog '{CATALOG}'. Pick a catalog you can write to. ({e})")

print(f"Your workshop space: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Generate fake KYC onboarding data
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
TODAY = pd.Timestamp(datetime.utcnow().date())
N_TRAIN, LIVE_DAYS, LIVE_PER_DAY = 15000, 30, 150

BUSINESS = ["Retail", "Hospitality", "Construction", "Professional services",
            "E-commerce", "Transport", "Money services", "Crypto"]
BUSINESS_P = [0.18, 0.14, 0.14, 0.20, 0.14, 0.10, 0.05, 0.05]
COUNTRY, COUNTRY_P = ["GB", "IE", "Other"], [0.90, 0.03, 0.07]


def make_customers(n, start, end, new_pattern_share=None):
    """n new business customers who signed up between start and end."""
    days = rng.integers(0, (end - start).days + 1, n)
    signup = start + pd.to_timedelta(days, "D") + pd.to_timedelta(rng.integers(7 * 60, 22 * 60, n), "m")
    df = pd.DataFrame({
        "signup_ts": signup,
        "business_type": rng.choice(BUSINESS, n, p=BUSINESS_P),
        "country": rng.choice(COUNTRY, n, p=COUNTRY_P),
        "channel": rng.choice(["app", "web"], n, p=[0.7, 0.3]),
        "company_age_months": np.clip(rng.lognormal(3.4, 0.9, n), 0, 360).round().astype(int),
        "director_count": rng.choice([1, 2, 3, 4], n, p=[0.55, 0.30, 0.10, 0.05]),
        "declared_turnover_gbp": (rng.lognormal(11.3, 1.0, n) / 100).round() * 100,
    })
    df["new_pattern"] = False
    if new_pattern_share is not None:
        # New risk pattern appearing in live traffic: brand new e-commerce companies from outside GB/IE
        share = new_pattern_share(df.signup_ts)
        hit = rng.random(n) < share
        df.loc[hit, "business_type"] = "E-commerce"
        df.loc[hit, "country"] = "Other"
        df.loc[hit, "company_age_months"] = rng.integers(0, 4, hit.sum())
        df.loc[hit, "channel"] = "web"
        df["new_pattern"] = hit
    return df


train = make_customers(N_TRAIN, TODAY - timedelta(days=400), TODAY - timedelta(days=LIVE_DAYS + 1))
train["cohort"] = "train"

live_start = TODAY - timedelta(days=LIVE_DAYS)
ramp = lambda ts: np.clip(((ts - live_start).dt.days - 14) / 16, 0, 1) * 0.35   # 0% until day 15, then up to 35%
live = make_customers(LIVE_DAYS * LIVE_PER_DAY, live_start, TODAY - timedelta(days=1), new_pattern_share=ramp)
live["cohort"] = "live"

cust = pd.concat([train, live], ignore_index=True).sort_values("signup_ts").reset_index(drop=True)
cust.insert(0, "customer_id", [f"C{i:06d}" for i in range(1, len(cust) + 1)])
n = len(cust)

# Hidden "bad actor" flag drives behaviour + label (the model never sees this column)
logit = (-3.4
         + 1.3 * (cust.company_age_months < 6)
         + 1.0 * cust.business_type.isin(["Money services", "Crypto"])
         + 0.8 * (cust.country == "Other")
         + 0.3 * (cust.channel == "web")
         + 1.8 * cust.new_pattern)
bad = rng.random(n) < 1 / (1 + np.exp(-logit))

# Document checks: bad actors retry more and get lower face-match scores
n_checks = 1 + rng.poisson(np.where(bad, 1.6, 0.25))
doc_rows = []
for cid, k, b, ts in zip(cust.customer_id, n_checks, bad, cust.signup_ts):
    for j in range(k):
        face = float(np.clip(rng.normal(0.70 if b else 0.90, 0.10), 0, 1))
        result = "pass" if (face > 0.78 and j == k - 1) else rng.choice(["fail", "manual_review"], p=[0.7, 0.3])
        doc_rows.append((f"{cid}-D{j + 1}", cid, ts + timedelta(minutes=5 + 7 * j),
                         rng.choice(["passport", "driving_licence", "national_id"], p=[0.5, 0.4, 0.1]),
                         str(result), round(face, 3)))
docs = pd.DataFrame(doc_rows, columns=["check_id", "customer_id", "check_ts", "document_type", "result", "face_match_score"])

# Screening hits (PEP / sanctions / adverse media): rare, more likely for bad actors
hit_rows = []
for cid, b, ts in zip(cust.customer_id, bad, cust.signup_ts):
    for htype, p_bad, p_good in [("PEP", 0.10, 0.015), ("sanctions", 0.06, 0.002), ("adverse_media", 0.20, 0.03)]:
        if rng.random() < (p_bad if b else p_good):
            hit_rows.append((f"{cid}-{htype}", cid, ts + timedelta(minutes=2), htype,
                             round(float(rng.uniform(0.75, 1.0) if b else rng.uniform(0.6, 0.9)), 3)))
hits = pd.DataFrame(hit_rows, columns=["hit_id", "customer_id", "hit_ts", "hit_type", "match_score"])

# Label: was the customer later confirmed high risk after review? (a little noise, like real life)
noise = rng.random(n)
label = np.where(bad, noise > 0.05, noise < 0.012).astype(int)   # 5% of bad actors missed, 1.2% false alarms
labels = pd.DataFrame({"customer_id": cust.customer_id, "is_high_risk": label,
                       "reviewed_at": cust.signup_ts + pd.to_timedelta(rng.integers(3, 30, n), "D")})

customers = cust.drop(columns=["new_pattern"])
for name, pdf, comment in [
    (CUSTOMERS, customers, "Business customers signing up to the bank (synthetic)"),
    (DOC_CHECKS, docs, "Identity document verification attempts, one row per attempt (synthetic)"),
    (SCREENING, hits, "PEP, sanctions and adverse media screening hits (synthetic)"),
    (LABELS, labels, "Outcome of manual risk review: 1 = confirmed high risk (synthetic)"),
]:
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name))
    spark.sql(f"COMMENT ON TABLE {name} IS '{comment}'")
    print(f"{name:16s} {len(pdf):>7,} rows")

results.append(("Generate data", True, f"{len(customers):,} customers, high-risk rate {label.mean():.1%}"))

# COMMAND ----------

# DBTITLE 1,Check what the workshop needs
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# MLflow + Unity Catalog model registry
try:
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    results.append(("MLflow experiment", True, EXPERIMENT))
except Exception as e:
    results.append(("MLflow experiment", False, str(e)[:200]))

try:
    spark.sql(f"SHOW GRANTS ON SCHEMA `{CATALOG}`.`{SCHEMA}`").collect()
    results.append(("Register models in your schema", True, "you own the schema"))
except Exception as e:
    results.append(("Register models in your schema", False, str(e)[:200]))

# Model serving
try:
    _ = list(w.serving_endpoints.list())
    results.append(("Model Serving", True, "can list endpoints"))
except Exception as e:
    results.append(("Model Serving", False, str(e)[:200]))

# Data quality monitoring (Lakehouse Monitoring)
try:
    w.quality_monitors.get(table_name=f"{CATALOG}.{SCHEMA}.{CUSTOMERS}")
    results.append(("Data quality monitoring", True, "API reachable"))
except Exception as e:
    msg = str(e)
    ok = any(k in msg.lower() for k in ("does not exist", "not found", "cannot find", "resource_does_not_exist"))
    results.append(("Data quality monitoring", ok, "API reachable" if ok else msg[:200]))

# COMMAND ----------

# DBTITLE 1,Start creating the online feature store (shared, takes 10-20 min)
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
try:
    store = fe.get_online_store(name=ONLINE_STORE)
    if store is None:
        fe.create_online_store(name=ONLINE_STORE, capacity="CU_1")
        results.append(("Online feature store", True, f"'{ONLINE_STORE}' is being created. It'll be ready by notebook 01."))
    else:
        results.append(("Online feature store", True, f"'{ONLINE_STORE}' already exists"))
    save_config(online_ok=True)
except Exception as e:
    msg = str(e)
    if "already exists" in msg.lower():
        results.append(("Online feature store", True, f"'{ONLINE_STORE}' already exists"))
        save_config(online_ok=True)
    else:
        results.append(("Online feature store", False, "Not available. The workshop still works: "
                        "the endpoint will receive features in the request instead. " + msg[:150]))
        save_config(online_ok=False)

# COMMAND ----------

# DBTITLE 1,Summary
summary = pd.DataFrame([(c, "✅" if ok else "❌", d) for c, ok, d in results], columns=["check", "status", "detail"])
display(summary)
if (summary.status == "❌").any():
    print("Something needs fixing. Share this table with your workspace admin or the workshop host.")
else:
    print("All good. Open 01_feature_engineering next.")

# COMMAND ----------

dbutils.notebook.exit(summary.to_json(orient="records"))

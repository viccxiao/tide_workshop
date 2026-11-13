# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Feature engineering
# MAGIC
# MAGIC **Goal:** turn raw onboarding data into a **feature table**. That's one governed table that the whole team can find, understand and reuse, for training *and* for live scoring.
# MAGIC
# MAGIC | Today (DIY) | With a feature table |
# MAGIC |---|---|
# MAGIC | Features built inside each notebook | Built once, reused by every model |
# MAGIC | "Where did this column come from?" | Lineage shows the source tables |
# MAGIC | Files at hardcoded paths | A table in Unity Catalog with an owner and descriptions |
# MAGIC | Backend rebuilds inputs for the model | Model looks features up by itself (online store) |
# MAGIC
# MAGIC **What you do:** click **Run all**, then follow the 👉 steps.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Build the features
# MAGIC One row per customer. Every feature is defined here, once, in plain Spark.

# COMMAND ----------

# DBTITLE 1,Feature definitions
from pyspark.sql import functions as F

customers = spark.table(CUSTOMERS)
docs = spark.table(DOC_CHECKS)
hits = spark.table(SCREENING)

doc_features = docs.groupBy("customer_id").agg(
    F.count("*").alias("doc_attempts"),
    F.sum(F.when(F.col("result") != "pass", 1).otherwise(0)).alias("doc_failed_attempts"),
    F.avg("face_match_score").alias("avg_face_match_score"),
    F.min("face_match_score").alias("min_face_match_score"),
)

screening_features = hits.groupBy("customer_id").agg(
    F.sum(F.when(F.col("hit_type") == "PEP", 1).otherwise(0)).alias("pep_hits"),
    F.sum(F.when(F.col("hit_type") == "sanctions", 1).otherwise(0)).alias("sanctions_hits"),
    F.sum(F.when(F.col("hit_type") == "adverse_media", 1).otherwise(0)).alias("adverse_media_hits"),
    F.max("match_score").alias("max_screening_match_score"),
)

features = (
    customers
    .join(doc_features, "customer_id", "left")
    .join(screening_features, "customer_id", "left")
    .select(
        "customer_id",
        "business_type",
        "country",
        "channel",
        F.col("company_age_months").cast("int"),
        F.col("director_count").cast("int"),
        F.col("declared_turnover_gbp").cast("double"),
        F.coalesce("doc_attempts", F.lit(0)).cast("int").alias("doc_attempts"),
        F.coalesce("doc_failed_attempts", F.lit(0)).cast("int").alias("doc_failed_attempts"),
        F.coalesce(F.col("doc_failed_attempts") / F.col("doc_attempts"), F.lit(0.0)).cast("double").alias("doc_fail_rate"),
        F.coalesce("avg_face_match_score", F.lit(0.0)).cast("double").alias("avg_face_match_score"),
        F.coalesce("min_face_match_score", F.lit(0.0)).cast("double").alias("min_face_match_score"),
        F.coalesce("pep_hits", F.lit(0)).cast("int").alias("pep_hits"),
        F.coalesce("sanctions_hits", F.lit(0)).cast("int").alias("sanctions_hits"),
        F.coalesce("adverse_media_hits", F.lit(0)).cast("int").alias("adverse_media_hits"),
        F.coalesce("max_screening_match_score", F.lit(0.0)).cast("double").alias("max_screening_match_score"),
    )
)
display(features.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Save it as a feature table
# MAGIC A feature table is a normal Unity Catalog table with a **primary key** (`customer_id`: which customer the row is about).
# MAGIC That's what lets training and serving look features up automatically.

# COMMAND ----------

# DBTITLE 1,Create the feature table
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

if spark.catalog.tableExists(fq(FEATURE_TABLE)):
    fe.write_table(name=fq(FEATURE_TABLE), df=features, mode="merge")
    print("Updated", fq(FEATURE_TABLE))
else:
    fe.create_table(
        name=fq(FEATURE_TABLE),
        primary_keys=["customer_id"],
        df=features,
        description="KYC onboarding risk features, one row per business customer. "
                    "Built from customers, document_checks and screening_hits.",
    )
    print("Created", fq(FEATURE_TABLE))

# Change data feed lets the online store pick up changes
spark.sql(f"ALTER TABLE {fq(FEATURE_TABLE)} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# Column descriptions, so anyone can understand the features
descriptions = {
    "customer_id": "Customer identifier (primary key)",
    "business_type": "Declared business sector",
    "country": "Country of incorporation (GB, IE, Other)",
    "channel": "Sign-up channel (app or web)",
    "company_age_months": "Months since the company was incorporated",
    "director_count": "Number of company directors",
    "declared_turnover_gbp": "Annual turnover declared at sign-up, in GBP",
    "doc_attempts": "Number of identity document verification attempts",
    "doc_failed_attempts": "Attempts that failed or went to manual review",
    "doc_fail_rate": "Share of document attempts that did not pass",
    "avg_face_match_score": "Average selfie-to-document face match score (0-1)",
    "min_face_match_score": "Lowest face match score across attempts (0-1)",
    "pep_hits": "Politically exposed person screening hits",
    "sanctions_hits": "Sanctions list screening hits",
    "adverse_media_hits": "Adverse media screening hits",
    "max_screening_match_score": "Highest screening match confidence (0-1)",
}
for col, text in descriptions.items():
    spark.sql(f"ALTER TABLE {fq(FEATURE_TABLE)} ALTER COLUMN {col} COMMENT '{text}'")
print("Descriptions added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · 👉 Explore your feature table (click through)
# MAGIC Open the link below, then:
# MAGIC 1. **Overview** tab: read the table and column descriptions. Note that you're the owner.
# MAGIC 2. **Sample data** tab: see the features.
# MAGIC 3. **Lineage** tab → **See lineage graph**: the three raw tables flow into your feature table. That answers "where did this feature come from?"
# MAGIC 4. **History** tab: every change to the table, who made it and when.
# MAGIC 5. In the left sidebar, open **Machine Learning → Features**: your table is listed there, so other people can find and reuse it.
# MAGIC
# MAGIC *After notebook 02, come back to the lineage graph. Your model will show up there too.*

# COMMAND ----------

link(f"Open {FEATURE_TABLE} in Catalog Explorer", f"/explore/data/{CATALOG}/{SCHEMA}/{FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Publish to the online store
# MAGIC The online store is a low-latency copy of the feature table. When the model is served, it looks up a customer's features here by itself, so **the calling service only sends a `customer_id`**.
# MAGIC
# MAGIC *If the online store isn't ready yet, this cell waits for it (it was started in 00_setup).*

# COMMAND ----------

# DBTITLE 1,Publish
import time

ONLINE_OK = load_config().get("online_ok", False)
if not ONLINE_OK:
    print("Online store not available in this workspace. Skipping. "
          "The model endpoint will receive features in the request instead.")
else:
    store = None
    for i in range(120):
        s = fe.get_online_store(name=ONLINE_STORE)
        state = str(getattr(s, "state", "")) if s is not None else "creating"
        if s is not None and any(k in state.upper() for k in ("AVAIL", "ONLINE", "READY", "ACTIVE")):
            store = s
            break
        if i % 3 == 0:
            print(f"waiting for online store ({state})...")
        time.sleep(10)

    if store is None:
        print("Online store still not ready. Re-run this cell in a few minutes.")
    else:
        fe.publish_table(
            online_store=store,
            source_table_name=fq(FEATURE_TABLE),
            online_table_name=fq(ONLINE_TABLE),
            publish_mode="TRIGGERED",
        )
        print(f"Published {fq(FEATURE_TABLE)} -> {fq(ONLINE_TABLE)}")
        link(f"Open {ONLINE_TABLE} in Catalog Explorer", f"/explore/data/{CATALOG}/{SCHEMA}/{ONLINE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Done.** You now have a documented, traceable feature table that's ready for training (notebook 02) and live scoring (notebook 03).

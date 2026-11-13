# Databricks notebook source
# MAGIC %md
# MAGIC # Shared settings
# MAGIC Every workshop notebook runs this first with `%run ./_config`. You don't need to open or change it.
# MAGIC
# MAGIC It works out **who you are** and **where your workshop objects live**, so no notebook has a hardcoded path.

# COMMAND ----------

import json, os, re

ME = spark.sql("SELECT current_user()").first()[0]
USER = re.sub(r"[^a-z0-9]", "_", ME.split("@")[0].lower())
CONFIG_PATH = f"/Workspace/Users/{ME}/.kyx_workshop_config.json"


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(**updates):
    cfg = load_config()
    cfg.update(updates)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


CFG = load_config()
CATALOG = CFG.get("catalog")
SCHEMA = f"kyx_workshop_{USER}"

# Raw data (created by 00_setup)
CUSTOMERS = "customers"
DOC_CHECKS = "document_checks"
SCREENING = "screening_hits"
LABELS = "risk_labels"

# Workshop objects
FEATURE_TABLE = "customer_features"
ONLINE_STORE = "kyx-workshop-online"  # shared by everyone in the workspace
ONLINE_TABLE = "customer_features_online"
MODEL = "kyx_risk_model"
ENDPOINT = f"kyx-risk-{USER.replace('_', '-')}"[:60]
EXPERIMENT = f"/Users/{ME}/kyx_workshop_experiment"
PRED_LOG = "predictions_log"
BASELINE = "predictions_baseline"
MONITOR_ASSETS = f"/Workspace/Users/{ME}/kyx_workshop_monitoring"


def fq(name):
    """Fully qualified name: catalog.schema.name"""
    return f"{CATALOG}.{SCHEMA}.{name}"


def host():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient().config.host.rstrip("/")


def link(label, path):
    """Show a clickable link to a page in this workspace."""
    displayHTML(f'<a href="{host()}{path}" target="_blank" style="font-size:16px">👉 {label}</a>')


if CATALOG:
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"USE SCHEMA `{SCHEMA}`")
    print(f"Working in {CATALOG}.{SCHEMA} as {ME}")
else:
    print("No settings yet. Run 00_setup first.")

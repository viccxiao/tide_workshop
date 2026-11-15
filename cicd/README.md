# CI/CD: the code side

Notebook `05_deployment_job` covers **model CD**: a new model version is tested, approved and deployed, all inside Databricks.
This folder covers **code CI/CD**: when someone changes the code, GitHub tests it and deploys it to each environment.

| | Code CI/CD (this folder) | Model CD (notebook 05) |
|---|---|---|
| Starts when | Code changes (pull request, merge, release) | A new model version is registered |
| Runs in | GitHub Actions | Databricks deployment job |
| Does | Tests the code, deploys jobs to dev / staging / prod | Tests the model, waits for approval, moves `champion`, updates the endpoint |

## What's here

| File | What it is |
|---|---|
| `../databricks.yml` | The **bundle**: the whole project as code, with `dev`, `staging` and `prod` targets. Same code everywhere, different catalog and schema per environment. |
| `resources/jobs.yml` | The training job (features → train) and the deployment job (Evaluation → Approval_Check → Deployment) |
| `github-actions/mlops.yml` | Example pipeline: tests on pull requests, deploy to staging on merge, deploy to prod on release |
| `tests/test_checks.py` | Unit tests for the model checks in `pipeline/checks.py`. They run without Databricks. |

## How it flows

1. A developer changes code and opens a **pull request** → unit tests + `bundle validate`
2. The pull request is merged to `main` → `bundle deploy -t staging`, then the training job runs in staging
3. Training registers a new model version → the **deployment job** tests it and waits for approval (notebook 05)
4. A **release** is published → `bundle deploy -t prod`

GitHub never trains or serves models. It tells Databricks what to deploy, and Databricks does the rest.

## Mapping to more environments

Add a target per environment in `databricks.yml` (for example a separate `shadow` target).
Targets can point at different workspaces (`workspace.host`) or at different catalogs in the same workspace.
Instead of building a container per environment, each environment uses the same registered model, and the
endpoint is pointed at an alias.

## Using it with Argo CD or another deployment tool

Anything that can run a shell command can deploy the bundle:

```bash
databricks bundle deploy -t staging --var catalog=<catalog>
```

So the bundle can be called from an existing GitOps pipeline, the same way infrastructure is deployed today.

## Try it from your laptop

```bash
pip install pytest pandas numpy && pytest cicd/tests       # unit tests
databricks bundle validate -t dev --var catalog=<catalog>  # check the bundle
databricks bundle deploy   -t dev --var catalog=<catalog>  # deploy to your own dev copy
```

You can also deploy from the workspace UI: open this Git folder and use the **Deploy** button in the bundle panel.

## Learn more

- MLOps Stacks, a full template: `databricks bundle init mlops-stacks`
- Docs: search "Databricks Asset Bundles" and "MLflow deployment jobs"

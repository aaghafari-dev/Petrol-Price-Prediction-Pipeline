# WTI Oil Price Predictor Pro v6.1

**Production-Oriented Machine Learning \& MLOps Platform for WTI Crude Oil Price Prediction**

&#x20; 
Target: WTI Crude Oil Futures (`CL=F`)  
Prediction Horizon: Next Business / Trading Day  
Architecture: Local-first, MLOps-ready, Cloud-ready  
Primary Interface: Browser-based Professional Control Center

\---

## 1\. Overview

**WTI Oil Price Predictor Pro** is a professional machine-learning platform designed to acquire, validate, engineer, train, evaluate, govern, monitor, and serve WTI crude-oil price predictions.

The system is designed around a complete ML lifecycle:


                    ┌─────────────────────┐
                    │    Yahoo Finance    │
                    │       CL=F          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion    │
                    │   Local Data Cache  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Quality     │
                    │ Validation \& Checks  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Leakage-safe        │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌─────────────────────────────────────┐
             │       Model Benchmark Center        │
             │                                     │
             │ XGBoost / Random Forest /           │
             │ Extra Trees / HistGradientBoosting  │
             └─────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Optional Fine       │
                    │ Tuning              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Time-Series         │
                    │ Evaluation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Persistence         │
                    │ Baseline            │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
                 REJECTED             ELIGIBLE
                     │                   │
                     │                   ▼
                     │        ┌────────────────────┐
                     │        │ Model Governance   │
                     │        │ Champion Selection │
                     │        └─────────┬──────────┘
                     │                  │
                     │        ┌─────────┼──────────┐
                     │        │         │          │
                     ▼        ▼         ▼          ▼
                   Stop    MLflow      S3       EKS/K8s
                                      Backup    Deployment
                                                │
                                                ▼
                                  ┌──────────────────────┐
                                  │ Next-Day Prediction  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │      Dashboard       │
                                  └──────────────────────┘
```

\---

# 2\. Main Objectives

The platform has six major objectives:

### 2.1 Data Acquisition

Automatically retrieve WTI crude-oil market data using:


Yahoo Finance
      ↓
CL=F
      ↓
Local cache
```

\---

### 2.2 Machine Learning

Train and evaluate multiple regression models:

* XGBoost
* Random Forest
* Extra Trees
* HistGradientBoosting

The architecture is intentionally based on a model factory so additional models can be added without redesigning the complete application.

\---

### 2.3 Model Selection

The **Model Benchmark Center** allows the user to evaluate multiple candidate models using a consistent evaluation methodology.

The system compares:

```text
Candidate Model
       vs
Persistence Baseline
```

The goal is not simply to select the model with the lowest RMSE.

The model must demonstrate meaningful predictive value compared with a strong naive baseline.

\---

### 2.4 Model Governance

The system maintains a distinction between:

```text
Candidate
   ↓
Evaluated
   ↓
Eligible
   ↓
Champion
```

A model should not become Champion merely because it successfully trained.

It must satisfy the governance rules.

\---

### 2.5 Next-Day Prediction

The primary business output is:

> \*\*Prediction of the next available trading/business day's WTI price.\*\*

The system uses the latest available market observation to construct a dedicated prediction feature row.

This is intentionally different from simply predicting the final training row.

\---

### 2.6 MLOps / Cloud Deployment

The project is designed to integrate with:

* MLflow
* AWS S3
* Kubernetes
* Amazon EKS
* Docker

These components are optional during local development but form the foundation for production deployment.

\---

# 3\. Technology Stack

|Component|Technology|
|-|-|
|Language|Python 3.12+|
|Data Processing|pandas|
|Numerical Computing|NumPy|
|Machine Learning|scikit-learn|
|Gradient Boosting|XGBoost|
|Market Data|Yahoo Finance|
|API|FastAPI|
|Web Server|Uvicorn|
|Experiment Tracking|MLflow|
|Cloud Storage|AWS S3 / boto3|
|Containerization|Docker|
|Orchestration|Kubernetes|
|Cloud Kubernetes|Amazon EKS|
|Testing|pytest|
|Model Serialization|joblib|
|Configuration|Environment variables / `.env`|

\---

# 4\. Project Architecture

The project follows a modular architecture.

```text
WTI-Oil-Price-Predictor-Pro\_v6.1/
│
├── config/
│   ├── \_\_init\_\_.py
│   └── project\_config.py
│
├── data/
│   ├── raw/
│   ├── models/
│   └── champion/
│
├── artifacts/
│
├── logs/
│
├── k8s/
│   └── deployment.yaml
│
├── src/
│   ├── \_\_init\_\_.py
│   │
│   ├── data/
│   │   ├── \_\_init\_\_.py
│   │   ├── data\_ingestion.py
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── \_\_init\_\_.py
│   │   ├── factory.py
│   │   └── evaluator.py
│   │
│   ├── pipeline/
│   │   ├── \_\_init\_\_.py
│   │   └── orchestrator.py
│   │
│   ├── infrastructure/
│   │   ├── \_\_init\_\_.py
│   │   └── cloud.py
│   │
│   └── serving/
│       ├── \_\_init\_\_.py
│       └── api.py
│
├── tests/
│   └── test\_smoke.py
│
├── run.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── docs\_FLOWCHART.md
└── README.md
```

\---

# 5\. Directory Responsibilities

## 5.1 `config/`

Contains centralized application configuration.

### `config/project\_config.py`

This file defines:

* project paths
* WTI symbol
* data window
* number of test folds
* available models
* MLflow URI
* MLflow experiment
* S3 configuration
* AWS region
* Kubernetes deployment permissions
* cache lifetime
* application version

Example:

```python
SYMBOL = "CL=F"
```

This is the market instrument used by the application.

\---

# 6\. `data/`

The data directory is divided into three logical areas.

```text
data/
├── raw/
├── models/
└── champion/
```

## `data/raw/`

Stores locally cached market data.

Purpose:

```text
Internet
   ↓
Yahoo Finance
   ↓
Local cache
```

This provides resilience when Yahoo Finance is temporarily unavailable.

\---

## `data/models/`

Stores trained model artifacts and model manifests.

\---

## `data/champion/`

Contains the currently promoted production model.

Typical files:

```text
champion\_model.joblib
champion\_manifest.json
```

The Champion is the model used for production prediction.

\---

# 7\. Data Ingestion

File:

```text
src/data/data\_ingestion.py
```

Responsibilities:

1. Connect to Yahoo Finance.
2. Retrieve WTI data.
3. Validate returned data.
4. Normalize the dataset.
5. Save/cache the data locally.
6. Reuse the cache when appropriate.

The default instrument is:

```text
CL=F
```

\---

# 8\. Feature Engineering

File:

```text
src/data/features.py
```

This module converts raw market observations into machine-learning features.

Conceptually:

```text
Raw OHLCV
    │
    ├── Lag features
    ├── Rolling features
    ├── Return features
    └── Statistical features
             │
             ▼
       ML Feature Matrix
```

Feature engineering must be **time-aware**.

The system must not use future information to construct a feature for an earlier observation.

This prevents data leakage.

\---

# 9\. Model Factory

File:

```text
src/models/factory.py
```

The model factory provides a single interface for creating different models.

Supported models:

```text
XGBoost
Random Forest
Extra Trees
HistGradientBoosting
```

Conceptually:

```text
User Selection
      │
      ▼
Model Factory
      │
      ├── XGBoost
      ├── Random Forest
      ├── Extra Trees
      └── HistGradientBoosting
```

This architecture makes it easier to add additional algorithms later.

\---

# 10\. Model Evaluation

File:

```text
src/models/evaluator.py
```

The system uses chronological evaluation rather than randomly shuffling financial time-series data.

This is important because:

```text
Past
 ↓
Future
```

must remain the direction of information flow.

Random train/test splitting can introduce unrealistic leakage in time-series forecasting.

\---

# 11\. Persistence Baseline

The platform uses a persistence baseline.

The basic idea is:

```text
Tomorrow's prediction ≈ Today's price
```

Although simple, this is an important baseline for financial forecasting.

Suppose:

```text
Today's price = $72.50
```

The persistence baseline predicts:

```text
Tomorrow = $72.50
```

If a sophisticated ML model produces:

```text
RMSE = 6.44
```

while persistence produces:

```text
RMSE = 2.43
```

the ML model should **not** become Champion.

This is why the system can correctly produce a message such as:

```text
Rejected:
Challenger RMSE does not beat persistence baseline
```

This is a governance decision, not necessarily a software error.

\---

# 12\. Model Benchmark Center

The Benchmark Center is the main model-selection component.

The user selects:

```text
☑ XGBoost
☑ Random Forest
☑ Extra Trees
☐ HistGradientBoosting
```

Then optionally selects:

```text
☑ Fine Tuning
```

and:

```text
☑ Auto Promote Best Eligible Model
```

The benchmark then performs:

```text
Selected Models
       ↓
Feature Engineering
       ↓
Time-Series Evaluation
       ↓
RMSE / MAE
       ↓
Persistence Baseline
       ↓
Eligibility Test
       ↓
Best Eligible Model
       ↓
Champion
```

\---

# 13\. Fine Tuning

Fine tuning is optional.

Without fine tuning:

```text
Model
 ↓
Default/Configured Hyperparameters
 ↓
Evaluation
```

With fine tuning:

```text
Model
 ↓
Hyperparameter Search
 ↓
Time-Series Cross Validation
 ↓
Best Parameters
 ↓
Final Evaluation
```

Fine tuning is computationally more expensive.

Recommended workflow:

```text
1. Benchmark all models
          ↓
2. Identify strongest candidates
          ↓
3. Fine tune top candidates
          ↓
4. Compare against baseline
          ↓
5. Promote best eligible model
```

\---

# 14\. Model Governance

A model becomes Champion only when it passes the configured governance criteria.

Conceptually:

```text
                 Candidate
                     │
                     ▼
              Evaluate Model
                     │
                     ▼
             Beats Baseline?
                /       \\
              NO         YES
              │           │
              ▼           ▼
           Reject       Eligible
                          │
                          ▼
                  Beats Champion?
                    /       \\
                  NO         YES
                  │           │
                  ▼           ▼
                Keep      Promote
                          Champion
```

This prevents automatic replacement of a good production model by an inferior candidate.

\---

# 15\. Next-Day Prediction

The prediction endpoint is:

```text
GET /api/prediction
```

The prediction workflow is:

```text
Latest WTI observation
          │
          ▼
Feature generation
          │
          ▼
Dedicated future feature row
          │
          ▼
Champion model
          │
          ▼
Next trading-day prediction
```

If no Champion exists, the system uses the persistence baseline and explicitly identifies it as such.

The system does **not** generate fake or synthetic market data.

\---

# 16\. Dashboard

The Dashboard is the primary operational view.

It should provide:

### Market information

* Latest WTI price
* Last observation date
* Data status

### Prediction

* Next trading-day prediction
* Prediction date
* Prediction source
* Champion status

### Model information

* Champion model
* RMSE
* MAE
* Baseline RMSE
* Improvement

### Price History

The Price History chart displays:

```text
X-axis:
Date

Y-axis:
WTI Close Price (USD/barrel)
```

The chart uses a gold/yellow price series.

Hovering over a point displays:

```text
Exact Date
Exact Price
```

\---

# 17\. FastAPI Application

File:

```text
src/serving/api.py
```

This is the application's web/API layer.

The API provides endpoints for:

```text
Dashboard
Data
Prediction
Pipeline
Benchmark
Models
AWS S3
Kubernetes
MLflow
```

Important endpoints include:

```text
GET  /
GET  /api/status
GET  /api/price-history
GET  /api/prediction

POST /api/data/refresh

POST /api/pipeline/run
GET  /api/pipeline/status

POST /api/benchmark
GET  /api/models

GET  /api/cloud

GET  /api/s3/objects
POST /api/s3/upload-champion

GET  /api/kubernetes/resources
POST /api/kubernetes/apply

GET  /api/mlflow
GET  /api/mlflow/runs
POST /api/mlflow/start
```

\---

# 18\. AWS S3 Integration

AWS functionality is implemented in:

```text
src/infrastructure/cloud.py
```

The S3 integration uses `boto3`.

It is not a simulated interface.

The application can:

```text
Test bucket access
       ↓
List objects
       ↓
Upload Champion artifacts
```

Champion artifacts:

```text
data/champion/champion\_model.joblib
data/champion/champion\_manifest.json
```

can be uploaded to:

```text
S3 bucket
   │
   └── wti-predictor/
          └── champion/
```

\---

# 19\. AWS Configuration

Create:

```text
.env
```

from:

```text
.env.example
```

Example:

```env
S3\_BUCKET=my-wti-ml-bucket
S3\_PREFIX=wti-predictor
AWS\_REGION=eu-central-1
```

AWS credentials should be configured using the normal AWS credential mechanisms.

Verify:

```bash
aws sts get-caller-identity
```

Then start the application.

\---

# 20\. AWS S3 Workflow

Recommended workflow:

```text
1. Configure AWS credentials
            ↓
2. Configure S3\_BUCKET
            ↓
3. Start application
            ↓
4. Open AWS S3
            ↓
5. Test Connection
            ↓
6. List Objects
            ↓
7. Train/Promote Champion
            ↓
8. Upload Champion
```

\---

# 21\. Kubernetes / EKS Integration

Kubernetes functionality is also implemented through:

```text
src/infrastructure/cloud.py
```

The application communicates with Kubernetes using the local:

```text
kubectl
```

executable.

The application can inspect:

* current Kubernetes context
* namespace
* nodes
* pods
* deployments

\---

# 22\. Kubernetes Prerequisites

Verify:

```bash
which kubectl
```

Then:

```bash
kubectl config current-context
```

Then:

```bash
kubectl get nodes
```

If these commands work in the terminal, the application can use the same Kubernetes configuration.

\---

# 23\. EKS Workflow

Conceptually:

```text
Local Application
       │
       ▼
     kubectl
       │
       ▼
Kubernetes API
       │
       ▼
Amazon EKS Cluster
       │
       ├── Nodes
       ├── Pods
       └── Deployments
```

The UI provides read-only cluster inspection by default.

Deployment mutation is intentionally protected.

\---

# 24\. Kubernetes Deployment Safety

Deployment operations are disabled by default.

Enable them only when intentionally required:

```bash
export DEPLOYMENT\_CONTROL\_ENABLED=true
```

Then the deployment operation requires explicit confirmation.

This prevents accidental deployment to a production cluster.

\---

# 25\. Kubernetes Manifest

The deployment template is located at:

```text
k8s/deployment.yaml
```

Before using it for a real deployment, update the container image:

```yaml
image: YOUR\_REGISTRY/YOUR\_IMAGE:TAG
```

For example:

```text
Docker
  ↓
Container Registry
  ↓
EKS
  ↓
Deployment
  ↓
Pod
```

\---

# 26\. MLflow

MLflow provides experiment tracking.

It records information such as:

* model
* hyperparameters
* RMSE
* MAE
* baseline RMSE
* improvement
* model artifacts

MLflow is not required for the core prediction calculation.

\---

# 27\. Starting MLflow

Start the local server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

The configured tracking URI is:

```text
http://127.0.0.1:5000
```

unless changed through:

```env
MLFLOW\_TRACKING\_URI
```

\---

# 28\. Docker

The project contains:

```text
Dockerfile
```

Build:

```bash
docker build -t wti-predictor:6.1.0 .
```

Run:

```bash
docker run --rm -p 8000:8000 wti-predictor:6.1.0
```

Open:

```text
http://127.0.0.1:8000
```

\---

# 29\. Installation

Navigate to the project:

```bash
cd  WTI-Oil-Price-Predictor-Pro\_v6.1
```

Activate the existing environment if necessary.

Then:

```bash
python -m pip install -r requirements.txt
```

Verify:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

\---

# 30\. Static Validation

Run:

```bash
python -m compileall -q .
```

If there is no output, Python successfully compiled the source files.

\---

# 31\. Test Suite

Run:

```bash
python -m pytest -q
```

The exact number of tests may change as the project evolves.

The important requirement is:

```text
0 failed
```

\---

# 32\. Starting the Application

The simplest method is:

```bash
python run.py --only serve
```

The server listens on:

```text
http://127.0.0.1:8000
```

Open the address in your browser.

Stop the application with:

```text
CTRL + C
```

\---

# 33\. Recommended First-Run Procedure

Do not immediately start model training.

Use the following sequence:

```text
                 START
                   │
                   ▼
              Start Server
                   │
                   ▼
               Dashboard
                   │
                   ▼
             Data Quality
                   │
                   ▼
          Refresh Data if needed
                   │
                   ▼
        Model Benchmark Center
                   │
                   ▼
        Select Multiple Models
                   │
                   ▼
          Run Benchmark
                   │
                   ▼
       Inspect RMSE / MAE
                   │
                   ▼
        Enable Fine Tuning
                   │
                   ▼
       Re-run best candidates
                   │
                   ▼
        Auto Promote Best
                   │
                   ▼
              Dashboard
                   │
                   ▼
        Next-Day Prediction
                   │
                   ▼
       Optional MLflow Tracking
                   │
                   ▼
          Optional S3 Upload
                   │
                   ▼
          Optional EKS Deploy
```

\---

# 34\. CLI Commands

The main command-line interface is:

```text
run.py
```

## Start Web Application

```bash
python run.py --only serve
```

\---

## Run Full Pipeline

```bash
python run.py --only run --model XGBoost
```

\---

## Run Full Pipeline with Fine Tuning

```bash
python run.py --only run --model XGBoost --fine-tuning
```

\---

## Benchmark All Models

```bash
python run.py --only benchmark
```

\---

## Benchmark with Fine Tuning

```bash
python run.py --only benchmark --fine-tuning
```

\---

## Disable Automatic Promotion

```bash
python run.py --only benchmark --no-auto-promote
```

\---

## Generate Prediction

```bash
python run.py --only prediction
```

\---

## Force Data Refresh

```bash
python run.py --only refresh
```

\---

# 35\. Model Selection Strategy

A professional workflow is:

### Stage 1

Benchmark:

```text
XGBoost
Random Forest
Extra Trees
HistGradientBoosting
```

without tuning.

### Stage 2

Select the strongest candidates.

### Stage 3

Fine tune those candidates.

### Stage 4

Compare them against:

```text
Persistence Baseline
```

### Stage 5

Promote only the best eligible candidate.

\---

# 36\. Understanding RMSE

RMSE means:

**Root Mean Squared Error**

It measures the typical magnitude of prediction error.

Lower is better.

For example:

```text
Model A = RMSE 2.1
Model B = RMSE 3.5
```

Model A is better according to RMSE.

However:

```text
Persistence = 1.8
Model A = 2.1
```

means Model A should not replace persistence.

\---

# 37\. Understanding MAE

MAE means:

**Mean Absolute Error**

For example:

```text
Actual = 75
Predicted = 73
Error = 2
```

MAE calculates the average absolute prediction error.

Lower is better.

\---

# 38\. Improvement Percentage

The system can compare the model against the baseline.

Conceptually:

```text
Improvement %
=
(Baseline RMSE - Model RMSE)
/
Baseline RMSE
× 100
```

Positive improvement:

```text
Model is better than baseline
```

Negative improvement:

```text
Model is worse than baseline
```

\---

# 39\. Model Manifest

Each trained model can have a manifest containing information such as:

```text
model\_id
model
fine\_tuning
feature\_count
training\_rows
RMSE
MAE
baseline RMSE
improvement %
prediction
prediction\_date
last\_price
updated\_at
governance status
```

This makes model artifacts auditable.

\---

# 40\. Troubleshooting

## Dashboard contains no data

Check:

```bash
curl http://127.0.0.1:8000/api/status
```

Then:

```bash
curl http://127.0.0.1:8000/api/price-history
```

Then:

```bash
curl http://127.0.0.1:8000/api/prediction
```

If `price-history` fails, investigate data ingestion/cache first.

\---

## S3 Upload returns 404

Check whether a Champion exists:

```text
data/champion/
```

You should have:

```text
champion\_model.joblib
champion\_manifest.json
```

If not, first run:

```text
Benchmark
```

or:

```text
Full Pipeline
```

with automatic promotion.

A `404` after that should be investigated as an API/configuration issue.

\---

## Kubernetes is unavailable

Run:

```bash
which kubectl
```

Then:

```bash
kubectl config current-context
```

Then:

```bash
kubectl get nodes
```

If these fail in the terminal, the application cannot communicate with the cluster.

\---

## MLflow is unavailable

Start:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

\---

## Port 8000 already in use

Find the process:

```bash
ss -ltnp | grep :8000
```

Stop the previous server or use another port.

\---

# 41\. Testing Philosophy

The project should maintain tests for:

```text
Data ingestion
Feature engineering
Model factory
Model evaluation
Pipeline orchestration
API endpoints
S3 integration
Kubernetes integration
Prediction
Governance
```

The goal is to prevent UI changes from silently breaking the ML pipeline.

\---

# 42\. Production Architecture

A future production architecture can be:

```text
                    Internet / User
                           │
                           ▼
                    Load Balancer
                           │
                           ▼
                    Kubernetes Ingress
                           │
                           ▼
                    FastAPI Service
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Model        MLflow        Metrics
           Service      Tracking      Monitoring
              │
              ▼
          Champion Model
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
      S3            EKS
       │
       ▼
 Model Artifacts
```

\---

# 43\. Recommended Production Enhancements

Future versions can add:

* MLflow Model Registry
* S3 versioning
* S3 encryption
* IAM least-privilege policies
* EKS IAM Roles for Service Accounts
* Kubernetes Secrets
* AWS Secrets Manager
* Prometheus
* Grafana
* OpenTelemetry
* model drift monitoring
* PSI monitoring
* automatic scheduled retraining
* model rollback
* CI/CD
* GitHub Actions
* Docker image registry
* Kubernetes Horizontal Pod Autoscaler
* TLS/HTTPS
* production database
* alerting

\---

# 44\. Security Principles

Never commit:

```text
AWS\_ACCESS\_KEY\_ID
AWS\_SECRET\_ACCESS\_KEY
passwords
tokens
private keys
production credentials
```

to Git.

Use:

```text
.env
```

for local development and proper secret-management systems for production.

The `.gitignore` file should prevent sensitive local configuration from being committed.

\---

# 45\. Environment Variables

Important variables include:

```env
WTI\_SYMBOL=CL=F

WTI\_WINDOW\_DAYS=730

TEST\_FOLDS=5

MLFLOW\_TRACKING\_URI=http://127.0.0.1:5000

S3\_BUCKET=

S3\_PREFIX=wti-predictor

AWS\_REGION=eu-central-1

DEPLOYMENT\_CONTROL\_ENABLED=false

CACHE\_MAX\_AGE\_HOURS=24
```

\---

# 46\. Operational Modes

The project can be thought of as having three operational levels.

## Level 1 — Local ML

```text
Yahoo Finance
     ↓
Python
     ↓
Model
     ↓
Prediction
```

No AWS or Kubernetes required.

\---

## Level 2 — MLOps

```text
Local ML
   ↓
MLflow
   ↓
Experiment Tracking
   ↓
Model Governance
```

\---

## Level 3 — Cloud Production

```text
MLflow
  +
S3
  +
Docker
  +
Kubernetes/EKS
  +
Monitoring
```

This separation allows the project to remain usable on a laptop while retaining a production-oriented architecture.

\---

# 47\. Complete Operational Checklist

Before considering the system operational:

```text
\[ ] requirements installed
\[ ] pip check successful
\[ ] compileall successful
\[ ] pytest successful

\[ ] Yahoo Finance accessible
\[ ] data cache working
\[ ] Data Quality passed

\[ ] benchmark completed
\[ ] multiple models evaluated
\[ ] persistence baseline calculated

\[ ] Fine Tuning tested
\[ ] Champion selected
\[ ] Champion manifest created

\[ ] Dashboard displays data
\[ ] next-day prediction available
\[ ] Price History working

\[ ] MLflow tested

\[ ] AWS credentials configured
\[ ] S3 bucket tested
\[ ] Champion upload tested

\[ ] kubectl configured
\[ ] Kubernetes context verified
\[ ] EKS nodes visible

\[ ] Docker image tested

\[ ] Kubernetes deployment tested
```

\---

# 48\. Quick Start

For the fastest local setup:

```bash
cd  WTI-Oil-Price-Predictor-Pro\_v6.1

python -m pip install -r requirements.txt

python -m pip check

python -m compileall -q .

python -m pytest -q

python run.py --only serve
```

Open:

```text
http://127.0.0.1:8000
```

Then:

```text
Dashboard
   ↓
Data Quality
   ↓
Model Benchmark Center
   ↓
Select Models
   ↓
Run Benchmark
   ↓
Fine Tune
   ↓
Auto Promote
   ↓
Dashboard
   ↓
Next-Day Prediction
```

\---

# 49\. Project Philosophy

The project follows five important principles:

### 1\. No fake data

If market data are unavailable, the system reports the real problem.

### 2\. No automatic blind model promotion

A model must demonstrate predictive value.

### 3\. Time-series aware evaluation

The temporal ordering of financial data must be respected.

### 4\. Cloud functionality must be real

AWS S3 and Kubernetes interfaces are designed to communicate with the actual services rather than merely displaying simulated information.

### 5\. Local-first architecture

The system should remain usable without AWS, EKS, or MLflow.

\---

# 50\. Summary

WTI Oil Price Predictor Pro v6.1 is structured as an end-to-end ML/MLOps platform:

```text
                ┌────────────────────┐
                │   Yahoo Finance    │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Data Ingestion      │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Data Quality        │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Feature Engineering│
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Benchmark Center    │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Fine Tuning         │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Model Evaluation    │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Model Governance    │
                └─────────┬──────────┘
                          ▼
                     ┌────┴────┐
                     │         │
                     ▼         ▼
                  Champion   Reject
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       MLflow       S3       Kubernetes/EKS
          │          │           │
          └──────────┼───────────┘
                     ▼
             Next-Day Prediction
                     │
                     ▼
                 Dashboard
```

The ultimate objective is not merely to train an XGBoost model.

The objective is to build a complete, auditable and extensible **ML production pipeline** capable of:

```text
Acquire
   ↓
Validate
   ↓
Engineer
   ↓
Benchmark
   ↓
Tune
   ↓
Evaluate
   ↓
Govern
   ↓
Track
   ↓
Store
   ↓
Deploy
   ↓
Predict
   ↓
Monitor
```

\---




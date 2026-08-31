from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; RAW=DATA/'raw'; MODELS=DATA/'models'; CHAMPION=DATA/'champion'; ARTIFACTS=ROOT/'artifacts'; LOGS=ROOT/'logs'
for p in (RAW,MODELS,CHAMPION,ARTIFACTS,LOGS): p.mkdir(parents=True,exist_ok=True)
SYMBOL=os.getenv('WTI_SYMBOL','CL=F'); WINDOW_DAYS=int(os.getenv('WTI_WINDOW_DAYS','730')); TEST_FOLDS=int(os.getenv('TEST_FOLDS','5'))
MODEL_NAMES=['XGBoost','Random Forest','Extra Trees','HistGradientBoosting']; APP_VERSION='6.1.0'
MLFLOW_URI=os.getenv('MLFLOW_TRACKING_URI','http://127.0.0.1:5000'); MLFLOW_EXPERIMENT='wti-oil-price-prediction'
S3_BUCKET=os.getenv('S3_BUCKET',''); S3_PREFIX=os.getenv('S3_PREFIX','wti-predictor'); AWS_REGION=os.getenv('AWS_REGION','eu-central-1')
DEPLOYMENT_CONTROL_ENABLED=os.getenv('DEPLOYMENT_CONTROL_ENABLED','false').lower()=='true'; CACHE_MAX_AGE_HOURS=int(os.getenv('CACHE_MAX_AGE_HOURS','24'))

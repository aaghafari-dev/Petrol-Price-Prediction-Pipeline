"""
Orchestrates the full ML pipeline: data, training, governance.
"""
import json
import joblib
import uuid
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.project_config import CHAMPION, MODELS, ARTIFACTS, TEST_FOLDS, MLFLOW_URI, MLFLOW_EXPERIMENT
from src.data.data_ingestion import load_data
from src.data.features import make_supervised, make_next_features
from src.models.evaluator import walk_forward, fit_final


class Pipeline:
    def __init__(self, status=None):
        self.status = status or (lambda p, s, m: None)
    
    def _data(self):
        self.status(8, 'data', 'Loading WTI market data (cached when possible)')
        df = load_data()
        self.status(20, 'features', 'Building leakage-safe features')
        X, y, frame = make_supervised(df)
        return df, X, y
    
    def _mlflow(self, metrics, params, artifact):
        if os.getenv('MLFLOW_TRACKING_URI', MLFLOW_URI).strip().lower() in ('', 'none', 'disabled'):
            return {'logged': False, 'reason': 'disabled'}
        
        try:
            import mlflow
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_URI))
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            
            with mlflow.start_run(run_name=f"{metrics['model']}-{datetime.utcnow():%Y%m%d-%H%M%S}"):
                mlflow.log_params({
                    'model': metrics['model'],
                    'fine_tuning': artifact['fine_tuning'],
                    **{k: str(v) for k, v in params.items()}
                })
                mlflow.log_metrics({
                    k: float(metrics[k])
                    for k in ('rmse', 'mae', 'baseline_rmse', 'improvement_pct')
                })
                mlflow.log_text(json.dumps(artifact, indent=2), 'model_manifest.json')
            
            return {'logged': True, 'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', MLFLOW_URI)}
        except Exception as e:
            return {'logged': False, 'reason': str(e)}
    
    def _save_and_maybe_promote(self, name, tune, metrics, df, X, y, promote=True, best_params=None):
        self.status(68, 'training', f'Training final {name} deployment model')
        model = fit_final(name, X, y, tune, best_params)
        
        model_id = f'{name.lower().replace(" ", "_")}_{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
        path = MODELS / f'{model_id}.joblib'
        joblib.dump(model, path)
        
        # Make next-day prediction
        last_price = float(df.price.iloc[-1])
        next_X, pred_date = make_next_features(df)
        pred = float(model.predict(next_X)[0])
        
        artifact = {
            'model_id': model_id,
            'model': name,
            'fine_tuning': bool(tune),
            'feature_count': len(X.columns),
            'training_rows': len(X),
            'metrics': metrics,
            'prediction': pred,
            'prediction_date': str(pred_date.date()),
            'last_price': last_price,
            'updated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        }
        
        # Governance check
        current = None
        cf = CHAMPION / 'champion_manifest.json'
        if cf.exists():
            try:
                current = json.loads(cf.read_text())
            except Exception:
                current = None
        
        eligible = bool(metrics['beats_baseline']) and (
            current is None or metrics['rmse'] < current.get('metrics', {}).get('rmse', float('inf'))
        )
        artifact['governance'] = (
            'PROMOTED' if (promote and eligible)
            else ('ELIGIBLE_NOT_PROMOTED' if eligible else 'REJECTED')
        )
        
        if promote and eligible:
            joblib.dump(model, CHAMPION / 'champion_model.joblib')
            cf.write_text(json.dumps(artifact, indent=2))
            self.status(88, 'governance', 'New Champion promoted')
        
        # MLflow tracking
        ml = self._mlflow(metrics, best_params or {}, artifact)
        artifact['mlflow'] = ml
        
        # Save artifacts
        (ARTIFACTS / f'{model_id}.json').write_text(json.dumps(artifact, indent=2))
        (MODELS / f'{model_id}.json').write_text(json.dumps(artifact, indent=2))
        
        self.status(96, 'complete', artifact['governance'])
        return artifact
    
    def run(self, name, tune=False):
        df, X, y = self._data()
        self.status(38, 'evaluation', f'Walk-forward evaluation: {name}')
        metrics = walk_forward(name, X, y, tune, TEST_FOLDS)
        return self._save_and_maybe_promote(name, tune, metrics, df, X, y, True)
    
    def benchmark(self, models, tune=False, auto_promote=True):
        df, X, y = self._data()
        results = []
        
        for i, name in enumerate(models):
            self.status(25 + int(i * 35 / max(1, len(models))), 'benchmark', f'Evaluating {name}')
            try:
                result = walk_forward(name, X, y, tune, TEST_FOLDS)
                results.append({'status': 'SUCCESS', **result})
            except Exception as e:
                results.append({'status': 'FAILED', 'model': name, 'error': str(e)})
        
        good = [r for r in results if r['status'] == 'SUCCESS']
        good.sort(key=lambda r: r['rmse'])
        best = good[0] if good else None
        promotion = None
        
        if auto_promote and best and best['beats_baseline']:
            # Extract best params from folds (use last fold's params)
            best_params = None
            if best['folds'] and best['folds'][-1]['best_params']:
                best_params = best['folds'][-1]['best_params']
            promotion = self._save_and_maybe_promote(
                best['model'], tune, best, df, X, y, True, best_params
            )
        
        return {
            'results': good + [r for r in results if r['status'] == 'FAILED'],
            'best': best,
            'promotion': promotion,
            'rows': len(X),
            'features': len(X.columns)
        }
    
    def prediction(self):
        df = load_data()
        cf = CHAMPION / 'champion_manifest.json'
        mf = CHAMPION / 'champion_model.joblib'
        pred_date = str((df.Date.iloc[-1] + pd.offsets.BDay(1)).date())
        
        if cf.exists() and mf.exists():
            m = json.loads(cf.read_text())
            model = joblib.load(mf)
            next_X, pdte = make_next_features(df)
            pred = float(model.predict(next_X)[0])
            pred_date = str(pdte.date())
            return {
                'prediction': pred,
                'prediction_date': pred_date,
                'source': 'Champion',
                'model_id': m.get('model_id'),
                'last_price': float(df.price.iloc[-1])
            }
        
        # Fallback to persistence baseline
        return {
            'prediction': float(df.price.iloc[-1]),
            'prediction_date': pred_date,
            'source': 'Persistence baseline',
            'model_id': None,
            'last_price': float(df.price.iloc[-1])
        }
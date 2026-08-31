"""
Time-series evaluation with walk-forward validation.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.models.factory import make_model

SPACES = {
    'XGBoost': {
        'n_estimators': [200, 400, 700],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.02, 0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'Random Forest': {
        'n_estimators': [200, 400, 700],
        'max_depth': [6, 10, 15, None],
        'min_samples_leaf': [1, 2, 4]
    },
    'Extra Trees': {
        'n_estimators': [200, 400, 700],
        'max_depth': [6, 10, 15, None],
        'min_samples_leaf': [1, 2, 4]
    },
    'HistGradientBoosting': {
        'max_iter': [150, 300, 500],
        'learning_rate': [0.02, 0.05, 0.1],
        'max_leaf_nodes': [15, 31, 63],
        'l2_regularization': [0, 0.1, 1]
    }
}


def _fit(name, Xtr, ytr, tune=False, best_params=None, seed=42):
    """Fit model, optionally with fine-tuning."""
    if best_params:
        model = make_model(name, best_params)
        model.fit(Xtr, ytr)
        return model, best_params
    
    model = make_model(name)
    if not tune:
        model.fit(Xtr, ytr)
        return model, {}
    
    # Deterministic RandomizedSearchCV with TimeSeriesSplit
    inner_splits = max(2, min(3, len(Xtr) // 40))
    search = RandomizedSearchCV(
        model,
        SPACES[name],
        n_iter=6,
        cv=TimeSeriesSplit(n_splits=inner_splits),
        scoring='neg_root_mean_squared_error',
        random_state=seed,
        n_jobs=1,
        error_score='raise'
    )
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


def walk_forward(name, X, y, tune=False, n_splits=5, progress=None):
    """Walk-forward validation to estimate out-of-sample performance."""
    if len(X) < n_splits + 2:
        raise RuntimeError('Not enough rows for time-series evaluation')
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    predictions = []
    actuals = []
    baselines = []
    folds = []
    
    for i, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        model, params = _fit(name, X.iloc[tr_idx], y[tr_idx], tune, None, 42 + i)
        pred = model.predict(X.iloc[te_idx])
        
        predictions.extend(pred.tolist())
        actuals.extend(y[te_idx].tolist())
        # Baseline: yesterday's price (lag_1)
        baselines.extend(X.iloc[te_idx]['lag_1'].to_numpy().tolist())
        
        folds.append({
            'fold': i,
            'train_rows': len(tr_idx),
            'test_rows': len(te_idx),
            'best_params': params
        })
        
        if progress:
            progress(i, n_splits)
    
    rmse = float(mean_squared_error(actuals, predictions) ** 0.5)
    mae = float(mean_absolute_error(actuals, predictions))
    baseline_rmse = float(mean_squared_error(actuals, baselines) ** 0.5)
    
    return {
        'model': name,
        'rmse': rmse,
        'mae': mae,
        'baseline_rmse': baseline_rmse,
        'improvement_pct': float((baseline_rmse - rmse) / baseline_rmse * 100),
        'beats_baseline': bool(rmse < baseline_rmse),
        'folds': folds
    }


def fit_final(name, X, y, tune=False, best_params=None):
    """Train final model on all data."""
    model, params = _fit(name, X, y, tune, best_params)
    return model
"""Factory for creating ML models with deterministic configuration."""
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

def make_model(name: str, params: dict = None):
    """Create and return a configured model instance."""
    p = params or {}
    if name == 'XGBoost':
        return XGBRegressor(
            n_estimators=p.get('n_estimators', 400),
            max_depth=p.get('max_depth', 5),
            learning_rate=p.get('learning_rate', 0.03),
            subsample=p.get('subsample', 0.9),
            colsample_bytree=p.get('colsample_bytree', 0.9),
            objective='reg:squarederror',
            random_state=42,
            n_jobs=2,
            tree_method='hist'
        )
    if name == 'Random Forest':
        return RandomForestRegressor(
            n_estimators=p.get('n_estimators', 300),
            max_depth=p.get('max_depth', 12),
            min_samples_leaf=p.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=2
        )
    if name == 'Extra Trees':
        return ExtraTreesRegressor(
            n_estimators=p.get('n_estimators', 300),
            max_depth=p.get('max_depth', 12),
            min_samples_leaf=p.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=2
        )
    if name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(
            max_iter=p.get('max_iter', 300),
            learning_rate=p.get('learning_rate', 0.05),
            max_leaf_nodes=p.get('max_leaf_nodes', 31),
            l2_regularization=p.get('l2_regularization', 0.1),
            random_state=42
        )
    raise ValueError(f'Unsupported model: {name}')
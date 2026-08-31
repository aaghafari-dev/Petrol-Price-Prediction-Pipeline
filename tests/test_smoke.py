from fastapi.testclient import TestClient
from src.serving.api import app
from src.data.features import make_next_features
import pandas as pd

def test_home_and_routes():
    c=TestClient(app)
    assert c.get('/').status_code==200
    assert c.get('/api/status').status_code==200
    assert c.post('/api/s3/upload-champion').status_code in (400,404)

def test_next_feature_uses_latest_price():
    dates=pd.date_range('2025-01-01',periods=30,freq='D')
    df=pd.DataFrame({'Date':dates,'price':range(100,130)})
    x,d=make_next_features(df)
    assert float(x.iloc[0]['lag_1'])==129.0
    assert d.date()==pd.Timestamp('2025-01-31').date()

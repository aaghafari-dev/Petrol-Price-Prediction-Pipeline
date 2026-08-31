import pandas as pd
FEATURES=['lag_1','lag_2','lag_3','lag_5','lag_10','ma_5','ma_10','ma_20','std_5','std_10','return_1','return_5','dow','month']
def _feature_frame(df):
    x=df.copy(); p=x.price
    x['lag_1']=p.shift(1); x['lag_2']=p.shift(2); x['lag_3']=p.shift(3); x['lag_5']=p.shift(5); x['lag_10']=p.shift(10)
    x['ma_5']=p.shift(1).rolling(5).mean(); x['ma_10']=p.shift(1).rolling(10).mean(); x['ma_20']=p.shift(1).rolling(20).mean()
    x['std_5']=p.shift(1).rolling(5).std(); x['std_10']=p.shift(1).rolling(10).std()
    x['return_1']=p.shift(1).pct_change(); x['return_5']=p.shift(1).pct_change(5)
    x['dow']=x.Date.dt.dayofweek; x['month']=x.Date.dt.month
    return x
def make_supervised(df):
    x=_feature_frame(df); x['target']=x.price.shift(-1); x=x.dropna(subset=FEATURES+['target']).reset_index(drop=True)
    return x[FEATURES],x.target.to_numpy(),x
def make_next_features(df):
    if len(df)<21: raise RuntimeError('At least 21 observations are required for next-day prediction')
    p=df.price.astype(float); d=df.Date.iloc[-1]; nd=d+pd.offsets.BDay(1)
    row={'lag_1':p.iloc[-1],'lag_2':p.iloc[-2],'lag_3':p.iloc[-3],'lag_5':p.iloc[-5],'lag_10':p.iloc[-10],
         'ma_5':p.iloc[-5:].mean(),'ma_10':p.iloc[-10:].mean(),'ma_20':p.iloc[-20:].mean(),
         'std_5':p.iloc[-5:].std(),'std_10':p.iloc[-10:].std(),'return_1':p.iloc[-1]/p.iloc[-2]-1,
         'return_5':p.iloc[-1]/p.iloc[-6]-1,'dow':nd.dayofweek,'month':nd.month}
    return pd.DataFrame([row],columns=FEATURES),nd

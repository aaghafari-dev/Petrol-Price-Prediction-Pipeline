import argparse, uvicorn, os
from dotenv import load_dotenv

# 
load_dotenv()

from config.project_config import APP_VERSION, MODEL_NAMES
from src.pipeline.orchestrator import Pipeline

p = argparse.ArgumentParser(description=f'WTI Oil Price Predictor Pro {APP_VERSION}')
p.add_argument('--only', default='serve', choices=['serve', 'run', 'benchmark', 'prediction', 'refresh'])
p.add_argument('--model', default='XGBoost', choices=MODEL_NAMES)
p.add_argument('--fine-tuning', action='store_true')
p.add_argument('--no-auto-promote', action='store_true')
a = p.parse_args()

if a.only == 'serve':
    uvicorn.run('src.serving.api:app', host='0.0.0.0', port=8000, reload=False)
elif a.only == 'run':
    print(Pipeline().run(a.model, a.fine_tuning))
elif a.only == 'benchmark':
    print(Pipeline().benchmark(MODEL_NAMES, a.fine_tuning, not a.no_auto_promote))
elif a.only == 'prediction':
    print(Pipeline().prediction())
else:
    from src.data.data_ingestion import load_data
    print(load_data(force=True).tail())
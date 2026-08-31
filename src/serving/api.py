import time 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import threading, json, os, subprocess
from config.project_config import APP_VERSION, CHAMPION, MODELS, MLFLOW_URI, MODEL_NAMES, ROOT
from src.pipeline.orchestrator import Pipeline
from src.data.data_ingestion import load_data, SNAPSHOT
from src.infrastructure.cloud import s3_status, s3_objects, upload, k8s_status, k8s_resources, apply_k8s

app = FastAPI(title='WTI Oil Price Predictor Pro', version=APP_VERSION)

state = {
    'running': False,
    'progress': 0,
    'stage': 'idle',
    'message': 'Ready',
    'result': None,
    'error': None
}
lock = threading.Lock()

class RunReq(BaseModel):
    model: str = Field('XGBoost')
    fine_tuning: bool = False

class BenchReq(BaseModel):
    models: list[str]
    fine_tuning: bool = False
    auto_promote: bool = True

def set_status(p, s, m):
    with lock:
        state.update(progress=int(p), stage=s, message=m)

def _task(fn):
    try:
        r = fn()
        with lock:
            state.update(running=False, progress=100, stage='complete', message='Completed successfully', result=r, error=None)
    except Exception as e:
        with lock:
            state.update(running=False, progress=100, stage='error', message=str(e), result=None, error=str(e))

def _start(fn):
    with lock:
        if state['running']:
            raise HTTPException(409, 'A job is already running')
        state.update(running=True, progress=1, stage='queued', message='Job queued', result=None, error=None)
    threading.Thread(target=_task, args=(fn,), daemon=True).start()
    return {'started': True}

def _champ():
    p = CHAMPION / 'champion_manifest.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except:
        return None

@app.get('/', response_class=HTMLResponse)
def home():
    return HTML

@app.get('/api/status')
def status():
    return {
        'version': APP_VERSION,
        'pipeline': dict(state),
        'champion': _champ(),
        'snapshot_exists': SNAPSHOT.exists()
    }

@app.get('/api/price-history')
def history():
    try:
        df = load_data()
        return {
            'dates': [x.strftime('%Y-%m-%d') for x in df.Date.tail(250)],
            'prices': [float(x) for x in df.price.tail(250)],
            'last_date': df.Date.iloc[-1].strftime('%Y-%m-%d'),
            'min': float(df.price.tail(250).min()),
            'max': float(df.price.tail(250).max()),
            'avg': float(df.price.tail(250).mean())
        }
    except Exception as e:
        raise HTTPException(502, f'Unable to load price history: {str(e)}')

@app.post('/api/data/refresh')
def refresh():
    try:
        df = load_data(force=True)
        return {'ok': True, 'rows': len(df), 'last_date': df.Date.iloc[-1].strftime('%Y-%m-%d')}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.get('/api/prediction')
def prediction():
    try:
        return Pipeline().prediction()
    except Exception as e:
        raise HTTPException(500, f'Prediction failed: {str(e)}')

@app.get('/api/data-quality')
def quality():
    try:
        df = load_data()
        return {
            'rows': len(df),
            'columns': list(df.columns),
            'missing': {k: int(v) for k, v in df.isna().sum().items()},
            'start': df.Date.min().strftime('%Y-%m-%d'),
            'end': df.Date.max().strftime('%Y-%m-%d'),
            'min': float(df.price.min()),
            'max': float(df.price.max()),
            'mean': float(df.price.mean()),
            'std': float(df.price.std()),
            'duplicates': int(df.Date.duplicated().sum()),
            'last_date': df.Date.iloc[-1].strftime('%Y-%m-%d'),
            'last_price': float(df.price.iloc[-1]),
            'data_source': 'Yahoo Finance' if SNAPSHOT.exists() else 'Unknown',
            'cache_age_hours': round((time.time() - SNAPSHOT.stat().st_mtime) / 3600, 2) if SNAPSHOT.exists() else None
        }
    except Exception as e:
        raise HTTPException(500, f'Quality check failed: {str(e)}')
# @app.get('/api/data-quality')
# def quality():
#     try:
#         df = load_data()
#         return {
#             'rows': len(df),
#             'columns': list(df.columns),
#             'missing': {k: int(v) for k, v in df.isna().sum().items()},
#             'start': df.Date.min().strftime('%Y-%m-%d'),
#             'end': df.Date.max().strftime('%Y-%m-%d'),
#             'min': float(df.price.min()),
#             'max': float(df.price.max()),
#             'mean': float(df.price.mean()),
#             'std': float(df.price.std()),
#             'duplicates': int(df.Date.duplicated().sum())
#         }
#     except Exception as e:
#         raise HTTPException(500, f'Quality check failed: {str(e)}')

@app.post('/api/pipeline/run')
def run(req: RunReq):
    if req.model not in MODEL_NAMES:
        raise HTTPException(400, f'Unsupported model: {req.model}')
    return _start(lambda: Pipeline(set_status).run(req.model, req.fine_tuning))

@app.get('/api/pipeline/status')
def pstatus():
    return dict(state)

@app.post('/api/benchmark')
def benchmark(req: BenchReq):
    models = [m for m in req.models if m in MODEL_NAMES]
    if not models:
        raise HTTPException(400, 'Select at least one supported model')
    return _start(lambda: Pipeline(set_status).benchmark(models, req.fine_tuning, req.auto_promote))

@app.get('/api/models')
def models():
    out = []
    for p in MODELS.glob('*.json'):
        try:
            out.append(json.loads(p.read_text()))
        except:
            pass
    out.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return {'models': out[:50]}

@app.get('/api/cloud')
def cloud():
    return {'s3': s3_status(), 'kubernetes': k8s_status()}

@app.get('/api/s3/objects')
def objects():
    return s3_objects()

@app.post('/api/s3/upload-champion')
def upload_champion():
    mf = CHAMPION / 'champion_model.joblib'
    jf = CHAMPION / 'champion_manifest.json'
    if not (mf.exists() and jf.exists()):
        raise HTTPException(404, 'No Champion artifacts exist. Run Benchmark or Pipeline first.')
    try:
        return {
            'model': upload(mf, 'champion/champion_model.joblib'),
            'manifest': upload(jf, 'champion/champion_manifest.json')
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get('/api/kubernetes/resources')
def resources(kind='pods', namespace=''):
    return k8s_resources(kind, namespace)

@app.post('/api/kubernetes/apply')
def kapply(confirm: str):
    if confirm != 'DEPLOY':
        raise HTTPException(400, 'Type DEPLOY to confirm')
    try:
        return apply_k8s(True)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get('/api/mlflow/runs')
def mlflow_runs():
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_URI))
        exp = mlflow.get_experiment_by_name('wti-oil-price-prediction')
        if not exp:
            return {'runs': []}
        rows = mlflow.search_runs([exp.experiment_id], order_by=['start_time DESC'], max_results=30)
        keep = [c for c in ('run_id', 'run_name', 'status', 'start_time', 'metrics.rmse', 'metrics.mae', 'metrics.baseline_rmse', 'metrics.improvement_pct') if c in rows.columns]
        return {'runs': rows[keep].fillna('').to_dict(orient='records')}
    except Exception as e:
        return {'runs': [], 'error': str(e)}

@app.get('/api/mlflow')
def mlflow_status():
    uri = os.getenv('MLFLOW_TRACKING_URI', MLFLOW_URI)
    try:
        import requests
        r = requests.get(uri.rstrip('/') + '/health', timeout=2)
        return {'uri': uri, 'healthy': r.ok, 'message': 'Connected' if r.ok else 'Unhealthy'}
    except Exception as e:
        return {'uri': uri, 'healthy': False, 'message': 'Not running / unreachable', 'error': str(e)}

@app.post('/api/mlflow/start')
def mlflow_start():
    try:
        import requests
        if requests.get(MLFLOW_URI.rstrip('/') + '/health', timeout=1).ok:
            return {'started': False, 'already_running': True, 'uri': MLFLOW_URI}
    except Exception:
        pass
    try:
        p = subprocess.Popen(
            ['mlflow', 'server', '--host', '127.0.0.1', '--port', '5000'],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return {'started': True, 'pid': p.pid, 'uri': MLFLOW_URI}
    except Exception as e:
        raise HTTPException(500, str(e))

# HTML with professional Plotly chart
HTML = r'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>WTI Oil Price Predictor Pro </title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg: #07111f;
            --side: #091626;
            --panel: #0e1b2d;
            --line: #20324a;
            --text: #e8eef7;
            --muted: #8ea2ba;
            --blue: #78bfff;
            --blue2: #173858;
            --gold: #e4bd4f;
            --green: #70d6a0;
            --red: #ff8c8c;
            --purple: #b388ff;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .app {
            display: flex;
            min-height: 100vh;
        }
        
        .side {
            width: 270px;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            background: linear-gradient(180deg, #091626 0%, #0d1f38 100%);
            border-right: 1px solid var(--line);
            padding: 20px 14px;
            overflow-y: auto;
            z-index: 100;
        }
        
        .brand {
            font-size: 20px;
            font-weight: 900;
            margin: 2px 10px 25px;
            color: var(--gold);
            letter-spacing: -0.5px;
            text-transform: uppercase;
            border-bottom: 2px solid var(--gold);
            padding-bottom: 10px;
        }
        
        .brand-sub {
            font-size: 11px;
            color: var(--muted);
            margin-top: -18px;
            margin-bottom: 20px;
            margin-left: 10px;
        }
        
        .nav {
            display: block;
            padding: 13px 14px;
            margin: 6px 0;
            border-radius: 10px;
            color: var(--text);
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
            border-left: 3px solid transparent;
        }
        
        .nav:hover {
            background: var(--blue2);
            color: var(--blue);
            border-left-color: var(--blue);
            transform: translateX(2px);
        }
        
        .nav.active {
            background: linear-gradient(90deg, #123b61 0%, #1a4d7a 100%);
            color: #fff;
            border-left-color: var(--gold);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        .main {
            margin-left: 270px;
            flex: 1;
            padding: 30px 40px;
            max-width: 1800px;
        }
        
        .title {
            text-align: center;
            font-size: 30px;
            margin-bottom: 30px;
            color: var(--gold);
            font-weight: 900;
            letter-spacing: -1px;
            text-shadow: 0 2px 10px rgba(228, 189, 79, 0.3);
        }
        
        .page { display: none; }
        .page.active { display: block; animation: fadeIn 0.3s ease; }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .grid2 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: linear-gradient(145deg, var(--panel) 0%, #0d1e35 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(0,0,0,0.3);
        }
        
        .card h2 {
            color: var(--blue);
            font-size: 18px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .metric {
            font-size: 32px;
            font-weight: 800;
            margin-top: 10px;
            color: var(--text);
        }
        
        .prediction {
            font-size: 52px;
            font-weight: 900;
            color: var(--gold);
            margin-top: 10px;
            text-shadow: 0 2px 15px rgba(228, 189, 79, 0.4);
            line-height: 1.2;
        }
        
        .muted {
            color: var(--muted);
            font-size: 13px;
        }
        
        .small { font-size: 12px; }
        
        button, select {
            background: linear-gradient(135deg, var(--blue2) 0%, #1e4a72 100%);
            color: #fff;
            border: 1px solid #315779;
            border-radius: 10px;
            padding: 11px 16px;
            margin: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        button:hover {
            background: linear-gradient(135deg, #1e4a72 0%, #235d8e 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(35, 93, 142, 0.4);
        }
        
        button.primary {
            background: linear-gradient(135deg, #235d8e 0%, #2a7db5 100%);
            font-weight: 600;
            font-size: 15px;
        }
        
        button.primary:hover {
            background: linear-gradient(135deg, #2a7db5 0%, #3a8dc5 100%);
        }
        
        .progress {
            height: 12px;
            background: #142338;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        .bar {
            height: 100%;
            background: linear-gradient(90deg, var(--blue) 0%, var(--gold) 100%);
            width: 0;
            transition: width 0.3s ease;
            border-radius: 8px;
        }
        
        .log, pre {
            white-space: pre-wrap;
            background: rgba(6, 16, 29, 0.8);
            padding: 15px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .ok { color: var(--green); }
        .bad { color: var(--red); }
        .warning { color: #ffd700; }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        th, td {
            padding: 12px;
            border-bottom: 1px solid var(--line);
            text-align: left;
        }
        
        th {
            background: rgba(23, 56, 88, 0.5);
            color: var(--blue);
            font-weight: 600;
        }
        
        tr:hover {
            background: rgba(23, 56, 88, 0.3);
        }
        
        .chart-container {
            background: linear-gradient(180deg, #0a1525 0%, #0d1b2e 100%);
            border-radius: 12px;
            padding: 10px;
            margin-top: 15px;
            border: 1px solid var(--line);
        }
        
        #priceChart {
            width: 100%;
            height: 450px;
        }
        
        .tooltip {
            position: absolute;
            background: #0b2a46;
            color: var(--blue);
            border: 1px solid var(--blue);
            padding: 10px 14px;
            border-radius: 8px;
            pointer-events: none;
            display: none;
            font-weight: 600;
            font-size: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        
        .flow {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin: 15px 0;
        }
        
        .node {
            padding: 12px 16px;
            border: 1px solid var(--blue);
            border-radius: 10px;
            background: linear-gradient(135deg, #102a44 0%, #123b61 100%);
            color: var(--text);
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .arrow {
            color: var(--gold);
            font-size: 22px;
            font-weight: bold;
        }
        
        .check {
            display: block;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(23, 56, 88, 0.3);
            transition: all 0.2s ease;
        }
        
        .check:hover {
            background: rgba(23, 56, 88, 0.6);
        }
        
        .check input[type="checkbox"] {
            margin-right: 10px;
            transform: scale(1.3);
        }
        
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-dot.success { background: var(--green); box-shadow: 0 0 8px var(--green); }
        .status-dot.failed { background: var(--red); box-shadow: 0 0 8px var(--red); }
        .status-dot.running { background: var(--gold); box-shadow: 0 0 8px var(--gold); animation: pulse 1s infinite; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px;
        }
        
        .badge.success { background: rgba(112, 214, 160, 0.2); color: var(--green); border: 1px solid var(--green); }
        .badge.failed { background: rgba(255, 140, 140, 0.2); color: var(--red); border: 1px solid var(--red); }
        .badge.info { background: rgba(120, 191, 255, 0.2); color: var(--blue); border: 1px solid var(--blue); }
        .badge.warning { background: rgba(255, 215, 0, 0.2); color: #ffd700; border: 1px solid #ffd700; }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(120, 191, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--blue);
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @media (max-width: 1200px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
        }
        
        @media (max-width: 900px) {
            .side { width: 220px; }
            .main { margin-left: 220px; padding: 20px; }
            .grid2 { grid-template-columns: 1fr; }
        }
        
        @media (max-width: 700px) {
            .side { position: relative; width: 100%; height: auto; }
            .app { display: block; }
            .main { margin-left: 0; }
            .grid { grid-template-columns: 1fr; }
            .title { font-size: 24px; }
        }
    </style>
</head>
<body>
    <div class="app">
        <aside class="side">
            <div class="brand">⚡ WTI Predictor</div>
            <div class="brand-sub">Pro v6.1.0</div>
            
            <div class="nav active" data-page="dashboard">📊 Dashboard</div>
            <div class="nav" data-page="pipeline">⚙️ Pipeline Control</div>
            <div class="nav" data-page="benchmark">🏆 Model Benchmark</div>
            <div class="nav" data-page="quality">🔍 Data Quality</div>
            <div class="nav" data-page="governance">📋 Governance</div>
            <div class="nav" data-page="mlflow">🔬 MLflow</div>
            <div class="nav" data-page="aws">☁️ AWS S3</div>
            <div class="nav" data-page="k8s">🚢 Kubernetes</div>
            <div class="nav" data-page="docs">📖 Explanations</div>
        </aside>
        
        <main class="main">
            <h1 class="title">WTI Oil Price Predictor </h1>
            
            <!-- Dashboard -->
            <section id="dashboard" class="page active">
                <div class="grid2">
                    <div class="card">
                        <h2>📈 Next Trading-Day Prediction</h2>
                        <div id="pred" class="prediction">Loading...</div>
                        <div id="predmeta" class="muted"></div>
                        <div style="margin-top: 15px;">
                            <span id="predSource" class="badge info">Loading...</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🏆 Champion Model</h2>
                        <div id="champ" class="metric">—</div>
                        <div id="champmeta" class="muted"></div>
                        <div style="margin-top: 15px;">
                            <span id="champStatus" class="badge info">Loading...</span>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Price History Chart</h2>
                    <div class="chart-container">
                        <div id="priceChart"></div>
                    </div>
                    <div class="muted small" style="margin-top: 10px; text-align: center;">
                        X-axis: Date | Y-axis: WTI Close Price (USD/barrel) | Source: Yahoo Finance
                    </div>
                    <div style="text-align: center; margin-top: 15px;">
                        <button onclick="refreshData()" class="primary">🔄 Refresh Market Data</button>
                    </div>
                </div>
            </section>
            
            <!-- Pipeline Control -->
            <section id="pipeline" class="page">
                <div class="card">
                    <h2>⚙️ Pipeline Control</h2>
                    
                    <div class="flow">
                        <span class="node">📥 Load Data</span>
                        <span class="arrow">→</span>
                        <span class="node">🧪 Features</span>
                        <span class="arrow">→</span>
                        <span class="node">📊 Walk-Forward</span>
                        <span class="arrow">→</span>
                        <span class="node">🏋️ Train</span>
                        <span class="arrow">→</span>
                        <span class="node">📋 Govern</span>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <label style="font-weight: 600; margin-right: 10px;">Select Model:</label>
                        <select id="model"></select>
                    </div>
                    
                    <div style="margin: 10px 0;">
                        <label class="check">
                            <input id="tune" type="checkbox">
                            🔬 Enable Fine Tuning (RandomizedSearchCV)
                        </label>
                    </div>
                    
                    <button class="primary" onclick="runPipeline()" style="font-size: 16px; padding: 14px 24px;">
                        ▶️ Run Full Pipeline
                    </button>
                    
                    <div class="progress" style="margin-top: 20px;">
                        <div id="bar" class="bar"></div>
                    </div>
                    
                    <div id="pmsg" class="muted" style="margin-top: 10px;">Idle</div>
                    
                    <pre id="plog" style="margin-top: 15px;">No run yet.</pre>
                </div>
            </section>
            
            <!-- Model Benchmark Center -->
            <section id="benchmark" class="page">
                <div class="card">
                    <h2>🏆 Model Benchmark Center</h2>
                    <p class="muted">
                        Select models for evaluation. Each model is tested with identical chronological folds.
                        Failed models are isolated and do not abort the entire benchmark.
                        The best model is automatically ranked and may be promoted to Champion if it beats the baseline.
                    </p>
                    
                    <div id="checks" style="margin: 20px 0;"></div>
                    
                    <div style="margin: 10px 0;">
                        <label class="check">
                            <input id="btune" type="checkbox">
                            🔬 Enable Fine Tuning
                        </label>
                        <label class="check">
                            <input id="autop" type="checkbox" checked>
                            🏆 Auto-promote best eligible model
                        </label>
                    </div>
                    
                    <button class="primary" onclick="runBenchmark()" style="font-size: 16px; padding: 14px 24px;">
                        🚀 Run Benchmark
                    </button>
                    
                    <div id="benchout" style="margin-top: 20px;"></div>
                </div>
            </section>
            
            <!-- Data Quality -->
            <section id="quality" class="page">
                <div class="card">
                    <h2>🔍 Data Quality Report</h2>
                    <button onclick="loadQuality()">🔄 Refresh Quality Report</button>
                    <pre id="qualityout" style="margin-top: 15px;">Loading...</pre>
                </div>
            </section>
            
            <!-- Model Governance -->
            <section id="governance" class="page">
                <div class="card">
                    <h2>📋 Model Governance</h2>
                    <p class="muted">Model manifests, Champion status, and governance history.</p>
                    <button onclick="loadGov()">🔄 Refresh Governance</button>
                    <pre id="govout" style="margin-top: 15px;">Loading...</pre>
                </div>
            </section>
            
            <!-- MLflow -->
            <section id="mlflow" class="page">
                <div class="card">
                    <h2>🔬 MLflow Experiments</h2>
                    <p class="muted">Track experiments, parameters, metrics, and model manifests.</p>
                    
                    <div style="margin: 15px 0;">
                        <button onclick="startMLflow()">▶️ Start Local MLflow</button>
                        <button onclick="checkMLflow()">📡 Check Connection</button>
                        <button onclick="loadMLRuns()">📋 Load Runs</button>
                    </div>
                    
                    <pre id="mlout" style="margin-top: 15px;">Checking...</pre>
                </div>
            </section>
            
            <!-- AWS S3 -->
            <section id="aws" class="page">
                <div class="card">
                    <h2>☁️ AWS S3 Integration</h2>
                    <p class="muted">
                        Real AWS S3 integration using boto3. Credentials are never stored in the UI.
                        Uses AWS standard credential chain. Actions include bucket validation, object listing, and Champion upload.
                    </p>
                    
                    <div style="margin: 15px 0;">
                        <button onclick="loadCloud()">🔌 Test S3 Connection</button>
                        <button onclick="listS3()">📁 List Objects</button>
                        <button onclick="uploadChampion()">📤 Upload Champion</button>
                    </div>
                    
                    <pre id="awsout" style="margin-top: 15px;">Not checked yet.</pre>
                </div>
            </section>
            
            <!-- Kubernetes -->
            <section id="k8s" class="page">
                <div class="card">
                    <h2>🚢 Kubernetes / EKS</h2>
                    <p class="muted">
                        Real Kubernetes integration through kubectl. Shows context, node health, pods, and deployments.
                        Deployment mutation is disabled by default for safety.
                    </p>
                    
                    <div style="margin: 15px 0;">
                        <button onclick="loadK8s()">🔌 Check Cluster</button>
                        <button onclick="k8sRes('pods')">📦 Pods</button>
                        <button onclick="k8sRes('deployments')">🚀 Deployments</button>
                        <button onclick="deployK8s()">⚠️ Deploy Manifest</button>
                    </div>
                    
                    <pre id="k8sout" style="margin-top: 15px;">Not checked yet.</pre>
                </div>
            </section>
            
            <!-- Explanations -->
            <section id="docs" class="page">
                <div class="card">
                    <h2>📖 Explanations & Architecture</h2>
                    <div id="docbody"></div>
                </div>
            </section>
        </main>
    </div>
    
    <script>
        // State management
        const pages = [...document.querySelectorAll('.page')];
        const navs = [...document.querySelectorAll('.nav')];
        
        // Navigation
        navs.forEach(n => n.onclick = () => showPage(n.dataset.page));
        
        function showPage(id) {
            navs.forEach(x => x.classList.toggle('active', x.dataset.page === id));
            pages.forEach(p => p.classList.toggle('active', p.id === id));
            history.replaceState(null, '', '#' + id);
            
            // Load data when page is shown
            if (id === 'quality') loadQuality();
            if (id === 'governance') loadGov();
            if (id === 'mlflow') checkMLflow();
            if (id === 'aws') loadCloud();
            if (id === 'k8s') loadK8s();
            if (id === 'docs') renderDocs();
        }
        
        // Handle hash navigation
        if (location.hash && document.querySelector('.nav[data-page="' + location.hash.slice(1) + '"]')) {
            showPage(location.hash.slice(1));
        }
        
        // API helper
        async function api(url, options) {
            const response = await fetch(url, options);
            let data = {};
            try {
                data = await response.json();
            } catch (e) {}
            
            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}`);
            }
            return data;
        }
        
        // Escape HTML
        function esc(s) {
            return String(s ?? '').replace(/[&<>"']/g, c => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
            }[c]));
        }
        
        // Load dashboard data
        async function loadDashboard() {
            try {
                const [status, prediction, history] = await Promise.all([
                    api('/api/status'),
                    api('/api/prediction'),
                    api('/api/price-history')
                ]);
                
                // Prediction
                document.getElementById('pred').textContent = '$' + Number(prediction.prediction).toFixed(2);
                document.getElementById('predmeta').textContent = 
                    prediction.prediction_date + ' · Last Price: $' + Number(prediction.last_price).toFixed(2);
                document.getElementById('predSource').textContent = 
                    prediction.source === 'Champion' ? '🏆 Champion Model' : '📊 Persistence Baseline';
                document.getElementById('predSource').className = 
                    prediction.source === 'Champion' ? 'badge success' : 'badge warning';
                
                // Champion
                document.getElementById('champ').textContent = 
                    status.champion ? status.champion.model : 'Persistence Baseline';
                document.getElementById('champmeta').textContent = 
                    status.champion ? 
                    `RMSE: ${Number(status.champion.metrics.rmse).toFixed(3)} · ${status.champion.updated_at}` :
                    'No Champion yet - run benchmark to create one';
                document.getElementById('champStatus').textContent = 
                    status.champion ? '✅ Active' : '⚠️ No Champion';
                document.getElementById('champStatus').className = 
                    status.champion ? 'badge success' : 'badge warning';
                
                // Draw professional chart
                drawProfessionalChart(history.dates, history.prices);
            } catch (error) {
                document.getElementById('pred').textContent = 'Unavailable';
                document.getElementById('predmeta').textContent = error.message;
            }
        }
        
        // Professional chart using Plotly
        function drawProfessionalChart(dates, prices) {
            if (!dates.length) return;
            
            // Create trace
            const trace = {
                x: dates,
                y: prices,
                mode: 'lines+markers',
                type: 'scatter',
                name: 'WTI Close Price',
                line: {
                    color: '#e4bd4f',
                    width: 3,
                    shape: 'spline',
                    smoothing: 0.5
                },
                marker: {
                    color: '#e4bd4f',
                    size: 8,
                    line: {
                        color: '#fff',
                        width: 1
                    }
                },
                fill: 'tozeroy',
                fillcolor: 'rgba(228, 189, 79, 0.08)',
                hovertemplate: '<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
            };
            
            // Layout configuration
            const layout = {
                title: {
                    text: 'WTI Crude Oil Price History',
                    font: {
                        size: 18,
                        color: '#e8eef7',
                        family: 'Inter, sans-serif'
                    },
                    x: 0.5,
                    xanchor: 'center'
                },
                xaxis: {
                    title: 'Date',
                    gridcolor: 'rgba(32, 50, 74, 0.5)',
                    zerolinecolor: 'rgba(32, 50, 74, 0.5)',
                    tickfont: { color: '#8ea2ba', size: 12 },
                    tickangle: -45,
                    rangeslider: {
                        visible: true,
                        thickness: 0.1,
                        bgcolor: '#0a1525',
                        bordercolor: '#315779'
                    }
                },
                yaxis: {
                    title: 'Price (USD/barrel)',
                    gridcolor: 'rgba(32, 50, 74, 0.5)',
                    zerolinecolor: 'rgba(32, 50, 74, 0.5)',
                    tickfont: { color: '#8ea2ba', size: 12 },
                    tickprefix: '$',
                    rangemode: 'tozero'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(10, 21, 37, 0.8)',
                font: {
                    color: '#e8eef7',
                    family: 'Inter, sans-serif'
                },
                hoverlabel: {
                    bgcolor: '#0b2a46',
                    bordercolor: '#78bfff',
                    font: {
                        color: '#78bfff',
                        size: 14
                    }
                },
                margin: {
                    l: 80,
                    r: 30,
                    t: 60,
                    b: 60
                },
                annotations: [
                    {
                        text: '🔍 Zoom | ↕ Pan | 🔄 Reset',
                        x: 0.02,
                        y: 1.05,
                        xref: 'paper',
                        yref: 'paper',
                        showarrow: false,
                        font: {
                            size: 12,
                            color: '#8ea2ba'
                        }
                    }
                ]
            };
            
            // Configuration options
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                modeBarButtonsToAdd: ['resetScale2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'wti_price_history',
                    width: 1920,
                    height: 1080,
                    scale: 1
                }
            };
            
            // Plot the chart
            Plotly.newPlot('priceChart', [trace], layout, config);
        }
        
        // Refresh data
        async function refreshData() {
            try {
                await api('/api/data/refresh', { method: 'POST' });
                await loadDashboard();
            } catch (error) {
                alert(`Error refreshing data: ${error.message}`);
            }
        }
        
        // Initialize models dropdown
        async function initModels() {
            try {
                const status = await api('/api/status');
                const select = document.getElementById('model');
                ['XGBoost', 'Random Forest', 'Extra Trees', 'HistGradientBoosting'].forEach(x => {
                    select.add(new Option(x, x));
                });
            } catch (error) {
                console.error('Failed to initialize models:', error);
            }
        }
        
        // Run pipeline
        async function runPipeline() {
            try {
                await api('/api/pipeline/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: document.getElementById('model').value,
                        fine_tuning: document.getElementById('tune').checked
                    })
                });
                
                document.getElementById('plog').textContent = 'Pipeline started. Monitoring progress...';
                pollPipeline();
            } catch (error) {
                document.getElementById('plog').textContent = `Error: ${error.message}`;
            }
        }
        
        // Poll pipeline status
        async function pollPipeline() {
            try {
                const status = await api('/api/pipeline/status');
                
                document.getElementById('bar').style.width = status.progress + '%';
                document.getElementById('pmsg').textContent = 
                    `${status.progress}% · ${status.stage} · ${status.message}`;
                
                if (status.running) {
                    setTimeout(pollPipeline, 800);
                } else {
                    document.getElementById('plog').textContent = 
                        JSON.stringify(status.result || status.error, null, 2);
                    loadDashboard();
                }
            } catch (error) {
                document.getElementById('plog').textContent = `Polling error: ${error.message}`;
            }
        }
        
        // Model checkboxes
        const modelNames = ['XGBoost', 'Random Forest', 'Extra Trees', 'HistGradientBoosting'];
        document.getElementById('checks').innerHTML = modelNames.map(x => `
            <label class="check">
                <input type="checkbox" value="${x}" checked>
                ${x}
            </label>
        `).join('');
        
        // Run benchmark
        async function runBenchmark() {
            const models = [...document.querySelectorAll('#checks input:checked')].map(x => x.value);
            if (!models.length) {
                alert('Please select at least one model');
                return;
            }
            
            try {
                await api('/api/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        models,
                        fine_tuning: document.getElementById('btune').checked,
                        auto_promote: document.getElementById('autop').checked
                    })
                });
                benchPoll();
            } catch (error) {
                document.getElementById('benchout').textContent = `Error: ${error.message}`;
            }
        }
        
        // Poll benchmark
        async function benchPoll() {
            try {
                const status = await api('/api/pipeline/status');
                
                if (status.running) {
                    document.getElementById('benchout').innerHTML = 
                        `<p><span class="loading-spinner"></span> ${status.progress}% · ${esc(status.message)}</p>`;
                    setTimeout(benchPoll, 800);
                } else if (status.error) {
                    document.getElementById('benchout').innerHTML = 
                        `<div class="bad">❌ ${esc(status.error)}</div>`;
                } else {
                    const result = status.result || {};
                    
                    // Results table
                    let html = '<h3>Benchmark Results</h3><table>';
                    html += '<tr><th>Rank</th><th>Model</th><th>RMSE</th><th>MAE</th><th>Baseline</th><th>Improvement</th><th>Status</th></tr>';
                    
                    (result.results || []).forEach((x, i) => {
                        html += `<tr>
                            <td>${i + 1}</td>
                            <td>${esc(x.model)}</td>
                            <td>${x.rmse != null ? Number(x.rmse).toFixed(3) : '—'}</td>
                            <td>${x.mae != null ? Number(x.mae).toFixed(3) : '—'}</td>
                            <td>${x.baseline_rmse != null ? Number(x.baseline_rmse).toFixed(3) : '—'}</td>
                            <td>${x.improvement_pct != null ? Number(x.improvement_pct).toFixed(2) + '%' : '—'}</td>
                            <td class="${x.status === 'SUCCESS' ? 'ok' : 'bad'}">${x.status}</td>
                        </tr>`;
                    });
                    
                    html += '</table>';
                    
                    // Best model info
                    if (result.best) {
                        html += `<div class="ok" style="margin: 15px 0;">
                            <strong>🏆 Best Model:</strong> ${esc(result.best.model)} 
                            (RMSE: ${Number(result.best.rmse).toFixed(3)}, Improvement: ${Number(result.best.improvement_pct).toFixed(2)}%)
                        </div>`;
                    }
                    
                    // Promotion info
                    if (result.promotion) {
                        html += `<div class="ok" style="margin: 10px 0;">
                            <strong>✅ Champion Promoted:</strong> ${esc(result.promotion.model_id)}
                            <span class="badge success">${esc(result.promotion.governance)}</span>
                        </div>`;
                    }
                    
                    // Full JSON for debugging
                    html += '<pre>' + esc(JSON.stringify({
                        best: result.best,
                        promotion: result.promotion,
                        rows: result.rows,
                        features: result.features
                    }, null, 2)) + '</pre>';
                    
                    document.getElementById('benchout').innerHTML = html;
                    loadDashboard();
                }
            } catch (error) {
                document.getElementById('benchout').textContent = `Error: ${error.message}`;
            }
        }
        
        // Data Quality
        async function loadQuality() {
            try {
                document.getElementById('qualityout').textContent = 
                    JSON.stringify(await api('/api/data-quality'), null, 2);
            } catch (error) {
                document.getElementById('qualityout').textContent = `Error: ${error.message}`;
            }
        }
        
        // Governance
        async function loadGov() {
            try {
                const status = await api('/api/status');
                document.getElementById('govout').textContent = 
                    JSON.stringify(status.champion || { status: 'No Champion' }, null, 2);
            } catch (error) {
                document.getElementById('govout').textContent = `Error: ${error.message}`;
            }
        }
        
        // MLflow
        async function checkMLflow() {
            try {
                document.getElementById('mlout').textContent = 
                    JSON.stringify(await api('/api/mlflow'), null, 2);
            } catch (error) {
                document.getElementById('mlout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function startMLflow() {
            try {
                const result = await api('/api/mlflow/start', { method: 'POST' });
                document.getElementById('mlout').textContent = JSON.stringify(result, null, 2);
                setTimeout(checkMLflow, 1200);
            } catch (error) {
                document.getElementById('mlout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function loadMLRuns() {
            try {
                document.getElementById('mlout').textContent = 
                    JSON.stringify(await api('/api/mlflow/runs'), null, 2);
            } catch (error) {
                document.getElementById('mlout').textContent = `Error: ${error.message}`;
            }
        }
        
        // AWS S3
        async function loadCloud() {
            try {
                const data = await api('/api/cloud');
                document.getElementById('awsout').textContent = 
                    JSON.stringify(data.s3, null, 2);
            } catch (error) {
                document.getElementById('awsout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function listS3() {
            try {
                document.getElementById('awsout').textContent = 
                    JSON.stringify(await api('/api/s3/objects'), null, 2);
            } catch (error) {
                document.getElementById('awsout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function uploadChampion() {
            try {
                const result = await api('/api/s3/upload-champion', { method: 'POST' });
                document.getElementById('awsout').textContent = JSON.stringify(result, null, 2);
                alert('✅ Champion uploaded to S3 successfully!');
            } catch (error) {
                document.getElementById('awsout').textContent = `Error: ${error.message}`;
            }
        }
        
        // Kubernetes
        async function loadK8s() {
            try {
                const data = await api('/api/cloud');
                document.getElementById('k8sout').textContent = 
                    JSON.stringify(data.kubernetes, null, 2);
            } catch (error) {
                document.getElementById('k8sout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function k8sRes(kind) {
            try {
                document.getElementById('k8sout').textContent = 
                    JSON.stringify(await api('/api/kubernetes/resources?kind=' + kind), null, 2);
            } catch (error) {
                document.getElementById('k8sout').textContent = `Error: ${error.message}`;
            }
        }
        
        async function deployK8s() {
            const confirm = prompt('Type DEPLOY to confirm deployment:');
            if (confirm !== 'DEPLOY') return;
            
            try {
                document.getElementById('k8sout').textContent = 
                    JSON.stringify(await api('/api/kubernetes/apply?confirm=DEPLOY', { method: 'POST' }), null, 2);
                alert('✅ Deployment initiated successfully!');
            } catch (error) {
                document.getElementById('k8sout').textContent = `Error: ${error.message}`;
            }
        }
        
        // Explanations
        function renderDocs() {
            document.getElementById('docbody').innerHTML = `
                <h3>📊 Pipeline Architecture</h3>
                <div class="flow">
                    <span class="node">📥 Yahoo Finance</span>
                    <span class="arrow">→</span>
                    <span class="node">💾 Cache</span>
                    <span class="arrow">→</span>
                    <span class="node">🧪 Features</span>
                    <span class="arrow">→</span>
                    <span class="node">📊 Benchmark</span>
                    <span class="arrow">→</span>
                    <span class="node">🏆 Champion</span>
                    <span class="arrow">→</span>
                    <span class="node">📈 Prediction</span>
                </div>
                
                <h3>📖 Page Explanations</h3>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>📊 Dashboard</h4>
                    <p><strong>What it does:</strong> Shows the next trading-day prediction, Champion model info, and the complete price history chart.</p>
                    <p><strong>Why needed:</strong> Provides an executive summary of the system's current state.</p>
                    <p><strong>Inputs:</strong> Prediction API, Status API, Price History API</p>
                    <p><strong>Outputs:</strong> Professional interactive price chart</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>⚙️ Pipeline Control</h4>
                    <p><strong>What it does:</strong> Runs the complete ML pipeline from data loading to model training.</p>
                    <p><strong>Why needed:</strong> One-click training workflow for any selected model.</p>
                    <p><strong>Inputs:</strong> Model selection, fine-tuning option</p>
                    <p><strong>Outputs:</strong> Trained model, metrics, Champion promotion</p>
                    <p><strong>Dependencies:</strong> Yahoo Finance, XGBoost, scikit-learn</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>🏆 Model Benchmark Center</h4>
                    <p><strong>What it does:</strong> Compares all selected models under identical conditions.</p>
                    <p><strong>Why needed:</strong> Ensures fair model selection and promotes the best model.</p>
                    <p><strong>Inputs:</strong> Multiple models, fine-tuning option</p>
                    <p><strong>Outputs:</strong> Ranked results table, Champion promotion</p>
                    <p><strong>Dependencies:</strong> Walk-forward validation, TimeSeriesSplit</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>🔍 Data Quality</h4>
                    <p><strong>What it does:</strong> Validates the dataset's integrity.</p>
                    <p><strong>Why needed:</strong> Ensures predictions are based on reliable data.</p>
                    <p><strong>Inputs:</strong> CSV/API data</p>
                    <p><strong>Outputs:</strong> Quality metrics report</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>☁️ AWS S3</h4>
                    <p><strong>What it does:</strong> Provides real AWS S3 integration for storing Champion models.</p>
                    <p><strong>Why needed:</strong> Persists models in cloud for production use.</p>
                    <p><strong>Inputs:</strong> Champion artifacts</p>
                    <p><strong>Outputs:</strong> Upload confirmation</p>
                    <p><strong>Dependencies:</strong> boto3, AWS credentials</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>🚢 Kubernetes/EKS</h4>
                    <p><strong>What it does:</strong> Integrates with Kubernetes/EKS through kubectl.</p>
                    <p><strong>Why needed:</strong> Enables container orchestration and deployment.</p>
                    <p><strong>Inputs:</strong> kubectl configuration</p>
                    <p><strong>Outputs:</strong> Cluster status, deployment results</p>
                    <p><strong>Dependencies:</strong> kubectl, K8s cluster access</p>
                </div>
                
                <div class="card" style="margin: 15px 0;">
                    <h4>🔬 MLflow</h4>
                    <p><strong>What it does:</strong> Tracks experiments, parameters, and metrics.</p>
                    <p><strong>Why needed:</strong> Provides reproducibility and experiment comparison.</p>
                    <p><strong>Inputs:</strong> Model training data</p>
                    <p><strong>Outputs:</strong> Experiment runs, metrics</p>
                    <p><strong>Dependencies:</strong> MLflow server</p>
                </div>
            `;
        }
        
        // Initialize
        initModels();
        loadDashboard();
    </script>
</body>
</html>'''
"""
Data ingestion from Yahoo Finance with caching.
"""
from pathlib import Path
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.project_config import RAW, SYMBOL, WINDOW_DAYS, CACHE_MAX_AGE_HOURS

SNAPSHOT = RAW / 'wti_daily.csv'


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataframe."""
    cols = [c for c in ('Date', 'price') if c in df.columns]
    if len(cols) != 2:
        raise RuntimeError('Data must contain Date and price columns')
    
    df = df[cols].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    df = df.dropna().drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    
    if len(df) < 60:
        raise RuntimeError(f'Insufficient WTI history: {len(df)} rows')
    
    return df


def load_data(force: bool = False) -> pd.DataFrame:
    """Load WTI price data, either from cache or from Yahoo Finance."""
    # Check if cache exists and is fresh
    if SNAPSHOT.exists() and not force:
        age_h = (time.time() - SNAPSHOT.stat().st_mtime) / 3600
        if age_h <= CACHE_MAX_AGE_HOURS:
            return _clean(pd.read_csv(SNAPSHOT))
    
    # Download from Yahoo Finance
    import yfinance as yf
    
    last_err = None
    for wait in (0, 2, 5):
        if wait:
            time.sleep(wait)
        
        try:
            # Calculate period based on WINDOW_DAYS
            period_years = max(2, WINDOW_DAYS // 365 + 1)
            
            df = yf.download(
                SYMBOL,
                period=f'{period_years}y',
                auto_adjust=False,
                progress=False,
                threads=False
            )
            
            if df is None or df.empty:
                raise RuntimeError('Yahoo Finance returned no data')
            
            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index().rename(columns={'Close': 'price'})
            df = _clean(df)
            
            # Save to cache
            df.to_csv(SNAPSHOT, index=False)
            
            return df
            
        except Exception as e:
            last_err = e
    
    # If download fails, try to use existing cache
    if SNAPSHOT.exists():
        return _clean(pd.read_csv(SNAPSHOT))
    
    raise RuntimeError(f'Unable to obtain WTI data: {last_err}')
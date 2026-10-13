"""MF 731 Homework 05: Yahoo closes, empirical/EWMA/normal VaR.
Run: python solve.py --download
Replay the exact saved dataset: python solve.py
Dependencies: numpy, pandas. All loss values are USD.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TICKERS = ['MSFT', 'AAPL', 'GOOGL']
START, END = '2024-05-23', '2026-05-22'
V = 1_000_000.0
CAPS = np.array([3.109, 4.536, 4.640])
DECAY = 0.97
ALPHA = 0.95
N_MC = 100_000
SEED = 731

def download_prices() -> pd.DataFrame:
    start = int(dt.datetime.fromisoformat(START).replace(tzinfo=dt.timezone.utc).timestamp())
    end = int((dt.datetime.fromisoformat(END) + dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
    series = []
    for ticker in TICKERS:
        query = urllib.parse.urlencode({'period1':start, 'period2':end, 'interval':'1d', 'events':'history'})
        last_error = None
        for attempt in range(4):
            host = 'query1' if attempt % 2 == 0 else 'query2'
            url = f'https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}?{query}'
            try:
                req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    payload = json.load(response)
                if payload['chart'].get('error'):
                    raise RuntimeError(payload['chart']['error'])
                item = payload['chart']['result'][0]
                dates = [dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat() for t in item['timestamp']]
                # Use actual Close, NOT adjclose or dividend-adjusted prices.
                s = pd.Series(item['indicators']['quote'][0]['close'], index=dates, name=ticker)
                series.append(s)
                break
            except (urllib.error.URLError, ValueError, KeyError, RuntimeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 * (attempt+1))
        else:
            raise RuntimeError(f'Yahoo download failed for {ticker}: {last_error}')
    prices = pd.concat(series, axis=1).sort_index()
    prices.index.name = 'Date'
    validate_prices(prices)
    return prices

def validate_prices(prices: pd.DataFrame) -> None:
    if list(prices.columns) != TICKERS:
        raise ValueError('Unexpected stock columns/order')
    if prices.empty or prices.index[0] != START or prices.index[-1] != END:
        raise ValueError('Dataset must cover both specified endpoint trading dates')
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Dates must be unique and sorted')
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Missing or invalid closes; do not silently drop rows')
    if any(dt.date.fromisoformat(str(d)).weekday() >= 5 for d in prices.index):
        raise ValueError('Weekend observation')

def quantile(values: np.ndarray, alpha: float = ALPHA) -> float:
    # Lower generalized inverse of the empirical CDF, with no interpolation.
    rank = math.ceil(alpha * len(values))
    return float(np.partition(values, rank - 1)[rank - 1])

def ewma(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(3)
    cov = np.zeros((3, 3))
    for x in returns:
        residual = x - mean  # IMPORTANT: old mean, before updating it
        cov = DECAY * cov + (1 - DECAY) * np.outer(residual, residual)
        mean = DECAY * mean + (1 - DECAY) * x
    return mean, cov

def losses(x: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    return {
        'full': -V * (np.expm1(x) @ weights),
        'linear': -V * (x @ weights),
        'quadratic': -V * ((x + 0.5 * x**2) @ weights),
    }

def analyze(prices: pd.DataFrame) -> dict:
    validate_prices(prices)
    weights = CAPS / CAPS.sum()
    x = np.diff(np.log(prices.to_numpy()), axis=0)
    mean, cov = ewma(x)
    if np.min(np.linalg.eigvalsh(cov)) < -1e-14:
        raise ValueError('EWMA covariance is not positive semidefinite')
    empirical_losses = losses(x, weights)
    historical = {k:quantile(v) for k,v in empirical_losses.items()}
    rng = np.random.default_rng(SEED)
    # Identical normal draws for all three loss operators.
    samples = rng.multivariate_normal(mean, cov, size=N_MC, check_valid='raise')
    simulated_losses = losses(samples, weights)
    normal = {k:quantile(v) for k,v in simulated_losses.items()}
    port_mean = -V * float(weights @ mean)
    port_sd = V * math.sqrt(float(weights @ cov @ weights))
    normal['linear_mc_check'] = normal['linear']
    normal['linear'] = port_mean + NormalDist().inv_cdf(ALPHA) * port_sd
    batch_se = {k:float(np.std([quantile(batch) for batch in np.array_split(v,20)],ddof=1)/math.sqrt(20)) for k,v in simulated_losses.items()}
    # Independent checks of the normal linear-loss mean and variance.
    assert abs(float(simulated_losses['linear'].mean())-port_mean) < 5*port_sd/math.sqrt(N_MC)
    assert abs(float(simulated_losses['linear'].std(ddof=1))/port_sd-1) < 0.02
    latest = prices.iloc[-1].to_numpy()
    tail_idx = np.argsort(empirical_losses['full'])[-5:][::-1]
    return {
        'source':'Yahoo Finance chart API, unadjusted daily Close',
        'date_start':START,'date_end':END,'tickers':TICKERS,
        'price_observations':len(prices),'return_observations':len(x),
        'historical_quantile_rank':math.ceil(ALPHA*len(x)),
        'first_five':prices.head().reset_index().to_dict('records'),
        'last_five':prices.tail().reset_index().to_dict('records'),
        'market_caps_trillion':CAPS.tolist(),'weights':weights.tolist(),
        'dollar_allocations':(V*weights).tolist(),
        'shares':(V*weights/latest).tolist(),
        'ewma_initial_mean':[0,0,0],'ewma_initial_covariance':np.zeros((3,3)).tolist(),
        'decay':DECAY,'mean':mean.tolist(),'covariance':cov.tolist(),
        'daily_volatility':np.sqrt(np.diag(cov)).tolist(),
        'correlation':(cov/np.outer(np.sqrt(np.diag(cov)),np.sqrt(np.diag(cov)))).tolist(),
        'historical_var':historical,'normal_var':normal,
        'normal_linear_loss_mean':port_mean,'normal_linear_loss_sd':port_sd,
        'normal_linear_variance_factor':float(weights@cov@weights),
        'mc_samples':N_MC,'mc_seed':SEED,'mc_batch_standard_error':batch_se,
        'worst_five_historical_full_losses':[{'Date':prices.index[int(i)+1], 'loss':float(empirical_losses['full'][i]), 'log_returns':x[i].tolist()} for i in tail_idx],
        'calendar_note':'2026-05-25 is Memorial Day; next US stock trading day is 2026-05-26. Treat the question as a one-trading-day forecast; no sqrt(3) or sqrt(4) scaling.',
        'numpy_version':np.__version__,'pandas_version':pd.__version__,
    }

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Refresh Yahoo Finance closes; otherwise use the committed snapshot')
    args=parser.parse_args()
    path=ROOT/'prices.csv'
    if args.download:
        prices=download_prices()
        prices.to_csv(path,float_format='%.17g')
    if not path.exists():
        raise FileNotFoundError('Run with --download first, or restore the repository prices.csv snapshot')
    prices=pd.read_csv(path,index_col='Date')
    results=analyze(prices)
    results['prices_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    log_returns=pd.DataFrame(np.diff(np.log(prices.to_numpy()),axis=0),index=prices.index[1:],columns=TICKERS)
    log_returns.index.name='Date'
    log_returns.to_csv(ROOT/'log_returns.csv',float_format='%.17g')
    print(json.dumps(results,indent=2,allow_nan=False))

if __name__=='__main__':
    main()

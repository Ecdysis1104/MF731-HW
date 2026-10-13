import json
import sys
from pathlib import Path
from statistics import NormalDist
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
file = root / 'prices.csv'
stocks = ['MSFT', 'AAPL', 'GOOGL']
if '--download' in sys.argv or not file.exists():
    start = int(pd.Timestamp('2024-05-23', tz='UTC').timestamp())
    end = int(pd.Timestamp('2026-05-23', tz='UTC').timestamp())
    columns = []
    for stock in stocks:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{stock}?period1={start}&period2={end}&interval=1d'
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.load(response)['chart']['result'][0]
        dates = pd.to_datetime(data['timestamp'], unit='s', utc=True).strftime('%Y-%m-%d')
        columns.append(pd.Series(data['indicators']['quote'][0]['close'], index=dates, name=stock))
    pd.concat(columns, axis=1).sort_index().to_csv(file, index_label='Date')
p = pd.read_csv(file, index_col='Date')[stocks]
x = np.diff(np.log(p.to_numpy()), axis=0)
w = np.array([3.109, 4.536, 4.640]) / 12.285
mu, cov = np.zeros(3), np.zeros((3, 3))
for r in x:
    d = r - mu
    cov = 0.97 * cov + 0.03 * np.outer(d, d)
    mu = 0.97 * mu + 0.03 * r

def losses(r):
    return -1e6 * np.column_stack((np.expm1(r) @ w, r @ w, (r + r**2 / 2) @ w))

historical = np.quantile(losses(x), 0.95, axis=0, method='inverted_cdf')
y = np.random.default_rng(731).multivariate_normal(mu, cov, 100000)
normal = np.quantile(losses(y), 0.95, axis=0, method='inverted_cdf')
normal[1] = 1e6 * (-w @ mu + NormalDist().inv_cdf(0.95) * np.sqrt(w @ cov @ w))
var = pd.DataFrame({'Historical': historical, 'Normal': normal}, index=['Full', 'Linear', 'Quadratic'])
print(len(p), len(x))
print(p.head())
print(p.tail())
print(w)
print(mu)
print(cov)
print(var)

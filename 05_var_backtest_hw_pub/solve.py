import json
import sys
from pathlib import Path
from statistics import NormalDist
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
file = root / 'prices.csv'
if '--download' in sys.argv or not file.exists():
    start = int(pd.Timestamp('2016-05-20', tz='UTC').timestamp())
    end = int(pd.Timestamp('2026-05-23', tz='UTC').timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={start}&period2={end}&interval=1d'
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
        data = json.load(response)['chart']['result'][0]
    dates = pd.to_datetime(data['timestamp'], unit='s', utc=True).strftime('%Y-%m-%d')
    pd.Series(data['indicators']['quote'][0]['close'], index=dates, name='Close').to_csv(file, index_label='Date')
p = pd.read_csv(file, index_col='Date')['Close']
x = np.diff(np.log(p.to_numpy()))
loss = -np.expm1(x)
M, J = 1010, len(x) - 1010
windows = np.lib.stride_tricks.sliding_window_view(loss, M)[:-1]
historical = np.quantile(windows, 0.95, axis=1, method='inverted_cdf')
mu, variance = x[:M].mean(), x[:M].var(ddof=1)
initial = np.array([mu, variance])
normal = np.empty(J)
z = NormalDist().inv_cdf(0.05)
for j, r in enumerate(x[M:]):
    normal[j] = -np.expm1(mu + np.sqrt(variance) * z)
    variance = 0.97 * variance + 0.03 * (r - mu)**2
    mu = 0.97 * mu + 0.03 * r
out = pd.DataFrame({'Loss': loss[M:], 'HistoricalVaR': historical, 'HistoricalExceedance': (loss[M:] > historical).astype(int),
                    'NormalVaR': normal, 'NormalExceedance': (loss[M:] > normal).astype(int)}, index=p.index[M+1:])
out.to_csv(root / 'backtest.csv', index_label='Date')
k = np.arange(J)
pmf = 0.95**J * np.r_[1, np.cumprod((J - k) / (k + 1) * 0.05 / 0.95)]
interval = np.searchsorted(pmf.cumsum(), [0.025, 0.975])
print(p.iloc[1:].head(), p.tail(), len(x), J, initial, interval, sep='\n')
print(out[['HistoricalExceedance', 'NormalExceedance']].sum())
print(out.head(), out.tail(), sep='\n')


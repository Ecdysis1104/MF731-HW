# Homework 05: VaR estimation

MF 731 Corporate Risk Management. Stock order: MSFT, AAPL, GOOGL.

## Reproduce the submitted values

Use Python 3.12 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python solve.py
```

The default run uses the committed `prices.csv` snapshot. It reproduces `results.json` and creates `log_returns.csv`; internet access is not required. To refresh the original Yahoo Finance daily **Close** data:

```bash
python solve.py --download
```

Yahoo may revise its historical data, so a fresh download can differ from the snapshot used in the homework. The CSV hash is saved in `results.json`.

## Choices used in the solution

- Inclusive date range: 2024-05-23 to 2026-05-22.
- Unadjusted daily Close, not Adjusted Close; 501 dates and 500 return vectors.
- Portfolio: USD 1,000,000. Weights use the assignment's supplied capitalizations: 3.109, 4.536, 4.640 trillion.
- Historical and MC quantiles: lower generalized inverse, order statistic ceil(0.95*n), without interpolation.
- EWMA: zero initial mean and covariance, decay 0.97, covariance residual uses the OLD mean.
- MC: 100,000 draws, NumPy default_rng seed 731, shared vectors for the loss operators. Linear normal VaR uses the analytical formula.
- Forecast horizon: one trading day. The assignment's May 25, 2026 date is a holiday (Memorial Day); the next US equity trading day is May 26. No calendar-day square-root scaling.

## Results (USD)

| Loss operator | Historical 95% VaR | EWMA normal 95% VaR |
|---|---:|---:|
| Full | 21,043.05 | 16,609.78 |
| Linear | 21,415.73 | 16,869.11 |
| Quadratic | 21,038.61 | 16,605.55 |

## Sources

- [Yahoo Finance historical prices](https://finance.yahoo.com/quote/MSFT/history/) (the code uses the Yahoo chart API for all three stocks).
- [Nasdaq trading calendar](https://www.nasdaqtrader.com/Trader.aspx?id=Calendar).

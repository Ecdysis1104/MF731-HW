import numpy as np
import yfinance as yf
from statistics import NormalDist

M, K, r = 50000, 10, 0.97
rng = np.random.default_rng(731)
p = yf.download(["MSFT", "AAPL"], start="2011-08-31", end="2016-09-02",
                auto_adjust=False, progress=False, threads=False)["Adj Close"][["MSFT", "AAPL"]]
mu, S = np.zeros(2), np.zeros((2, 2))
for x in np.diff(np.log(p.to_numpy()), axis=0):
    e = x - mu
    S = r * S + (1 - r) * np.outer(e, e)
    mu = r * mu + (1 - r) * x

v = 1e6 * np.array([448.77, 575.11]) / (448.77 + 575.11)
sd = np.sqrt(np.diag(S))
rho = S[0, 1] / np.prod(sd)
V1 = -v @ mu + NormalDist().inv_cdf(0.95) * np.sqrt(v @ S @ v)
V = np.sqrt(K) * V1
x = np.array([mu[0] - 5 * sd[0] * (rho + np.sqrt(1 - rho**2)), mu[1] - 5 * sd[1]])
print("mu:", mu, "\nSigma:", S, "\nrho:", rho)
print("VaR1, VaR10, capital:", V1, V, 3 * V)
print("shock:", x)
e = x - mu
S = r * S + (1 - r) * np.outer(e, e)
mu = r * mu + (1 - r) * x
print("mu2:", mu, "\nSigma2:", S)
mu, S, total = np.tile(mu, (M, 1)), np.tile(S, (M, 1, 1)), np.tile(x, (M, 1))
for k in range(2, K + 1):
    e = (np.linalg.cholesky(S) @ rng.standard_normal((M, 2, 1))).squeeze(-1)
    x = mu + e
    total += x
    if k < K:
        S = r * S + (1 - r) * e[:, :, None] * e[:, None, :]
        mu = r * mu + (1 - r) * x
loss = -total @ v
print("mean, VaR95:", loss.mean(), np.quantile(loss, 0.95))
print("exceedance counts:", np.sum(loss > V), np.sum(loss > 3 * V))
print("exceedance percentages:", 100 * np.mean(loss > V), 100 * np.mean(loss > 3 * V))

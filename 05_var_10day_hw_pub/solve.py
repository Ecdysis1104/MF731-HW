import numpy as np
from scipy.special import ndtr

S0, strike, T, r, sigma, mu = 152.51, 170, 0.292, 0.00119, 0.4907, 0.1691
dt, n = 1 / 252, 40000

def put(t, s):
    tau = T - t
    d1 = (np.log(s / strike) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return strike * np.exp(-r * tau) * ndtr(-d2) - s * ndtr(-d1), ndtr(d1) - 1

p0, delta = put(0, S0)
s = np.full(n, S0)
h = np.full(n, 100 * delta)
cash = -h * s
v0 = -100 * p0
z = np.random.default_rng(731).standard_normal((10, n))
for j in range(10):
    s *= np.exp((mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z[j])
    cash *= np.exp(r * dt)
    p, delta = put((j + 1) * dt, s)
    loss = v0 - (h * s + cash - 100 * p)
    if j == 0:
        q1 = np.quantile(loss, 0.95, method='inverted_cdf')
    cash += (h - 100 * delta) * s
    h = 100 * delta
q10 = np.quantile(loss, 0.95, method='inverted_cdf')
print(q1, q1 * np.sqrt(10), q10)

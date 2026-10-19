from itertools import product
from math import exp
import json

r, T, H = 0.0132, 0.25, 5 / 252
s1, s2, v1, v2, rho = 158.12, 170.33, 0.2214, 0.3036, 0.237
returns = {-0.2: 0.5, -0.1: 0.75, -0.05: 1, 0.05: 1, 0.1: 0.75, 0.2: 0.5}
vols = {0.5: 0.5, 0.75: 1.25, 1.25: 0.75, 1.5: 0.75, 1.75: 1.25, 2: 0.5}
shifts = {-0.5: 1.25, 0: 0.5, 0.5: 1.25}


def price(a, b, u, v, c, tau):
    return exp(r * tau) * (a**2 * exp(u**2 * tau)
        - 2 * a * b * exp(c * u * v * tau) + b**2 * exp(v**2 * tau))


h1 = 2 * exp(r * T) * (s1 * exp(v1**2 * T) - s2 * exp(rho * v1 * v2 * T))
h2 = 2 * exp(r * T) * (s2 * exp(v2**2 * T) - s1 * exp(rho * v1 * v2 * T))
p0 = price(s1, s2, v1, v2, rho, T) - h1 * s1 - h2 * s2
scenarios = []

for x1, x2, m1, m2, d in product(returns, returns, vols, vols, shifts):
    a, b = s1 * exp(x1), s2 * exp(x2)
    u, v, c = v1 * m1, v2 * m2, rho + d
    loss = p0 - (price(a, b, u, v, c, T - H) - h1 * a - h2 * b)
    w = returns[x1] * returns[x2] * vols[m1] * vols[m2] * shifts[d]
    scenarios.append(dict(x1=x1, x2=x2, m1=m1, m2=m2, sigma1=u, sigma2=v,
                          correlation=c, loss=loss, weight=w, weighted_loss=w * loss))

print(json.dumps({
    "b": max(scenarios, key=lambda s: s["loss"]),
    "c": max(scenarios, key=lambda s: s["weighted_loss"])
}, indent=2))

"""MF731 Scenario Analysis. Python 3 standard library only.
Run: python scenario_analysis.py
Optionally export all scenarios: python scenario_analysis.py --csv scenarios.csv
"""
from itertools import product
from math import exp, isclose
import argparse
import csv
import json

R, T, H = 0.0132, 0.25, 5 / 252
S1, S2, SIG1, SIG2, CORR = 158.12, 170.33, 0.2214, 0.3036, 0.237
RETURNS = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
MULTIPLIERS = (0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
SHIFTS = (-0.5, 0.0, 0.5)

def price(s1, s2, v1, v2, corr, tau):
    return exp(R*tau)*(s1*s1*exp(v1*v1*tau)
           - 2*s1*s2*exp(v1*v2*corr*tau) + s2*s2*exp(v2*v2*tau))

def calculate():
    v0 = price(S1, S2, SIG1, SIG2, CORR, T)
    h1 = 2*exp(R*T)*(S1*exp(SIG1**2*T)-S2*exp(SIG1*SIG2*CORR*T))
    h2 = 2*exp(R*T)*(S2*exp(SIG2**2*T)-S1*exp(SIG1*SIG2*CORR*T))
    p0 = v0-h1*S1-h2*S2
    assert isclose(p0, -v0, abs_tol=1e-9)  # Euler identity, degree two
    wr = {0.20: 0.5, 0.10: 0.75, 0.05: 1.0}
    wm = {0.5: 0.5, 2.0: 0.5, 0.75: 1.25, 1.75: 1.25, 1.25: 0.75, 1.5: 0.75}
    wc = {0.0: 0.5, -0.5: 1.25, 0.5: 1.25}
    rows = []
    for x1,x2,m1,m2,dc in product(RETURNS,RETURNS,MULTIPLIERS,MULTIPLIERS,SHIFTS):
        s1,s2 = S1*exp(x1),S2*exp(x2)
        v1,v2,c = SIG1*m1,SIG2*m2,CORR+dc
        assert -1 < c < 1
        vh = price(s1,s2,v1,v2,c,T-H)
        ph = vh-h1*s1-h2*s2  # Shares fixed at their time-zero deltas
        loss = p0-ph
        weight = wr[abs(x1)]*wr[abs(x2)]*wm[m1]*wm[m2]*wc[dc]
        # Independent algebra: zero initial delta eliminates the linear term.
        old = price(s1,s2,SIG1,SIG2,CORR,T)
        move = price(s1-S1,s2-S2,SIG1,SIG2,CORR,T)
        assert isclose(loss,old-vh-move,abs_tol=1e-8)
        rows.append(dict(x1=x1,x2=x2,m1=m1,m2=m2,correlation_shift=dc,
                    s1=s1,s2=s2,sigma1=v1,sigma2=v2,correlation=c,
                    option_value=vh,portfolio_value=ph,loss=loss,weight=weight,
                    weighted_loss=weight*loss))
    assert len(rows)==3888
    worst=max(rows,key=lambda z:z['loss'])
    weighted=max(rows,key=lambda z:z['weighted_loss'])
    assert sum(isclose(z['loss'],worst['loss'],abs_tol=1e-8) for z in rows)==1
    assert sum(isclose(z['weighted_loss'],weighted['weighted_loss'],abs_tol=1e-8) for z in rows)==1
    # Direct finite differences independently check both hedge deltas.
    eps=0.001
    assert isclose(h1,(price(S1+eps,S2,SIG1,SIG2,CORR,T)-price(S1-eps,S2,SIG1,SIG2,CORR,T))/(2*eps),abs_tol=1e-7)
    assert isclose(h2,(price(S1,S2+eps,SIG1,SIG2,CORR,T)-price(S1,S2-eps,SIG1,SIG2,CORR,T))/(2*eps),abs_tol=1e-7)
    return dict(horizon=H,remaining_maturity=T-H,option0=v0,h1=h1,h2=h2,portfolio0=p0,
                count=len(rows),worst=worst,weighted=weighted),rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv')
    args=parser.parse_args()
    result,rows=calculate()
    print(json.dumps(result,indent=2))
    if args.csv:
        with open(args.csv,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

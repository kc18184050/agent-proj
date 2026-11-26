import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def optimize(store,t,recommendations,config,kind='mean_variance'):
    buys=[r for r in recommendations if r['label']=='Buy']
    keys=[r['id'] for r in buys]
    if not keys: return {'weights':{},'cash':1.0,'status':'no_buy_calls','kind':kind}
    hist=store.history(keys,t)
    # Eligibility from historical coverage only. Never inspect future prices or lifetime metadata.
    eligible=[k for k in keys if k in hist and hist[k].notna().sum()>=200]
    exclusions=[k for k in keys if k not in eligible]
    if not eligible:return {'weights':{},'cash':1.,'status':'insufficient_history','excluded':exclusions,'kind':kind}
    x=hist[eligible];x=x.fillna(x.mean())
    cov=LedoitWolf().fit(x.values).covariance_*252+np.eye(len(eligible))*1e-8
    mu_map={r['id']:r['expected_excess_21'] for r in buys}
    mu=np.array([mu_map[k]*252/config['horizon_sessions'] for k in eligible])
    n=len(eligible);cap=config['max_weight'];budget=min(1.,n*cap)
    w0=np.full(n,budget/n)
    if kind=='minimum_variance': objective=lambda w:float(w@cov@w)
    elif kind=='maximum_sharpe': objective=lambda w:float(-(w@mu)/np.sqrt(max(w@cov@w,1e-12)))
    else: objective=lambda w:float(.5*config['risk_aversion']*(w@cov@w)-w@mu)
    res=minimize(objective,w0,method='SLSQP',bounds=[(0,cap)]*n,
                 constraints=[{'type':'eq','fun':lambda w:w.sum()-budget}],
                 options={'ftol':1e-10,'maxiter':1500})
    if not res.success: raise ValueError('Portfolio optimization failed: '+res.message)
    w=res.x
    if abs(w.sum()-budget)>1e-6 or (w< -1e-7).any() or (w>cap+1e-6).any(): raise ValueError('Invalid weights')
    return {'weights':{k:float(v) for k,v in zip(eligible,w) if v>1e-10},'cash':float(1-budget),
            'status':'optimized','kind':kind,'excluded':exclusions,'objective':float(res.fun),
            'covariance_end':str(hist.index.max().date()),'covariance_observations':len(hist),
            'annualized_expected_excess':float(w@mu),'annualized_volatility':float(np.sqrt(w@cov@w))}

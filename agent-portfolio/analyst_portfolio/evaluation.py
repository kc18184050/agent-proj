from math import isclose

import numpy as np
import pandas as pd
from .data import clean

CALL_MODES = ('ridge', 'blind', 'with', 'ibes')


def classify_excess(excess, hold_band):
    # Subtracting the sector return can leave rounding noise at the boundary.
    if isclose(abs(excess), hold_band, rel_tol=0, abs_tol=1e-12):
        return 'Hold'
    if excess > hold_band:
        return 'Buy'
    if excess < -hold_band:
        return 'Sell'
    return 'Hold'


def score_calls(store, decisions, date_rows, config):
    records=[]
    for d in date_rows.itertuples():
        decision=decisions[str(d.decision_date.date())]
        maps={mode:{r['id']:r for r in decision[mode]} for mode in CALL_MODES if mode in decision}
        keys=list(next(iter(maps.values())))
        outcomes,_=store.forward_returns(keys,d.entry_date,d.exit_date)
        sector=outcomes.mean()
        for key in keys:
            ret=outcomes[key];excess=ret-sector
            analyst_available=maps.get('ibes',{}).get(key,{}).get('label') is not None
            for mode,lookup in maps.items():
                r=lookup[key];label=r.get('label')
                valid=pd.notna(excess) and label is not None
                target=classify_excess(excess,config['hold_band']) if pd.notna(excess) else None
                hit=(excess>0 if label=='Buy' else (excess<0 if label=='Sell' else abs(excess)<=config['hold_band'])) if valid else None
                raw_hit=(ret>0 if label=='Buy' else (ret<0 if label=='Sell' else abs(ret)<=config['hold_band'])) if valid else None
                records.append({'date':str(d.decision_date.date()),'entry_date':str(d.entry_date.date()),
                    'exit_date':str(d.exit_date.date()),'gvkey':key,'mode':mode,'label':label,
                    'return_21':ret,'sector_return_21':sector,'excess_21':excess,'outcome_label':target if valid else None,
                    'hit':hit,'raw_hit':raw_hit,'class_correct':label==target if valid else None,
                    'valid':valid,'analyst_available':analyst_available})
    return pd.DataFrame(records)

def call_metrics(calls):
    metrics={}
    for mode, group in calls.groupby('mode', sort=False):
        d=group[group.valid]
        directional=d[d.label!='Hold'];matched=d[d.analyst_available]
        metrics[mode]={'n':len(d),'directional_n':len(directional),'coverage':len(d)/len(group),
            'hit_rate':float(d.hit.astype(float).mean()),'directional_hit_rate':float(directional.hit.astype(float).mean()),
            'raw_hit_rate':float(d.raw_hit.astype(float).mean()),'classification_accuracy':float(d.class_correct.astype(float).mean()),
            'matched_n':len(matched),'matched_hit_rate':float(matched.hit.astype(float).mean()),
            'matched_classification_accuracy':float(matched.class_correct.astype(float).mean()),
            'labels':d.label.value_counts().to_dict()}
    return clean(metrics)


def paired_classification(calls):
    # Date-block bootstrap respects within-date cross-sectional dependence; only seven main dates.
    results={}
    valid=calls[calls.valid]
    columns=['date','gvkey','class_correct']
    for baseline in ['ridge','ibes']:
        comparator=valid.loc[valid['mode']==baseline,columns]
        for mode in ['blind','with']:
            agent=valid.loc[valid['mode']==mode,columns]
            pair=agent.merge(comparator,on=['date','gvkey'],suffixes=('_agent','_baseline'))
            if pair.empty:
                continue
            diff=pair.class_correct_agent.astype(float)-pair.class_correct_baseline.astype(float)
            blocks=diff.groupby(pair.date).agg(['sum','count'])
            rng=np.random.default_rng(2026)
            draws=rng.integers(0,len(blocks),(2000,len(blocks)))
            boot=blocks['sum'].to_numpy()[draws].sum(axis=1)/blocks['count'].to_numpy()[draws].sum(axis=1)
            results.setdefault(mode,{})[baseline]={
                'n':len(pair),'accuracy_difference':float(diff.mean()),
                'date_block_ci95':np.quantile(boot,[.025,.975]).tolist(),
                'wins':float((diff>0).mean()),'ties':float((diff==0).mean()),'losses':float((diff<0).mean())}
    return clean(results)

def backtest(store, decisions, date_rows, config):
    start=date_rows.entry_date.min();end=date_rows.exit_date.max()
    market=store.table('market_daily').set_index('date')
    returns=store.table('returns_daily').pivot(index='date',columns='gvkey',values='ret_total')
    sessions=store.table('sessions').date
    dates=sessions[(sessions>=start)&(sessions<=end)]
    entries={d.entry_date:decisions[str(d.decision_date.date())]['portfolios'] for d in date_rows.itertuples()}
    strategies=list(next(iter(entries.values())).keys());records=[]
    for strategy in strategies:
        holdings={};cash=1.;wealth=1.;broken=False
        for date in dates:
            rf=market.at[date,'rf_daily'] if date in market.index else np.nan
            cash_rate=float(rf) if pd.notna(rf) else 0.
            valid=True;gross=0.
            for key,w in holdings.items():
                ret=returns.at[date,key] if date in returns.index else np.nan
                if pd.isna(ret) and w>1e-10: valid=False
                else:gross+=w*float(ret)
            gross+=cash*cash_rate if date!=start else 0.
            cost=0.;turnover=0.
            if valid:
                grown={k:w*(1+float(returns.at[date,k]))/(1+gross) for k,w in holdings.items()}
                grown_cash=cash*(1+(cash_rate if date!=start else 0.))/(1+gross)
                if date in entries:
                    target=entries[date][strategy];weights=target['weights'];target_cash=target['cash']
                    # One-way traded notional including cash, starting from drifted positions.
                    turnover=.5*(sum(abs(weights.get(k,0)-grown.get(k,0)) for k in set(weights)|set(grown))+abs(target_cash-grown_cash))
                    # Per-side fee on traded STOCK notional: 10 bps per buy/sell dollar.
                    cost=config['transaction_cost_bps']/10000*sum(abs(weights.get(k,0)-grown.get(k,0)) for k in set(weights)|set(grown))
                    holdings,cash=weights.copy(),target_cash
                else:holdings,cash=grown,grown_cash
                net=(1+gross)*(1-cost)-1
                wealth*=1+net
            else:
                broken=True;net=np.nan;gross=np.nan;wealth=np.nan
                # Missing realizations are never silently replaced by zero or removed.
            # Liquidate at final horizon close, charging the sell-side fee.
            if date==end and valid:
                sell_cost=config['transaction_cost_bps']/10000*sum(holdings.values())
                net=(1+net)*(1-sell_cost)-1;wealth*=1-sell_cost
                cost+=sell_cost;turnover+=sum(holdings.values());holdings={};cash=1.
            records.append({'date':str(date.date()),'strategy':strategy,'gross_return':gross,'net_return':net,
                'rf_daily':rf,'cash_rate_zero_proxy':pd.isna(rf),'turnover':turnover,'cost_fraction':cost,
                'wealth':wealth,'valid_valuation':valid and not broken})
    return pd.DataFrame(records)

def portfolio_metrics(daily):
    result={}
    for name,d in daily.groupby('strategy'):
        valid=d.valid_valuation.all();r=d.net_return.astype(float)
        rf=d.rf_daily.astype(float);covered=r[rf.notna()]-rf.dropna()
        proxy=r-rf.fillna(0)
        def sharpe(x):return float(np.sqrt(252)*x.mean()/x.std()) if len(x)>1 and x.std()>0 else None
        wealth=d.wealth.astype(float)
        # Include initial wealth=1 so a first-day cost counts toward drawdown.
        peak=wealth.cummax().clip(lower=1)
        result[name]={'sessions':len(d),'valid':bool(valid),'total_return':float(wealth.iloc[-1]-1) if valid else None,
            'annualized_volatility':float(r.std()*np.sqrt(252)) if valid else None,
            'sharpe_rf_covered':sharpe(covered) if valid else None,'rf_covered_sessions':int(rf.notna().sum()),
            'sharpe_zero_rf_proxy':sharpe(proxy) if valid else None,
            'max_drawdown':float((wealth/peak-1).min()) if valid else None,
            'total_cost_fraction_sum':float(d.cost_fraction.sum()),'turnover_sum':float(d.turnover.sum()),
            'missing_rf_sessions':int(rf.isna().sum())}
    return clean(result)

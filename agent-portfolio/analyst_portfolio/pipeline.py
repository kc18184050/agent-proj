from pathlib import Path
import json
import os
import pandas as pd
from dotenv import load_dotenv
from .data import DataStore, clean, save
from .training import load_frozen, predict
from .agents import AgentRunner, specialists, manager
from .portfolio import optimize
from .evaluation import classify_excess, score_calls, call_metrics, paired_classification, backtest, portfolio_metrics

def setup(root):
    root=Path(root).resolve();load_dotenv(root/'.env',override=False)
    config=json.loads((root/'config.json').read_text())
    if config['model']!='gpt-5.6-luna':raise ValueError('This experiment is frozen to the user-requested model')
    return root,config

def prepare(root,date,role='core',explain=False,ticker=None,rating=None,*,include_analysts=True):
    root,config=setup(root);model=load_frozen(root,config)
    t=pd.Timestamp(date)
    if t<=pd.Timestamp(config['cutoff']):raise ValueError('This flow is for post-cutoff testing; use CLI train for open data')
    store=DataStore(config['full_dir'],config)
    dd=store.table('decision_dates');dd=dd[dd.decision_date==t]
    if dd.empty:raise ValueError('Choose a decision_date listed in the supplied data')
    price_cutoff=dd.iloc[0].prev_session if explain else t
    universe=store.universe(role,t)
    if ticker:
        universe=universe[(universe.ticker_symbol==ticker.upper())|(universe.gvkey==ticker)]
        if universe.empty:raise ValueError('Ticker/gvkey is not eligible in the requested sector/date')
    if rating and not explain:raise ValueError('A supplied rating is allowed only in explanation mode')
    if rating and rating not in ['Strong Buy','Buy','Hold','Sell','Strong Sell']:raise ValueError('Invalid five-level rating')
    keys=universe.gvkey.tolist()
    packs=[store.pack(k,t,price_cutoff) for k in keys]
    analysts=[store.analyst(k,t) for k in keys] if include_analysts or rating else []
    if rating:
        if len(keys)!=1:raise ValueError('Select one ticker when explaining a supplied rating')
        analysts[0]['consensus']={'label':rating.replace('Strong ',''),'supplied_rating_5':rating,
            'source':'user-supplied hypothetical or historical rating; not verified as an actual broker action'}
    return {'root':str(root),'date':str(t.date()),'role':role,'explain':explain,
            'packs':packs,'analysts':analysts,'freeze_sha256':model['freeze_sha256']}

def specialist_stage(payload,role):
    root,config=setup(payload['root']);runner=AgentRunner(config,root)
    return {'role':role,'reports':specialists(payload['packs'],role,runner)}

def ridge_recommendations(packs, predictions, hold_band):
    return [
        {'id': pack['id'], 'label': classify_excess(mu, hold_band), 'expected_excess_21': float(mu)}
        for pack, mu in zip(packs, predictions)
    ]

def manager_stage(payload,reports):
    root,config=setup(payload['root']);model=load_frozen(root,config);runner=AgentRunner(config,root)
    packs=payload['packs']
    blind=manager(packs,reports,model,runner)
    with_=manager(packs,reports,model,runner,payload['analysts'],blind)
    prior=predict(model,[p['features'] for p in packs]);ibes=[]
    for p,a,mu in zip(packs,payload['analysts'],prior):
        c=a['consensus'];label=c.get('label');score={'Buy':1,'Hold':0,'Sell':-1}.get(label,0)
        ibes.append({'id':p['id'],'label':label,'expected_excess_21':float(.5*mu+.0125*score)})
    ridge=ridge_recommendations(packs,prior,config['hold_band'])
    return {**payload,'specialists':reports,'blind':blind,'with':with_,'ibes':ibes,'ridge':ridge}

def portfolio_stage(payload):
    if payload.get('explain'):return {**payload,'portfolios':{},'note':'Explanation-only mode: prior-session prices; not part of recommendation backtest.'}
    root,config=setup(payload['root']);store=DataStore(config['full_dir'],config);t=pd.Timestamp(payload['date'])
    portfolios={}
    for mode in ['blind','with','ibes']:
        portfolios[f'{mode}_mean_variance']=optimize(store,t,payload[mode],config)
    portfolios['ridge_mean_variance']=optimize(store,t,payload['ridge'],config)
    for kind in ['minimum_variance','maximum_sharpe']:
        portfolios[f'with_{kind}']=optimize(store,t,payload['with'],config,kind)
    keys=[p['id'] for p in payload['packs']]
    portfolios['equal_weight']={'weights':{k:1/len(keys) for k in keys},'cash':0.,'kind':'equal_weight','status':'benchmark'}
    result={**payload,'portfolios':portfolios}
    save(root/f'artifacts/decisions/{payload["role"]}/{payload["date"]}.json',result)
    return result

def run_date(root,date,role='core',explain=False,ticker=None,rating=None):
    payload=prepare(root,date,role,explain,ticker,rating)
    # Reuse the same role functions from CLI and actual Langflow nodes.
    reports={r:specialist_stage(payload,r)['reports'] for r in ['fundamental','technical','news']}
    return portfolio_stage(manager_stage(payload,reports))

def evaluate(root,role='core',conditional=False,*,ridge_only=False):
    root,config=setup(root);model=load_frozen(root,config);store=DataStore(config['full_dir'],config)
    dates=store.dates('test_conditional' if conditional else 'test')
    if dates.empty:raise ValueError('No decision dates for the requested test split')
    if not dates.exit_in_data.all():raise ValueError('A test horizon is not fully observed')
    decisions={}
    for d in dates.itertuples():
        date=str(d.decision_date.date());print(f'Running {role} {date}',flush=True)
        if ridge_only:
            packs=prepare(root,date,role,include_analysts=False)['packs']
            prior=predict(model,[p['features'] for p in packs])
            ridge=ridge_recommendations(packs,prior,config['hold_band'])
            decisions[date]={'ridge':ridge,'portfolios':{
                'ridge_mean_variance':optimize(store,d.decision_date,ridge,config)}}
        else:
            # Caching is at the API request level, tied to prompts, evidence and exact model.
            decisions[date]=run_date(root,date,role)
    calls=score_calls(store,decisions,dates,config);daily=backtest(store,decisions,dates,config)
    split='conditional' if conditional else 'main'
    suffix='_ridge_only' if ridge_only else ''
    out=root/'artifacts/results'/f'{role}_{split}{suffix}';out.mkdir(parents=True,exist_ok=True)
    calls.to_csv(out/'call_outcomes.csv',index=False);daily.to_csv(out/'daily_portfolios.csv',index=False)
    band=f'{config["hold_band"]:.2%}'
    metrics={'scope':role,'conditional':conditional,'ridge_only':ridge_only,'dates':dates.decision_date.dt.strftime('%Y-%m-%d').tolist(),
        'freeze_sha256':model['freeze_sha256'],'calls':call_metrics(calls),'portfolios':portfolio_metrics(daily),
        'paired_classification':paired_classification(calls),
        'methodology':{'primary_accuracy':f'Buy: sector-relative return > {band}; Sell: < -{band}; Hold: inclusive band',
        'directional_hit':'Buy: sector-relative return >0; Sell: <0; Hold excluded',
        'ridge_comparator':'Same frozen Ridge predictions for ratings and expected returns; same optimizer as the agent',
        'sector_benchmark':'equal-weight same-date eligible sector; missing 21-session outcomes explicitly excluded from call scoring',
        'call_horizon':'21 sessions after entry close; entry is first session after decision date',
        'portfolio_schedule':'Rebalance at next decision entry close; last period liquidated at final 21-session exit. No double-counted overlapping horizons.',
        'primary_sharpe':'Annualized daily excess return / daily excess-return SD, restricted to reported risk-free dates',
        'full_period_proxy':'Zero cash rate on missing rf dates; explicitly labeled proxy',
        'ibes_comparator':'IBES Buy selection + same optimizer and open-trained numeric prior; fixed +1 analyst conviction',
        'survivorship':'Fixed 2026-03-01 universe supplied by user, not a historical constituent universe',
        'pit_limitation':'Current-vintage Compustat ratios and prebuilt consensus; source-vintage PIT cannot be proven from these files',
        'small_sample':'7 decision-date clusters; daily Sharpe and block-bootstrap intervals are descriptive, not proof of superiority'}}
    save(out/'metrics.json',metrics)
    return metrics

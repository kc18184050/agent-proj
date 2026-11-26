import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from analyst_portfolio.data import DataStore
from analyst_portfolio.training import load_frozen
from analyst_portfolio.agents import specialist_inputs
from analyst_portfolio.portfolio import optimize
from analyst_portfolio.evaluation import backtest
from analyst_portfolio.pipeline import prepare

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config.json').read_text())

def test_future_rows_do_not_change_pack():
    s=DataStore(CONFIG['full_dir'],CONFIG);t=pd.Timestamp('2026-03-02');k=s.universe('core',t).gvkey.iloc[0]
    before=s.pack(k,t)
    for name,col in [('returns_daily','date'),('valuation_daily','date'),('fundamentals_q','available_date'),('news_events','usable_date')]:
        frame=s.table(name).copy();future=frame[col]>t
        for c in frame.select_dtypes(include='number').columns:frame.loc[future,c]=99999
        s.tables[name]=frame
    assert s.pack(k,t)==before

def test_same_day_releases_are_not_visible():
    s=DataStore(CONFIG['full_dir'],CONFIG);t=pd.Timestamp('2026-03-02');k=s.universe('core',t).gvkey.iloc[0]
    f=s.table('fundamentals_q').iloc[[0]].copy();f['gvkey']=k;f['available_date']=t;f['rev_yoy']=987.
    s.tables['fundamentals_q']=pd.concat([s.table('fundamentals_q'),f])
    assert s.pack(k,t)['fundamental'].get('rev_yoy')!=987.

def test_blind_payload_excludes_analyst_inputs():
    s=DataStore(CONFIG['full_dir'],CONFIG);t=pd.Timestamp('2026-03-02');k=s.universe('core',t).gvkey.iloc[0]
    pack=s.pack(k,t)
    for role in ['fundamental','technical','news']:
        text=json.dumps(specialist_inputs([pack],role))
        assert all(term not in text for term in ['mean_rating','price_target','buy_share','eps_revisions'])

def test_training_only_open_and_pre_cutoff():
    m=load_frozen(ROOT,CONFIG)
    assert m['max_label_date']<=CONFIG['cutoff']
    assert max(m['training_dates'])<=CONFIG['cutoff']
    assert m['frozen_config']['open_dir'].endswith('/open')

def test_optimizer_constraints_and_no_future_access():
    class Store:
        def history(self,keys,t):
            x=np.random.default_rng(2).normal(0,.01,(252,len(keys)))
            return pd.DataFrame(x,index=pd.bdate_range(end=t,periods=252),columns=keys)
    rec=[{'id':str(i),'label':'Buy' if i<5 else 'Sell','expected_excess_21':.02} for i in range(12)]
    p=optimize(Store(),pd.Timestamp('2026-03-02'),rec,CONFIG)
    assert set(p['weights'])<=set(str(i) for i in range(5))
    assert max(p['weights'].values())<=.100001
    assert sum(p['weights'].values())+p['cash']==pytest.approx(1)
    assert p['cash']==pytest.approx(.5)

def test_entry_day_returns_excluded_and_costs_charged():
    dates=pd.to_datetime(['2026-03-03','2026-03-04','2026-03-05'])
    tables={'sessions':pd.DataFrame({'date':dates}),
       'market_daily':pd.DataFrame({'date':dates,'rf_daily':[0.,0.,0.]}),
       'returns_daily':pd.DataFrame({'date':dates,'gvkey':['A']*3,'ret_total':[9.,.1,.2]})}
    class Store:
        def table(self,name):return tables[name]
    decision={'2026-03-02':{'portfolios':{'example':{'weights':{'A':1.},'cash':0.}}}}
    dd=pd.DataFrame({'decision_date':pd.to_datetime(['2026-03-02']),'entry_date':[dates[0]],'exit_date':[dates[-1]]})
    out=backtest(Store(),decision,dd,CONFIG)
    fee=CONFIG['transaction_cost_bps']/10000
    assert out.wealth.iloc[-1]==pytest.approx((1-fee)*1.1*1.2*(1-fee))

def test_missing_held_returns_invalidate_backtest():
    dates=pd.to_datetime(['2026-03-03','2026-03-04'])
    tables={'sessions':pd.DataFrame({'date':dates}),
       'market_daily':pd.DataFrame({'date':dates,'rf_daily':[0.,0.]}),
       'returns_daily':pd.DataFrame({'date':dates,'gvkey':['A']*2,'ret_total':[0.,np.nan]})}
    class Store:
        def table(self,name):return tables[name]
    decisions={'2026-03-02':{'portfolios':{'example':{'weights':{'A':1.},'cash':0.}}}}
    dd=pd.DataFrame({'decision_date':pd.to_datetime(['2026-03-02']),'entry_date':[dates[0]],'exit_date':[dates[-1]]})
    out=backtest(Store(),decisions,dd,CONFIG)
    assert not out.valid_valuation.iloc[-1]
    assert pd.isna(out.wealth.iloc[-1])

def test_explanation_supports_five_level_rating_and_prior_prices():
    p=prepare(ROOT,'2026-03-02',explain=True,ticker='AAPL',rating='Strong Buy')
    assert len(p['packs'])==1
    assert p['packs'][0]['price_cutoff']=='2026-02-27'
    assert p['analysts'][0]['consensus']['supplied_rating_5']=='Strong Buy'

def test_overlapping_call_horizons_are_not_double_counted_in_portfolio():
    dates=pd.bdate_range('2026-03-03',periods=4)
    tables={'sessions':pd.DataFrame({'date':dates}),
       'market_daily':pd.DataFrame({'date':dates,'rf_daily':np.zeros(4)}),
       'returns_daily':pd.DataFrame({'date':list(dates)*2,'gvkey':['A']*4+['B']*4,'ret_total':[0,.1,.1,.1,0,.2,.2,.2]})}
    class Store:
        def table(self,name):return tables[name]
    decisions={'2026-03-02':{'portfolios':{'example':{'weights':{'A':1.},'cash':0.}}},
               '2026-03-04':{'portfolios':{'example':{'weights':{'B':1.},'cash':0.}}}}
    dd=pd.DataFrame({'decision_date':pd.to_datetime(['2026-03-02','2026-03-04']),
                     'entry_date':[dates[0],dates[2]],'exit_date':[dates[3],dates[3]]})
    config={**CONFIG,'transaction_cost_bps':0}
    out=backtest(Store(),decisions,dd,config)
    assert out.wealth.iloc[-1]==pytest.approx(1.1*1.1*1.2)
    assert len(out)==4

import json

import numpy as np
import pandas as pd
import pytest

from analyst_portfolio import pipeline
from analyst_portfolio.evaluation import call_metrics, classify_excess, paired_classification, score_calls


@pytest.mark.parametrize(('excess', 'label'), [
    (-.02, 'Sell'), (-.01, 'Hold'), (0., 'Hold'), (.01, 'Hold'), (.04 - .03, 'Hold'), (.02, 'Buy'),
])
def test_inclusive_hold_band(excess, label):
    assert classify_excess(excess, .01) == label


def test_ridge_and_agent_use_the_same_outcomes():
    class Store:
        def forward_returns(self, keys, entry, exit):
            return pd.Series({'A': .005, 'B': -.005}), None

    date = pd.Timestamp('2026-03-02')
    dates = pd.DataFrame({'decision_date': [date], 'entry_date': [date], 'exit_date': [date]})
    decisions = {'2026-03-02': {
        'ridge': [{'id': 'A', 'label': 'Hold'}, {'id': 'B', 'label': 'Hold'}],
        'blind': [{'id': 'A', 'label': 'Buy'}, {'id': 'B', 'label': 'Sell'}],
        'ibes': [{'id': 'A', 'label': 'Buy'}, {'id': 'B', 'label': None}],
    }}
    calls = score_calls(Store(), decisions, dates, {'hold_band': .01})
    metrics = call_metrics(calls)
    assert metrics['ridge']['classification_accuracy'] == 1.
    assert metrics['blind']['classification_accuracy'] == 0.
    assert metrics['blind']['directional_hit_rate'] == 1.
    assert metrics['ridge']['matched_n'] == metrics['ibes']['matched_n'] == 1
    assert paired_classification(calls)['blind']['ridge']['accuracy_difference'] == -1.


def test_ridge_only_evaluation_uses_no_agents_and_writes_separate_results(tmp_path, monkeypatch):
    decision, entry, exit = pd.to_datetime(['2026-03-02', '2026-03-03', '2026-03-04'])
    dates = pd.DataFrame({'decision_date': [decision], 'entry_date': [entry], 'exit_date': [exit], 'exit_in_data': [True]})
    config = {'full_dir': 'full', 'hold_band': .01, 'horizon_sessions': 21,
              'max_weight': .1, 'risk_aversion': 5., 'transaction_cost_bps': 10}

    class Store:
        def __init__(self, root, config):
            pass

        def dates(self, split):
            assert split == 'test'
            return dates

        def forward_returns(self, keys, entry, exit):
            return pd.Series({'A': .02, 'B': -.02}), None

        def history(self, keys, date):
            assert keys == ['A']
            return pd.DataFrame({'A': np.tile([-.01, .01], 126)}, index=pd.bdate_range(end=date, periods=252))

        def table(self, name):
            return {
                'market_daily': pd.DataFrame({'date': [entry, exit], 'rf_daily': [0., 0.]}),
                'returns_daily': pd.DataFrame({'date': [entry, exit], 'gvkey': ['A', 'A'], 'ret_total': [0., .02]}),
                'sessions': pd.DataFrame({'date': [entry, exit]}),
            }[name]

    def prepare(root, date, role, *, include_analysts):
        assert not include_analysts
        return {'packs': [{'id': 'A', 'features': {}}, {'id': 'B', 'features': {}}]}

    def unexpected_agent(*args, **kwargs):
        pytest.fail('Ridge-only must not call any Agent')

    monkeypatch.setattr(pipeline, 'setup', lambda root: (tmp_path, config))
    monkeypatch.setattr(pipeline, 'load_frozen', lambda root, config: {'freeze_sha256': 'test'})
    monkeypatch.setattr(pipeline, 'prepare', prepare)
    monkeypatch.setattr(pipeline, 'DataStore', Store)
    monkeypatch.setattr(pipeline, 'predict', lambda model, features: np.array([.02, -.02]))
    monkeypatch.setattr(pipeline, 'run_date', unexpected_agent)
    monkeypatch.setattr(pipeline, 'AgentRunner', unexpected_agent)

    metrics = pipeline.evaluate(tmp_path, ridge_only=True)
    assert list(metrics['calls']) == ['ridge']
    assert metrics['calls']['ridge']['classification_accuracy'] == 1.
    assert metrics['portfolios']['ridge_mean_variance']['valid']
    output = tmp_path / 'artifacts/results/core_main_ridge_only'
    assert json.loads((output / 'metrics.json').read_text()) == metrics
    assert (output / 'call_outcomes.csv').is_file()
    assert (output / 'daily_portfolios.csv').is_file()

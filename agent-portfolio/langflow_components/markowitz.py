from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output
from lfx.schema.data import Data
import sys

class MarkowitzPortfolio(Component):
    display_name = 'Buy Calls → Markowitz'
    description = 'Buy recommendations feed long-only capped mean-variance portfolios; also minimum-variance, maximum-Sharpe and equal-weight controls.'
    icon = 'ChartPie'
    name = 'MarkowitzPortfolio'
    inputs = [HandleInput(name='recommendations',display_name='Recommendations',input_types=['Data'],required=True)]
    outputs = [Output(name='portfolio',display_name='Portfolio and audit',method='construct')]

    def construct(self) -> Data:
        p=self.recommendations.data
        if p['root'] not in sys.path:sys.path.insert(0,p['root'])
        from analyst_portfolio.pipeline import portfolio_stage
        return Data(data=portfolio_stage(p))

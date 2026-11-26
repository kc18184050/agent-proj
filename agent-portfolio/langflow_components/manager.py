from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output
from lfx.schema.data import Data
import sys

class AnalystManager(Component):
    display_name = 'Independent + IBES Manager'
    description = 'First make independent recommendations, then explain and confirm/upgrade/downgrade IBES consensus. Separate API calls prevent rating leakage.'
    icon = 'Users'
    name = 'AnalystManager'
    inputs = [HandleInput(name=n,display_name=title,input_types=['Data'],required=True)
              for n,title in [('payload','Evidence packs'),('fundamental','Fundamental report'),
                              ('technical','Technical report'),('news','News report')]]
    outputs = [Output(name='recommendations',display_name='Recommendations',method='recommend')]

    def recommend(self) -> Data:
        p=self.payload.data
        if p['root'] not in sys.path:sys.path.insert(0,p['root'])
        from analyst_portfolio.pipeline import manager_stage
        reports={r:getattr(self,r).data['reports'] for r in ['fundamental','technical','news']}
        return Data(data=manager_stage(p,reports))

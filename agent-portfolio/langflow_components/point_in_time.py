import json
import sys
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, StrInput, Output
from lfx.schema.data import Data

class PointInTimePack(Component):
    display_name = 'Point-in-Time Evidence'
    description = 'Read dated evidence only. Requires open-only frozen calibration. JSON request: date, role, explain.'
    icon = 'CalendarCheck'
    name = 'PointInTimePack'
    inputs = [MessageTextInput(name='request',display_name='Request JSON',required=True,
                value='{"date":"2026-03-02","role":"core","explain":false}'),
              StrInput(name='project_root',display_name='Project directory',required=True)]
    outputs = [Output(name='payload',display_name='Evidence packs',method='build_payload')]

    def build_payload(self) -> Data:
        if self.project_root not in sys.path:sys.path.insert(0,self.project_root)
        from analyst_portfolio.pipeline import prepare
        req=json.loads(self.request)
        return Data(data=prepare(self.project_root,req['date'],req.get('role','core'),req.get('explain',False),req.get('ticker'),req.get('rating')))

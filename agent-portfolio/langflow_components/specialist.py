from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, DropdownInput, Output
from lfx.schema.data import Data
import sys

class AnalystSpecialist(Component):
    display_name = 'GPT-5.6 Luna Specialist'
    description = 'One distinct specialist role; blind to analyst-derived inputs. Responses API with validated evidence citations.'
    icon = 'Brain'
    name = 'AnalystSpecialist'
    inputs = [HandleInput(name='payload',display_name='Evidence packs',input_types=['Data'],required=True),
              DropdownInput(name='role',display_name='Specialist role',options=['fundamental','technical','news'],value='fundamental')]
    outputs = [Output(name='report',display_name='Specialist report',method='analyze')]

    def analyze(self) -> Data:
        data=self.payload.data
        if data['root'] not in sys.path:sys.path.insert(0,data['root'])
        from analyst_portfolio.pipeline import specialist_stage
        return Data(data=specialist_stage(data,self.role))

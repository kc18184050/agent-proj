from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output
from lfx.schema.message import Message
import json
from pathlib import Path

class ResearchOutput(Component):
    display_name = 'Research Output'
    description = 'Show recommendation explanations, evidence, weights, frozen policy hash and saved output. CLI evaluates all seven dates.'
    icon = 'FileText'
    name = 'ResearchOutput'
    inputs = [HandleInput(name='portfolio',display_name='Portfolio and audit',input_types=['Data'],required=True)]
    outputs = [Output(name='message',display_name='Research results',method='render')]

    def render(self) -> Message:
        p=self.portfolio.data
        root=Path(p['root'])
        import pandas as pd
        u=pd.read_parquet(json.loads((root/'config.json').read_text())['full_dir']+'/universe.parquet').set_index('gvkey')
        lines=[f"## {p['date']} · {p['role']} · GPT-5.6 Luna",f"Frozen calibration: `{p['freeze_sha256'][:16]}`",
               'Buy recommendations feed Markowitz. Research output; 21-session call horizon.',
               '|Stock|Independent|IBES|With IBES|Action|','|---|---|---|---|---|']
        for b,w,a in zip(p['blind'],p['with'],p['ibes']):
            name=u.at[b['id'],'ticker_symbol']
            lines.append(f"|{name}|{b['rating_5']}|{a['label'] or 'unavailable'}|{w['rating_5']}|{w['action']}|")
        for name,obj in p['portfolios'].items():
            lines.append(f"\n### {name} (cash {obj['cash']:.1%})")
            for k,v in sorted(obj['weights'].items(),key=lambda kv:-kv[1]):lines.append(f"- {u.at[k,'ticker_symbol']}: {v:.2%}")
        lines.append('\n### Explanations with analyst context')
        for r in p['with']:
            lines.append(f"\n**{u.at[r['id'],'ticker_symbol']} — {r['rating_5']}**: {r['rationale']}\nEvidence: {', '.join(r['evidence_ids'])}")
        lines.append('\nRun `analyst-portfolio evaluate` to compute all-date hit rates and OOS portfolio Sharpe.')
        return Message(text='\n'.join(lines))

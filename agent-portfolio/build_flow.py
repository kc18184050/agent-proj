"""Build a real Langflow export using installed component-generated templates."""
import copy
import importlib.util
import json
from pathlib import Path
import uuid
import sys

ROOT=Path(__file__).resolve().parent

def build():
    from lfx.components.input_output.chat import ChatInput
    from lfx.components.input_output.chat_output import ChatOutput
    nodes=[];edges=[]
    def node(cls,key,x,y,values=None):
        obj=cls();frontend=obj.to_frontend_node()['data']['node']
        # API versions may return (template, instance) in async builder helpers; this public method returns dict.
        if not isinstance(frontend,dict):raise TypeError(type(frontend))
        for k,v in (values or {}).items():frontend['template'][k]['value']=v
        nid=f'{cls.__name__}-{key}'
        nodes.append({'id':nid,'type':'genericNode','position':{'x':x,'y':y},
            'data':{'id':nid,'type':cls.__name__,'node':frontend,'display_name':frontend['display_name']}})
        return nid
    def custom(filename,name):
        spec=importlib.util.spec_from_file_location('component_'+filename,ROOT/'langflow_components'/f'{filename}.py')
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return getattr(mod,name)
    ci=node(ChatInput,'request',0,400,{'input_value':'{"date":"2026-03-02","role":"core","explain":false}','should_store_message':False})
    pit=node(custom('point_in_time','PointInTimePack'),'packs',360,400,{'project_root':str(ROOT)})
    f=node(custom('specialist','AnalystSpecialist'),'fundamental',760,0,{'role':'fundamental'})
    t=node(custom('specialist','AnalystSpecialist'),'technical',760,420,{'role':'technical'})
    n=node(custom('specialist','AnalystSpecialist'),'news',760,840,{'role':'news'})
    m=node(custom('manager','AnalystManager'),'manager',1160,400)
    port=node(custom('markowitz','MarkowitzPortfolio'),'portfolio',1570,400)
    report=node(custom('report_output','ResearchOutput'),'research',1970,400)
    co=node(ChatOutput,'output',2370,400,{'should_store_message':False})
    def edge(source,target,output,field,typ='Data'):
        st=next(n['data']['type'] for n in nodes if n['id']==source)
        sh={'dataType':st,'id':source,'name':output,'output_types':[typ]}
        th={'fieldName':field,'id':target,'inputTypes':[typ],'type':'str' if typ=='Message' else 'other'}
        enc=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':')).replace('"','œ')
        edges.append({'id':f'edge-{len(edges)}','source':source,'target':target,'sourceHandle':enc(sh),'targetHandle':enc(th),
            'data':{'sourceHandle':sh,'targetHandle':th},'animated':False,'className':''})
    edge(ci,pit,'message','request','Message')
    for dest in [f,t,n,m]:edge(pit,dest,'payload','payload')
    for src,field in [(f,'fundamental'),(t,'technical'),(n,'news')]:edge(src,m,'report',field)
    edge(m,port,'recommendations','recommendations');edge(port,report,'portfolio','portfolio')
    edge(report,co,'message','input_value','Message')
    flow={'id':str(uuid.uuid5(uuid.NAMESPACE_URL,'analyst-portfolio-v1')),'name':'Analyst Recommendations → Portfolio',
        'description':'Open-only frozen calibration. Three GPT-5.6 Luna specialists → independent manager → IBES explanation and adjustment → Markowitz portfolio. Post-2026-02-16 testing.',
        'data':{'nodes':nodes,'edges':edges,'viewport':{'x':30,'y':30,'zoom':.45}},
        'is_component':False,'last_tested_version':'1.12.0','tags':['finance','multi-agent','point-in-time'],'endpoint_name':'analyst-portfolio'}
    path=ROOT/'Analyst_Portfolio_Langflow.json';path.write_text(json.dumps(flow,ensure_ascii=False,indent=2));return path

if __name__=='__main__':print(build())

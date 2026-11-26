from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .data import DataStore, FEATURES, save, dumps

def transform(rows, medians=None, center=None, scale=None):
    x = np.array([[r.get(f) if r.get(f) is not None else np.nan for f in FEATURES] for r in rows],dtype=float)
    x[~np.isfinite(x)] = np.nan
    if medians is None:
        medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in x.T])
    x = np.where(np.isnan(x), medians, x)
    # Fixed clipping prevents anomalous ratio denominators dominating the fit.
    x = np.clip(x,-10,10)
    if center is None: center = x.mean(axis=0)
    if scale is None: scale = np.maximum(x.std(axis=0),1e-6)
    return np.clip((x-center)/scale,-5,5), np.asarray(medians),np.asarray(center),np.asarray(scale)

def predict(model, features):
    x,*_ = transform(features,model['medians'],model['center'],model['scale'])
    return np.clip(x@np.array(model['coef'])+model['intercept'],-.15,.15)

def train(config, root):
    store = DataStore(config['open_dir'],config,development=True)
    records=[]
    for d in store.dates('dev').itertuples():
        keys=store.universe('core',d.decision_date).gvkey.tolist()
        targets,_=store.forward_returns(keys,d.entry_date,d.exit_date)
        sector=float(targets.mean())
        for key in keys:
            if pd.notna(targets[key]):
                pack=store.pack(key,d.decision_date)
                records.append({'date':str(d.decision_date.date()),'exit_date':str(d.exit_date.date()),
                    'gvkey':key,'features':pack['features'],'target':float(targets[key]-sector)})
    # Purged chronological holdout: training labels must mature before validation starts.
    dates=sorted({r['date'] for r in records}); val_start=dates[-3]
    tr=[r for r in records if r['exit_date']<val_start]
    va=[r for r in records if r['date']>=val_start]
    x,med,center,scale=transform([r['features'] for r in tr]); y=np.array([r['target'] for r in tr])
    vx,*_=transform([r['features'] for r in va],med,center,scale)
    scores=[]
    for alpha in [10,100,1000]:
        reg=Ridge(alpha=alpha).fit(x,y)
        mse=float(np.mean((reg.predict(vx)-[r['target'] for r in va])**2))
        scores.append({'alpha':alpha,'validation_mse':mse})
    alpha=min(scores,key=lambda r:r['validation_mse'])['alpha']
    x,med,center,scale=transform([r['features'] for r in records])
    reg=Ridge(alpha=alpha).fit(x,[r['target'] for r in records])
    model={'schema_version':1,'cutoff':config['cutoff'],'model':config['model'],'features':FEATURES,
           'medians':med.tolist(),'center':center.tolist(),'scale':scale.tolist(),'coef':reg.coef_.tolist(),
           'intercept':float(reg.intercept_),'alpha':alpha,'validation':scores,'rows':len(records),
           'training_dates':dates,'validation_start':val_start,'purged_training_rows':len(tr),
           'max_label_date':max(r['exit_date'] for r in records),'source_hashes':store.file_hashes(),
           'frozen_config':config,'note':'Open-only numeric calibration; GPT weights unchanged. Development LLM memory is not OOS evidence.'}
    model['freeze_sha256']=hashlib.sha256(dumps(model).encode()).hexdigest()
    save(Path(root)/'artifacts/frozen_model.json',model)
    save(Path(root)/'artifacts/training_manifest.json',{'rows':[{k:v for k,v in r.items() if k!='features'} for r in records],
         'freeze_sha256':model['freeze_sha256'],'validation':scores})
    return model

def load_frozen(root, config):
    path=Path(root)/'artifacts/frozen_model.json'
    if not path.exists(): raise ValueError('Run train with open data before test; test never trains automatically')
    model=json.loads(path.read_text()); sha=model['freeze_sha256']
    if hashlib.sha256(dumps({k:v for k,v in model.items() if k!='freeze_sha256'}).encode()).hexdigest()!=sha:
        raise ValueError('Frozen training artifact hash mismatch')
    if model['frozen_config']!=config: raise ValueError('Config changed since training; freeze a new experiment explicitly')
    return model

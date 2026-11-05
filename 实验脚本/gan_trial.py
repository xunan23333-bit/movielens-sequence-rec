import os, time, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

root=Path(__file__).resolve().parent.parent; os.chdir(root)
OUT='out'; SEED=42; dev='cuda' if torch.cuda.is_available() else 'cpu'
nb=json.loads((Path(__file__).resolve().parent/'搜索基线.ipynb').read_text(encoding='utf8'))
trials=json.loads(Path('out/validation_sweep.json').read_text(encoding='utf8'))
best=max((t for t in trials if t['config']['features']=='both'),key=lambda t:(t['valid']['NDCG10'],-t['config']['dim'],-t['config']['n_neg']))
seq_len=best['config'].get('seq_len',20)
ratings=pd.read_csv('ratings.csv'); movies=pd.read_csv('movies.csv')
for i in (8,9,10,12,13,15,16,17,18,26):
    source=''.join(nb['cells'][i]['source'])
    if i==8: source=source.replace('L, N_TEST, N_VALID, N_NEG = 20, 5, 5, 100',f'L, N_TEST, N_VALID, N_NEG = {seq_len}, 5, 5, 100')
    exec(source,globals())
torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
exec(''.join(nb['cells'][27]['source']),globals())
original=(tr_u,tr_x,tr_r,tr_y,tr_neg)
results=[]
for threshold in (3.5,4.0):
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    rng=np.random.default_rng(SEED)
    source=''.join(nb['cells'][28]['source']).replace('fu[k] >= 4.0',f'fu[k] >= {threshold}')
    exec(source,globals()); plt.close('all')
    tr_u,tr_x,tr_r,tr_y=build_samples(ns,nr)
    gan_seen=train_seen.copy()
    for u,seq in ns.items(): gan_seen[u,seq]=True
    tr_neg=np.array([rng.choice(np.flatnonzero(~gan_seen[u,1:])+1) for u in tr_u])
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    m=SASRecLite().to(dev); start=time.time()
    hist=train_model(m,epochs=30,tag=f'gan threshold {threshold}')
    valid=evaluate(m,valid_cand)
    result=dict(threshold=threshold,min_len=60,added=added,valid=valid,seconds=round(time.time()-start,1))
    results.append(result)
    Path('out/gan_validation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print('GAN validation',result,flush=True)
    tr_u,tr_x,tr_r,tr_y,tr_neg=original

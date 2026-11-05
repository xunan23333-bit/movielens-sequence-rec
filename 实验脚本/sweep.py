import os, time, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parent.parent
os.chdir(root)
nb = json.loads((Path(__file__).resolve().parent/'搜索基线.ipynb').read_text(encoding='utf-8'))
SEED=42
dev='cuda' if torch.cuda.is_available() else 'cpu'
ratings=pd.read_csv('ratings.csv'); movies=pd.read_csv('movies.csv')
for i in (8, 9, 10, 12, 13, 15, 16, 17):
    exec(''.join(nb['cells'][i]['source']), globals())
g_t=torch.tensor(gmat,device=dev); i_t=torch.tensor(imat,device=dev)
for i in (31,32):
    exec(''.join(nb['cells'][i]['source']), globals())

results=[]
configs=[
    dict(name='dim32_neg100',dim=32,n_neg=100,wd=1e-5,features='both'),
    dict(name='dim64_neg100',dim=64,n_neg=100,wd=1e-5,features='both'),
    dict(name='dim64_neg200',dim=64,n_neg=200,wd=1e-5,features='both'),
    dict(name='dim64_no_decay',dim=64,n_neg=200,wd=0.,features='both'),
    dict(name='history10',dim=64,n_neg=200,wd=1e-5,features='both',seq_len=10),
    dict(name='history40',dim=64,n_neg=200,wd=1e-5,features='both',seq_len=40),
    dict(name='without_genre',dim=64,n_neg=200,wd=1e-5,features='stats'),
    dict(name='without_stats',dim=64,n_neg=200,wd=1e-5,features='genre'),
]
g_orig, i_orig=g_t_ext.clone(), i_t_ext.clone()
for cfg in configs:
    print('\n===',cfg['name'],flush=True)
    L = cfg.get('seq_len', 20)
    tr_u, tr_x, tr_r, tr_y = build_samples(seq_train, seq_train_r, max_len=L)
    g_t_ext = g_orig if cfg['features'] != 'stats' else torch.zeros_like(g_orig)
    i_t_ext = i_orig if cfg['features'] != 'genre' else torch.zeros_like(i_orig)
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    model=SASRecOpt(dim=cfg['dim']).to(dev)
    start=time.time()
    hist,ep=train_ce(model,epochs=30,tag=cfg['name'],select='NDCG',n_neg=cfg['n_neg'],wd=cfg['wd'],clip=5,patience=8)
    val=evaluate(model,valid_cand,bi=True)
    results.append(dict(config=cfg,best_epoch=ep,valid=val,seconds=round(time.time()-start,1)))
    Path('out/validation_sweep.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('validation',results[-1],flush=True)
print('Finished validation-only sweep',flush=True)

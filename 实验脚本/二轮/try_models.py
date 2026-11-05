import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from helpers import fit_scale, blend_scores

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / '结果'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
nb = json.loads((ROOT / '推荐系统.ipynb').read_text(encoding='utf8'))
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

for i in (8, 9, 10, 12, 13, 15, 16, 17, 23, 31, 32):
    exec(''.join(nb['cells'][i]['source']), globals())

alt_rng = np.random.default_rng(5719)
valid_alt = {}
for u, (pos, neg) in valid_cand.items():
    pool = np.flatnonzero(~seen[u, 1:]) + 1
    pool = pool[~np.isin(pool, neg)]
    assert len(pool) >= 1000
    valid_alt[u] = (pos, alt_rng.choice(pool, 1000, replace=False).tolist())

configs = [
    {'name': 'ce64_l40', 'dim': 64, 'length': 40, 'loss': 'CE'},
    {'name': 'bce64_l40', 'dim': 64, 'length': 40, 'loss': 'BCE'},
    {'name': 'ce96_l40', 'dim': 96, 'length': 40, 'loss': 'CE'},
    {'name': 'ce128_l40', 'dim': 128, 'length': 40, 'loss': 'CE'},
    {'name': 'ce64_l80', 'dim': 64, 'length': 80, 'loss': 'CE'},
]

results = []
for cfg in configs:
    L = cfg['length']
    BEST_LOSS = cfg['loss']
    tr_u, tr_x, tr_r, tr_y = build_samples(seq_train, seq_train_r, max_len=L)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = SASRecOpt(dim=cfg['dim']).to(dev)
    start = time.time()
    print('\n开始', cfg, flush=True)
    history, epoch = train_ce(model, epochs=60, tag=cfg['name'], select='NDCG',
                              n_neg=200, wd=1e-5, clip=5.0)
    primary, labels = score_matrix(model, valid_cand, bi=True)
    alternate, alt_labels = score_matrix(model, valid_alt, bi=True)
    m = scores_to_metrics(primary, labels)
    m_alt = scores_to_metrics(alternate, alt_labels)
    macro = auc_macro(model, valid_cand, bi=True)
    result = {'config': cfg, 'epoch': epoch, 'seconds': round(time.time()-start, 1),
              'params': sum(p.numel() for p in model.parameters()),
              'valid105': m, 'valid1005': m_alt, 'valid_auc_macro': macro}
    results.append(result)
    torch.save(model.state_dict(), OUT / (cfg['name'] + '.pt'))
    np.savez_compressed(OUT / (cfg['name'] + '.npz'),
                        primary=primary, labels=labels,
                        alternate=alternate, alt_labels=alt_labels)
    (OUT / '对照结果.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    print('验证结果', json.dumps(result, ensure_ascii=False), flush=True)

ce = np.load(OUT / 'ce64_l40.npz')
bce = np.load(OUT / 'bce64_l40.npz')
scales = (fit_scale(ce['primary']), fit_scale(bce['primary']))
blends = []
for weight in (0., .1, .2, .3, .4, .5, .7, 1.):
    primary = blend_scores(ce['primary'], bce['primary'], weight, scales)
    alternate = blend_scores(ce['alternate'], bce['alternate'], weight, scales)
    blends.append({'bce_weight': weight,
                   'valid105': scores_to_metrics(primary, ce['labels']),
                   'valid1005': scores_to_metrics(alternate, ce['alt_labels'])})
(OUT / '融合对照.json').write_text(json.dumps(blends, ensure_ascii=False, indent=2), encoding='utf8')
print('CE/BCE 融合验证对照完成', flush=True)

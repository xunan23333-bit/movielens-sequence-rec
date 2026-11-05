import json
import os
import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from helpers import fit_scale, blend_scores, rank_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / '结果'
os.chdir(ROOT)
nb = json.loads((ROOT / '推荐系统.ipynb').read_text(encoding='utf8'))
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
for i in (8, 9, 10, 12, 13, 15, 16, 17, 23, 31):
    exec(''.join(nb['cells'][i]['source']), globals())

trials = json.loads((OUT / '对照结果.json').read_text(encoding='utf8'))
users = sorted(valid_cand)


def validate_full(scores):
    vals = [rank_metrics(scores[k], valid_cand[u][0], seen[u, 1:])
            for k, u in enumerate(users)]
    return {'MAP10': float(np.mean([x[0] for x in vals])),
            'NDCG10': float(np.mean([x[1] for x in vals]))}


matrices = {}
metrics = {}
for row in trials:
    cfg = row['config']
    L = cfg['length']
    model = SASRecOpt(dim=cfg['dim']).to(dev)
    model.load_state_dict(torch.load(OUT / (cfg['name'] + '.pt'), map_location=dev, weights_only=True))
    model.eval()
    _, xs, rs = make_seq_input(valid_cand, bi=True)
    full = np.empty((len(users), n_item - 1), dtype=np.float32)
    with torch.no_grad():
        items = model.item(torch.arange(1, n_item, device=dev))
        for start in range(0, len(users), 64):
            h = model.encode(torch.tensor(xs[start:start+64], device=dev),
                             torch.tensor(rs[start:start+64], device=dev))
            full[start:start+64] = (h @ items.T).cpu().numpy()
    matrices[cfg['name']] = full
    np.save(OUT / (cfg['name'] + '_full.npy'), full)
    metrics[cfg['name']] = validate_full(full)
    print(cfg['name'], metrics[cfg['name']], flush=True)

ce = np.load(OUT / 'ce64_l40.npz')
bce = np.load(OUT / 'bce64_l40.npz')
scales = (fit_scale(ce['primary']), fit_scale(bce['primary']))
mixes = []
for weight in (0., .1, .2, .3, .4, .5, .7, 1.):
    scores = blend_scores(matrices['ce64_l40'], matrices['bce64_l40'], weight, scales)
    mixes.append({'bce_weight': weight, 'valid_full': validate_full(scores)})
    print('BCE weight', weight, mixes[-1]['valid_full'], flush=True)

pairs = []
ce_names = ['ce64_l40', 'ce96_l40', 'ce128_l40', 'ce64_l80']
for left, right in itertools.combinations(ce_names, 2):
    ma, sa = fit_scale(np.load(OUT / (left + '.npz'))['primary'])
    mb, sb = fit_scale(np.load(OUT / (right + '.npz'))['primary'])
    scores = ((matrices[left] - ma) / sa + (matrices[right] - mb) / sb) / 2
    result = {'models': [left, right], 'valid_full': validate_full(scores)}
    pairs.append(result)
    print(left, right, result['valid_full'], flush=True)

ma, sa = fit_scale(ce['primary'])
mb, sb = fit_scale(np.load(OUT / 'ce96_l40.npz')['primary'])
mc, sc = fit_scale(bce['primary'])
for weight in (.1, .2):
    scores = ((1-weight) * ((matrices['ce64_l40']-ma)/sa + (matrices['ce96_l40']-mb)/sb)/2
              + weight * (matrices['bce64_l40']-mc)/sc)
    result = {'models': ['ce64_l40', 'ce96_l40', 'bce64_l40'],
              'bce_weight': weight, 'valid_full': validate_full(scores)}
    pairs.append(result)
    print('64 + 96 + BCE', weight, result['valid_full'], flush=True)

result = {'models': metrics, 'blends': mixes, 'pairs': pairs}
(OUT / '全量验证.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')

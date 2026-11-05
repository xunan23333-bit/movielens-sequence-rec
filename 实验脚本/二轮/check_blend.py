import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from helpers import fit_scale, blend_scores

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / '结果'
nb = json.loads((ROOT / '推荐系统.ipynb').read_text(encoding='utf8'))
for i in (15, 16):
    exec(''.join(nb['cells'][i]['source']), globals())

a = np.load(OUT / 'ce64_l40.npz')
b = np.load(OUT / 'bce64_l40.npz')
scales = (fit_scale(a['primary']), fit_scale(b['primary']))
for weight in (0., .05, .1, .2, .3, .5, 1.):
    s1 = blend_scores(a['primary'], b['primary'], weight, scales)
    s2 = blend_scores(a['alternate'], b['alternate'], weight, scales)
    m1 = scores_to_metrics(s1, a['labels'])
    m2 = scores_to_metrics(s2, a['alt_labels'])
    print(f"BCE {weight:.2f}: 105 AUC={m1['AUC']:.4f} NDCG={m1['NDCG10']:.4f}; "
          f"1005 AUC={m2['AUC']:.4f} NDCG={m2['NDCG10']:.4f}")

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from helpers import fit_scale

HERE = Path(__file__).resolve().parent
OUT = HERE / '结果'
nb = json.loads((HERE.parent.parent / '推荐系统.ipynb').read_text(encoding='utf8'))
for i in (15, 16):
    exec(''.join(nb['cells'][i]['source']), globals())

names = ['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40']
models = [np.load(OUT / (name + '.npz')) for name in names]
primary, alternate = [], []
for item in models:
    mean, sd = fit_scale(item['primary'])
    primary.append((item['primary'] - mean) / sd)
    alternate.append((item['alternate'] - mean) / sd)
for weight in (0., .05, .1, .2, .3):
    a = np.mean(primary[:3], axis=0) * (1-weight) + primary[3] * weight
    b = np.mean(alternate[:3], axis=0) * (1-weight) + alternate[3] * weight
    m1 = scores_to_metrics(a, models[0]['labels'])
    m2 = scores_to_metrics(b, models[0]['alt_labels'])
    print(f"BCE {weight:.2f}: 105 AUC={m1['AUC']:.4f} NDCG={m1['NDCG10']:.4f}; "
          f"1005 AUC={m2['AUC']:.4f} NDCG={m2['NDCG10']:.4f}")

import itertools
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

names = ['ce64_l40', 'ce96_l40', 'ce128_l40', 'ce64_l80']
saved = {}
for name in names:
    path = OUT / (name + '.npz')
    if path.exists():
        saved[name] = np.load(path)

all_results = []
for left, right in itertools.combinations(saved, 2):
    a, b = saved[left], saved[right]
    ma, sa = fit_scale(a['primary'])
    mb, sb = fit_scale(b['primary'])
    s1 = ((a['primary'] - ma) / sa + (b['primary'] - mb) / sb) / 2
    s2 = ((a['alternate'] - ma) / sa + (b['alternate'] - mb) / sb) / 2
    m1 = scores_to_metrics(s1, a['labels'])
    m2 = scores_to_metrics(s2, a['alt_labels'])
    result = {'models': [left, right], 'valid105': m1, 'valid1005': m2}
    all_results.append(result)
    print(f"{left} + {right}: 105 AUC={m1['AUC']:.4f} NDCG={m1['NDCG10']:.4f}; "
          f"1005 AUC={m2['AUC']:.4f} NDCG={m2['NDCG10']:.4f}")

if 'ce64_l40' in saved and 'ce96_l40' in saved:
    bce = np.load(OUT / 'bce64_l40.npz')
    a, b = saved['ce64_l40'], saved['ce96_l40']
    ma, sa = fit_scale(a['primary'])
    mb, sb = fit_scale(b['primary'])
    mc, sc = fit_scale(bce['primary'])
    for weight in (.1, .2):
        s1 = (1-weight)*((a['primary']-ma)/sa + (b['primary']-mb)/sb)/2 + weight*(bce['primary']-mc)/sc
        s2 = (1-weight)*((a['alternate']-ma)/sa + (b['alternate']-mb)/sb)/2 + weight*(bce['alternate']-mc)/sc
        m1 = scores_to_metrics(s1, a['labels'])
        m2 = scores_to_metrics(s2, a['alt_labels'])
        all_results.append({'models': ['ce64_l40', 'ce96_l40', 'bce64_l40'],
                            'bce_weight': weight, 'valid105': m1, 'valid1005': m2})
        print(f"64 + 96 + BCE {weight:.1f}: 105 AUC={m1['AUC']:.4f} NDCG={m1['NDCG10']:.4f}; "
              f"1005 AUC={m2['AUC']:.4f} NDCG={m2['NDCG10']:.4f}")

(OUT / '模型融合对照.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf8')

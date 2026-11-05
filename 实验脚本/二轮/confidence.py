import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from helpers import fit_scale, rank_metrics

here = Path(__file__).resolve().parent
root = here.parent.parent
out = here / '结果'
nb = json.loads((root / '推荐系统.ipynb').read_text(encoding='utf8'))
ratings = pd.read_csv(root / 'ratings.csv')
SEED = 42
for i in (8, 9):
    exec(''.join(nb['cells'][i]['source']), globals())

names = ['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40', 'bce64_l40_7']
full = []
sampled_scores = []
labels = None
for name in names:
    saved = np.load(out / (name + '.npz'))
    sampled = saved['primary']
    labels = saved['labels']
    mean, sd = fit_scale(sampled)
    sampled_scores.append((sampled - mean) / sd)
    full.append((np.load(out / (name + '_full.npy')) - mean) / sd)
old = np.mean(full[:3], axis=0)
new = .8 * old + .2 * np.mean(full[3:], axis=0)
diff = []
for k, u in enumerate(sorted(valid_cand)):
    a = rank_metrics(old[k], valid_cand[u][0], seen[u, 1:])[1]
    b = rank_metrics(new[k], valid_cand[u][0], seen[u, 1:])[1]
    diff.append(b-a)
diff = np.asarray(diff)
rng = np.random.default_rng(287)
samples = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
result = {'NDCG_difference': float(diff.mean()),
          'bootstrap_95': [float(x) for x in np.percentile(samples, [2.5, 97.5])],
          'users': len(diff)}
old_sample = np.mean(sampled_scores[:3], axis=0)
new_sample = .8 * old_sample + .2 * np.mean(sampled_scores[3:], axis=0)
auc_diffs = []
for _ in range(500):
    ix = rng.integers(0, len(labels), len(labels))
    y = labels[ix].ravel()
    auc_diffs.append(roc_auc_score(y, new_sample[ix].ravel()) -
                     roc_auc_score(y, old_sample[ix].ravel()))
result['sampled_AUC_difference'] = float(roc_auc_score(labels.ravel(), new_sample.ravel()) -
                                         roc_auc_score(labels.ravel(), old_sample.ravel()))
result['sampled_AUC_bootstrap_95'] = [float(x) for x in np.percentile(auc_diffs, [2.5, 97.5])]
(out / '五模型波动.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, ensure_ascii=False), flush=True)

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from helpers import fit_scale, rank_metrics

source = (Path(__file__).resolve().parent / 'check_full.py').read_text(encoding='utf8')
exec(compile(source.split('trials = ')[0], '<full-setup>', 'exec'), globals())
users = sorted(valid_cand)


def get_full(name, dim):
    path = OUT / (name + '_full.npy')
    if path.exists():
        return np.load(path)
    model = SASRecOpt(dim=dim).to(dev)
    model.load_state_dict(torch.load(OUT / (name + '.pt'), map_location=dev, weights_only=True))
    model.eval()
    _, xs, rs = make_seq_input(valid_cand, bi=True)
    scores = np.empty((len(users), n_item - 1), dtype=np.float32)
    with torch.no_grad():
        items = model.item(torch.arange(1, n_item, device=dev))
        for start in range(0, len(users), 64):
            h = model.encode(torch.tensor(xs[start:start+64], device=dev),
                             torch.tensor(rs[start:start+64], device=dev))
            scores[start:start+64] = (h @ items.T).cpu().numpy()
    np.save(path, scores)
    return scores


def scores_for(names, weights):
    primary, alternate, full = [], [], []
    for name in names:
        item = np.load(OUT / (name + '.npz'))
        mu, sd = fit_scale(item['primary'])
        primary.append((item['primary'] - mu) / sd)
        alternate.append((item['alternate'] - mu) / sd)
        dim = 96 if name.startswith('ce96') else 64
        full.append((get_full(name, dim) - mu) / sd)
    shape = np.array(weights)[:, None, None]
    return (np.sum(np.stack(primary) * shape, axis=0),
            np.sum(np.stack(alternate) * shape, axis=0),
            np.sum(np.stack(full) * shape, axis=0), item['labels'], item['alt_labels'])


def full_metrics(scores):
    vals, aucs = [], []
    for k, u in enumerate(users):
        pos = np.asarray(valid_cand[u][0]) - 1
        vals.append(rank_metrics(scores[k], valid_cand[u][0], seen[u, 1:]))
        neg = scores[k, ~seen[u, 1:]][None, :]
        positive = scores[k, pos][:, None]
        aucs.append(float(((neg < positive) + .5 * (neg == positive)).mean()))
    return {'MAP10': float(np.mean([x[0] for x in vals])),
            'NDCG10': float(np.mean([x[1] for x in vals])),
            'AUC_macro': float(np.mean(aucs))}, np.array([x[1] for x in vals])


plans = {
    '原三种子集成': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026'], [1/3]*3),
    '新方案种子42': (['ce64_l40', 'ce96_l40', 'bce64_l40'], [.45, .45, .1]),
    '新方案种子7': (['ce64_l40_7', 'ce96_l40_7', 'bce64_l40_7'], [.45, .45, .1]),
    '新方案20%种子42': (['ce64_l40', 'ce96_l40', 'bce64_l40'], [.4, .4, .2]),
    '三种子加BCE10%': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40'], [.3, .3, .3, .1]),
    '三种子加BCE20%': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40'], [.8/3, .8/3, .8/3, .2]),
    '三种子加BCE种子7': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40_7'], [.3, .3, .3, .1]),
    '三种子加双BCE10%': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40', 'bce64_l40_7'], [.3, .3, .3, .05, .05]),
    '三种子加双BCE20%': (['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026', 'bce64_l40', 'bce64_l40_7'], [.8/3, .8/3, .8/3, .1, .1]),
}
rows, per_user = {}, {}
for name, (names, weights) in plans.items():
    a, b, full, y, yb = scores_for(names, weights)
    fm, each = full_metrics(full)
    rows[name] = {'models': names, 'weights': weights,
                  'valid105': scores_to_metrics(a, y),
                  'valid1005': scores_to_metrics(b, yb), 'valid_full': fm}
    per_user[name] = each
    print(name, json.dumps(rows[name], ensure_ascii=False), flush=True)

rng = np.random.default_rng(287)
delta = per_user['新方案种子42'] - per_user['原三种子集成']
boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)])
rows['新旧全量NDCG差值'] = {'mean': float(delta.mean()),
                            'bootstrap95': [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
delta2 = per_user['三种子加BCE10%'] - per_user['原三种子集成']
boot2 = np.array([delta2[rng.integers(0, len(delta2), len(delta2))].mean() for _ in range(2000)])
rows['BCE10%全量NDCG差值'] = {'mean': float(delta2.mean()),
                             'bootstrap95': [float(x) for x in np.percentile(boot2, [2.5, 97.5])]}
(OUT / '验证稳定性.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
print('用户重采样差值', rows['新旧全量NDCG差值'], flush=True)

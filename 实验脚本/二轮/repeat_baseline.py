import json
from pathlib import Path

source = (Path(__file__).resolve().parent / 'try_models.py').read_text(encoding='utf8')
exec(compile(source.split('configs = [')[0], '<experiment-setup>', 'exec'), globals())

for seed, dim, loss in ((7, 64, 'CE'), (2026, 64, 'CE'), (7, 96, 'CE'), (7, 64, 'BCE')):
    L = 40
    BEST_LOSS = loss
    tr_u, tr_x, tr_r, tr_y = build_samples(seq_train, seq_train_r, max_len=L)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SASRecOpt(dim=dim).to(dev)
    start = time.time()
    history, epoch = train_ce(model, epochs=60, tag=f'{loss} {dim} seed {seed}', select='NDCG',
                              n_neg=200, wd=1e-5, clip=5.0)
    primary, labels = score_matrix(model, valid_cand, bi=True)
    alternate, alt_labels = score_matrix(model, valid_alt, bi=True)
    name = f'{loss.lower()}{dim}_l40_{seed}'
    torch.save(model.state_dict(), OUT / (name + '.pt'))
    np.savez_compressed(OUT / (name + '.npz'), primary=primary, labels=labels,
                        alternate=alternate, alt_labels=alt_labels)
    print(name, 'epoch', epoch, 'seconds', round(time.time() - start, 1),
          scores_to_metrics(primary, labels), flush=True)

names = ['ce64_l40', 'ce64_l40_7', 'ce64_l40_2026']
models = [np.load(OUT / (name + '.npz')) for name in names]
primary, alternate = [], []
for item in models:
    mu, sd = fit_scale(item['primary'])
    primary.append((item['primary'] - mu) / sd)
    alternate.append((item['alternate'] - mu) / sd)
results = {'valid105': scores_to_metrics(np.mean(primary, axis=0), models[0]['labels']),
           'valid1005': scores_to_metrics(np.mean(alternate, axis=0), models[0]['alt_labels'])}
(OUT / '三种子基线.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
print('三种子验证基线', json.dumps(results, ensure_ascii=False), flush=True)

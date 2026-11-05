from pathlib import Path

source=(Path(__file__).parent/'sweep.py').read_text(encoding='utf8')
exec(compile(source.split('results=[]')[0],'<sweep-setup>','exec'),globals())
cell=''.join(nb['cells'][32]['source'])
function=cell[cell.index('def train_ce('):]
old='loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=dev))'
new='loss = F.binary_cross_entropy_with_logits(logits[:, 0], torch.ones(len(idx), device=dev)) + F.binary_cross_entropy_with_logits(logits[:, 1:], torch.zeros_like(logits[:, 1:]))'
assert old in function
exec(function.replace('def train_ce(', 'def train_bce(').replace(old,new),globals())
cfg=dict(name='balanced_bce',dim=64,n_neg=200,wd=1e-5,features='both',loss='BCE')
torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
model=SASRecOpt(dim=64).to(dev); start=time.time()
hist,ep=train_bce(model,epochs=30,tag='balanced_bce',select='NDCG',n_neg=200,wd=1e-5,clip=5,patience=8)
val=evaluate(model,valid_cand,bi=True)
result=dict(config=cfg,best_epoch=ep,valid=val,seconds=round(time.time()-start,1))
path=Path('out/validation_sweep.json')
records=json.loads(path.read_text(encoding='utf8'))
records=[r for r in records if r['config']['name']!='balanced_bce']+[result]
path.write_text(json.dumps(records,indent=2),encoding='utf8')
print('BCE validation result',result,flush=True)

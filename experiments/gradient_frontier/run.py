"""Frozen-checkpoint, finite-population trace variance and synchronized GPU timing."""
import argparse, hashlib, json, os, time
from pathlib import Path
import fsspec
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .kernels import restore, batch_grad, features, per_grad, corrected, plain

ROOT='marin-us-east-04a/marin/experiments/gradient-frontier-20261121'
DEPTHS=(-2,0,1,4)
NAMES={-2:'unigram',0:'embedding',1:'layer1',4:'layer4'}
MICRO=4
NFIT=1024
TREP=17

def ready(x):
    return jax.block_until_ready(x)

def measured(f):
    t=time.perf_counter()
    result=ready(f())
    return result,time.perf_counter()-t

def provenance(fs):
    folder=Path('/tmp/gradient-frontier-inputs')
    folder.mkdir(exist_ok=True)
    meta=json.loads(fs.cat(ROOT+'/inputs/inputs.json'))
    for name,expected in meta['sha256'].items():
        dest=folder/name
        if not dest.exists():fs.get_file(ROOT+'/inputs/'+name,str(dest))
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected
    return folder,meta

def gather_features(model,windows,depth,stats):
    f=lambda x:features(model,x,depth,stats)
    ready(f(windows[:32]))
    return np.concatenate([np.asarray(f(windows[i:i+32])) for i in range(0,len(windows),32)])

def designs(train_features):
    weights=[]; specs={}
    for depth,raw in train_features.items():
        x=raw.astype(np.float64)
        x-=x.mean(0)
        _,s,v=np.linalg.svd(x,full_matrices=False)
        rank=min(4,x.shape[1])
        projection=v[:rank].T/(s[:rank]/np.sqrt(len(x))+1e-10)
        z=x@projection
        start=sum(w.shape[1] for w in weights)
        weights.append(z/(len(x)*1.05))
        specs[NAMES[depth]]=(depth,projection,slice(start,start+rank))
        if depth==0:
            zg=z[:,:1].reshape(-1,8,1).mean(1)
            wg=zg/(max(float(np.sum(zg**2)),1e-12)*1.05*8)
            weights.append(np.repeat(wg,8,axis=0))
            specs['embedding-batchfit']=(depth,projection[:,:1],slice(start+rank,start+rank+1))
    return j.asarray(np.concatenate(weights,axis=1).astype('float32')),specs

@jax.jit
def accumulate(acc,w,g):
    return acc+j.matmul(w.T,g,precision=jax.lax.Precision.HIGHEST)

@jax.jit
def moments(g,bank):
    return g.sum(0),j.einsum('ij,ij->i',g,g),j.matmul(g,bank.T,precision=jax.lax.Precision.HIGHEST)

def fit(model,train,stats,w):
    dummy=ready(per_grad(model,train[:MICRO],stats))
    bank=j.zeros((w.shape[1],dummy.shape[-1]),dtype=j.float32)
    ready(accumulate(bank,w[:MICRO],dummy))
    t=time.perf_counter()
    for i in range(0,len(train),MICRO):
        bank=accumulate(bank,w[i:i+MICRO],per_grad(model,train[i:i+MICRO],stats))
        if i%32==0:ready(bank)
    return ready(bank),time.perf_counter()-t

def main(seed):
    assert jax.default_backend()=='gpu'
    configure_storage()
    fs=fsspec.filesystem('s3')
    folder,meta=provenance(fs)
    model=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens']
    split=np.random.default_rng(8501).permutation(len(data))
    rng=np.random.default_rng(seed)
    train=j.asarray(data[rng.choice(split[:2048],NFIT,replace=False)])
    test=j.asarray(data[split[2048:]])
    counts=np.bincount(np.asarray(train[:,:-1]).ravel(),minlength=128256)+1
    stats=j.asarray(-np.log(counts/counts.sum()),dtype=j.float32)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    ident=os.urandom(5).hex()
    cfg=dict(seed=seed,fit_samples=NFIT,test_samples=len(test),sequence_predictions=256,
             checkpoint_step=4000,params=33786496,device=jax.devices()[0].device_kind,inputs=meta,
             split_seed=8501,feature_pca_ridge=.05,microbatch=MICRO,precision='BF16 forward, FP32 weights and outputs',
             batch_sizes=[1,2,4,5,6,8,9,10,12,16,24,32],timing_repeats=TREP)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='w128-step4000',
                   id=ident,name=f'frontier-s{seed}-{ident}',mode='online',config=cfg)
    result=dict(config=cfg,wandb=run.url,rows=[],checks={})
    def save(stage):
        result['stage']=stage
        fs.pipe(ROOT+f'/attempts/s{seed}-{ident}.json',json.dumps(result,allow_nan=False).encode())
        print('FRONTIER',stage,flush=True)
    save('verified inputs')
    train_features={};feature_fit_s={}
    for d in DEPTHS:
        ready(features(model,train[:32],d,stats))
        t=time.perf_counter()
        train_features[d]=gather_features(model,train,d,stats)
        feature_fit_s[NAMES[d]]=time.perf_counter()-t
    weights,specs=designs(train_features)
    g,x,loss=batch_grad(model,train[:MICRO],1,stats)
    per=per_grad(model,train[:MICRO],stats)
    discrepancy=float(j.linalg.norm(g-per.mean(0))/j.linalg.norm(g))
    assert np.isfinite(discrepancy) and discrepancy<0.025
    result['checks'].update(loss=float(loss),batch_vs_per_sample_relative_error=discrepancy)
    run.log({'diagnostic/train_loss':float(loss),'diagnostic/batch_gradient_error':discrepancy})
    bank,fit_s=fit(model,train,stats,weights)
    result.update(shared_fit_seconds=fit_s,feature_fit_seconds=feature_fit_s)
    save('calibrated')
    test_features={d:gather_features(model,test,d,stats) for d in DEPTHS}
    sumg=j.zeros((bank.shape[1],),j.float32)
    sq=[];cross=[]
    for i in range(0,len(test),MICRO):
        gradients=per_grad(model,test[i:i+MICRO],stats)
        sg,ng,cg=moments(gradients,bank)
        sumg=sumg+sg
        sq.extend(np.asarray(ng));cross.extend(np.asarray(cg))
    avg=sumg/len(test)
    raw=float(np.mean(sq,dtype=np.float64)-float(j.vdot(avg,avg)))
    gram=np.asarray(j.matmul(bank,bank.T,precision=jax.lax.Precision.HIGHEST),np.float64)
    cross=np.asarray(cross,np.float64)
    result['raw_trace']=raw
    assert raw>0
    plans={}
    for name,(depth,proj,sl) in specs.items():
        x=test_features[depth].astype(np.float64)
        center=x.mean(0)
        z=(x-center)@proj
        for r in sorted(set([1,proj.shape[1]])):
            sub=slice(sl.start,sl.start+r)
            small=z[:,:r]
            var_h=float(np.sum((small.T@small/len(x))*gram[sub,sub]))
            cov=float((small*cross[:,sub]).sum()/len(x))
            resid=raw+var_h-2*cov
            assert resid>0
            plans[name,r]=(depth,j.asarray(proj[:,:r],j.float32),bank[sub],j.asarray(center,j.float32),resid)
    save('measured full-gradient variance')
    # Each timing launches a real estimator, using device-resident examples.
    # Both arms return one identically laid-out FP32 full-model gradient.
    cases=[('SGD',b,0,0) for b in cfg['batch_sizes']]
    for name,r in plans:
        depth=plans[name,r][0]
        for b in (4,8,16):
            for ratio in ((0,) if depth<=0 else (2,4)):
                cases.append((name,b,r,ratio))
    rng.shuffle(cases)
    for case_no,(name,b,r,ratio) in enumerate(cases):
        blocks=[j.asarray(data[rng.choice(split[2048:],max(1,ratio)*b,replace=True)]) for _ in range(TREP+2)]
        if name=='SGD':
            calc=lambda x:plain(model,x,stats)
            variance=raw/b
        else:
            depth,proj,beta,center,resid=plans[name,r]
            def calc(x):
                ref=x[b:] if ratio else None
                return corrected(model,x[:b],ref,depth,stats,proj,beta,center)
            variance=(resid+(raw-resid)/ratio)/b if ratio else resid/b
        ready(calc(blocks[0]));ready(calc(blocks[1]))
        times=[measured(lambda z=z:calc(z))[1]*1000 for z in blocks[2:]]
        row=dict(method=name,b=b,rank=r,pool_ratio=ratio,ms_samples=times,
                 ms=float(np.median(times)),trace=variance,relative_to_same_batch=variance*b/raw)
        result['rows'].append(row)
        run.log({'arm/trace':variance,'arm/ms':row['ms'],'arm/index':case_no})
        if case_no%8==0:save(f'timed {case_no+1}/{len(cases)}')
    # Independent direct batch-pair checks against the moment-based scaling identity.
    for name,b,r,ratio in [('SGD',4,0,0),('SGD',16,0,0),('embedding',4,1,0),('layer1',4,4,4)]:
        pair=[]
        for rep in range(48):
            items=[test[rng.integers(0,len(test),max(ratio,1)*b)] for _ in range(2)]
            if name=='SGD':
                v=[plain(model,it,stats) for it in items]
            else:
                depth,proj,beta,center,_=plans[name,r]
                v=[corrected(model,it[:b],it[b:] if ratio else None,depth,stats,proj,beta,center) for it in items]
            delta=v[0]-v[1]
            pair.append(float(j.vdot(delta,delta)/2))
        result['checks'][f'{name}-b{b}-r{r}-pool{ratio}']=pair
    save('complete')
    fs.pipe(ROOT+f'/completed/s{seed}.json',json.dumps(result,allow_nan=False).encode())
    run.finish()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    main(parser.parse_args().seed)

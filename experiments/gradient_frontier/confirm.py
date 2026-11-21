"""Direct paired variance contrasts and interleaved latency checks at batch eight."""
import os,json,time,sys
import jax
import jax.numpy as j
import numpy as np
import wandb,fsspec
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .kernels import plain,corrected,restore
from .run import ROOT,provenance,gather_features,fit,ready

def main(seed):
    assert jax.default_backend()=='gpu'
    configure_storage()
    fs=fsspec.filesystem('s3')
    folder,meta=provenance(fs)
    m=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens']
    split=np.random.default_rng(8501).permutation(len(data))
    rng=np.random.default_rng(seed)
    train=j.asarray(data[rng.choice(split[:2048],1024,replace=False)])
    test=j.asarray(data[split[2048:]])
    counts=np.bincount(np.asarray(train[:,:-1]).ravel(),minlength=128256)+1
    stats=j.asarray(-np.log(counts/counts.sum()),j.float32)
    ident=os.urandom(5).hex()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='w128-confirm',
        name=f'paired-s{seed}-{ident}',id=ident,mode='online',config=dict(seed=seed,inputs=meta,fit=1024,b=8,pairs=384,randomized_timing_blocks=100))
    methods={};cost={}
    for name,d in [('unigram',-2),('embedding-RRR',0),('layer1-RRR',1)]:
        raw=gather_features(m,train,d,stats).astype(np.float64)
        start=time.perf_counter()
        x=raw-raw.mean(0);_,s,v=np.linalg.svd(x,full_matrices=False)
        p=v[:16].T/(s[:16]/np.sqrt(len(x))+1e-10)
        w=j.asarray((x@p/(len(x)*1.05)).astype(np.float32))
        ready(w)
        prep=time.perf_counter()-start
        bank,fit_s=fit(m,train,stats,w)
        start=time.perf_counter()
        if d>=0:
            signal=np.asarray(j.matmul(bank,bank.T,precision=jax.lax.Precision.HIGHEST),np.float64)
            _,u=np.linalg.eigh(signal);u=u[:,::-1][:,:8].copy()
            p=p@u;bank=ready(j.asarray(u.T,j.float32)@bank)
        cost[name]=fit_s+prep+time.perf_counter()-start
        center=gather_features(m,test,d,stats).mean(0,dtype=np.float64)
        methods[name]=(d,j.asarray(p,j.float32),bank,j.asarray(center,j.float32))
    def estimate(name,x):
        if name.startswith('SGD'):
            return plain(m,x[:int(name[3:])],stats)
        d,p,bank,center=methods[name]
        return corrected(m,x[:8],x[8:32] if d==1 else None,d,stats,p,bank,center)
    names=['SGD8','SGD9','SGD10','unigram','embedding-RRR','layer1-RRR']
    for name in names:ready(estimate(name,test[:32]))
    timings={name:[] for name in names}
    for k in range(100):
        # Materialize identical host-sampled inputs before timed calls.
        x=ready(test[rng.integers(len(test),size=32)])
        for name in rng.permutation(names):
            start=time.perf_counter();ready(estimate(name,x))
            timings[name].append(1000*(time.perf_counter()-start))
    out=dict(seed=seed,wandb=run.url,timing_ms=timings,fit_seconds=cost,direct={name:[] for name in names if name not in ['SGD9','SGD10']})
    for k in range(384):
        batches=[test[rng.integers(len(test),size=32)] for _ in range(2)]
        for name in out['direct']:
            delta=estimate(name,batches[0])-estimate(name,batches[1])
            out['direct'][name].append(float(j.vdot(delta,delta))/2)
        if k%96==95:
            fs.pipe(ROOT+f'/confirmation/attempts/{seed}-{ident}.json',json.dumps(out).encode())
            run.log({'pair/count':k+1,'pair/raw_trace':np.mean(out['direct']['SGD8'])})
    assert all(np.isfinite(values).all() for values in out['direct'].values())
    fs.pipe(ROOT+f'/confirmation/s{seed}.json',json.dumps(out).encode())
    print('PAIRED CONFIRMATION COMPLETE',seed,flush=True)
    run.finish()
if __name__=='__main__':main(int(sys.argv[1]))

"""Warm every case before randomized same-GPU crossover timing at one target batch."""
import json,sys,time
import fsspec,jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .kernels import plain,corrected,restore
from .run import ROOT,provenance,gather_features,fit,designs,ready,NAMES

def main(seed,b):
    configure_storage(); fs=fsspec.filesystem('s3')
    folder,meta=provenance(fs);m=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens']
    split=np.random.default_rng(8501).permutation(len(data))
    rng=np.random.default_rng(seed)
    train=j.asarray(data[rng.choice(split[:2048],1024,replace=False)]);test=j.asarray(data[split[2048:]])
    counts=np.bincount(np.asarray(train[:,:-1]).ravel(),minlength=128256)+1
    stats=j.asarray(-np.log(counts/counts.sum()),j.float32)
    raw={d:gather_features(m,train,d,stats) for d in (-2,0,1,4)}
    centers={d:j.asarray(gather_features(m,test,d,stats).mean(0,dtype=np.float64),j.float32) for d in raw}
    weights,specs=designs(raw)
    bank,_=fit(m,train,stats,weights)
    states={name:(d,j.asarray(p,j.float32),bank[sl]) for name,(d,p,sl) in specs.items()}
    for d in (0,1,4):
        x=raw[d].astype(np.float64);x-=x.mean(0)
        _,s,v=np.linalg.svd(x,full_matrices=False);p=v[:16].T/(s[:16]/32+1e-10)
        w=j.asarray((x@p/(1024*1.05)).astype(np.float32))
        bank,_=fit(m,train,stats,w)
        _,u=np.linalg.eigh(np.asarray(j.matmul(bank,bank.T,precision=jax.lax.Precision.HIGHEST),np.float64))
        u=u[:,::-1][:,:8].copy()
        states[NAMES[d]+'-RRR']=(d,j.asarray(p@u,j.float32),ready(j.asarray(u.T,j.float32)@bank))
    cases={}; desc={}
    sizes=[1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,24,32]
    for n in sizes:
        name=f'SGD.{n}';cases[name]=lambda x,ref:plain(m,x,stats);desc[name]=('SGD',n,0,0)
    for method,(depth,projection,basis) in states.items():
        ranks=(1,4,8) if 'RRR' in method else sorted(set([1,projection.shape[1]]))
        for rank in ranks:
            proj=projection[:,:rank];beta=basis[:rank]
            for ratio in ((0,) if depth<=0 else (2,4)):
                key=f'{method}.{b}.{rank}.{ratio}'
                cases[key]=lambda x,ref,d=depth,p=proj,be=beta:corrected(m,x,ref,d,stats,p,be,centers[d])
                desc[key]=(method,b,rank,ratio)
    # Independent device-resident token samples. Build slices before starting any timer.
    # All cases see the same core sequence IDs within a repetition.
    shape_keys={(n,ratio) for _,n,_,ratio in desc.values()}
    samples=[]
    for it in range(82):
        batch=rng.choice(split[2048:],max(32,4*b),replace=True)
        samples.append({(n,ratio):(j.asarray(data[batch[:n]]),j.asarray(data[batch[n:ratio*n]]) if ratio else None)
                        for n,ratio in shape_keys})
    ready(samples)
    for name,(method,n,rank,ratio) in desc.items():
        for it in (0,1):ready(cases[name](*samples[it][n,ratio]))
    times={name:[] for name in cases}
    for i,sample in enumerate(samples[2:]):
        for name in rng.permutation(list(cases)):
            _,n,_,ratio=desc[name]
            x,ref=sample[n,ratio]
            start=time.perf_counter()
            ready(cases[name](x,ref))
            times[name].append(1000*(time.perf_counter()-start))
    out=dict(seed=seed,b=b,device=jax.devices()[0].device_kind,rows=[
        dict(method=me,b=n,rank=r,pool_ratio=pool,ms_samples=times[tag]) for tag,(me,n,r,pool) in desc.items()])
    fs.pipe(ROOT+f'/timing/s{seed}-b{b}.json',json.dumps(out,allow_nan=False).encode())
    print('CROSSOVER COMPLETE',b,seed,len(cases),flush=True)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

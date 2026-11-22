"""Measure alignment before allocating end-to-end timing to an input-only control."""
import json,sys,time
import equinox as eqx,fsspec,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .run import ROOT,ready,provenance
from .kernels import restore
from .multifidelity import shortlist,estimator
from .adaptive_correction import sample
from .bigram import setup,probe,mixed

def main(k,seed):
    configure_storage();fs=fsspec.filesystem('s3');folder,_=provenance(fs);m=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens'];split=np.random.default_rng(8501).permutation(len(data))
    cal=j.asarray(data[split[:1024]]);ds=data[split[2048:]];rng=np.random.default_rng(seed)
    bank=ready(setup(m,cal,k));head=ready(shortlist(cal,8192,128256))
    prep=[]
    for _ in range(20):
        t=time.perf_counter();ready(setup(m,cal,k));prep.append((time.perf_counter()-t)*1000)
    vals=[]
    for it in range(256):
        a,b=[j.asarray(ds[rng.integers(len(ds),size=4)]) for _ in range(2)]
        ka,kb=jax.random.split(jax.random.PRNGKey(seed+it))
        dh=probe(m,a,bank)-probe(m,b,bank)
        g=estimator(m,a,head,4,'full')-estimator(m,b,head,4,'full')
        c=sample(m,a,ka,head,32,1)-sample(m,b,kb,head,32,1)
        vals.append(np.asarray(j.stack([j.vdot(g,g),j.vdot(c,c),j.vdot(dh,dh),j.vdot(g,dh),j.vdot(c,dh)])).tolist())
    gg,cc,hh,gh,ch=np.mean(vals,axis=0)
    out=dict(k=k,seed=seed,prep_ms=prep,pairs=vals,power=float(ch/hh),fraction=float(ch*ch/(cc*hh)),true_fraction=float(gh*gh/(gg*hh)))
    fs.pipe(ROOT+f'/bigram/k{k}-s{seed}.json',json.dumps(out).encode())
    print('BIGRAM',k,out['fraction'],out['true_fraction'],out['power'],flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))

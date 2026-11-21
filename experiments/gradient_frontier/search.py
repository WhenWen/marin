"""Charge one shortlist build per ten estimates; measure exact finite-population moments."""
from collections import defaultdict
import sys,json,os,time
import numpy as np
import jax
import jax.numpy as j
import fsspec,wandb
from rigging.secrets import require_secret
from .kernels import restore
from .run import ROOT,provenance,ready
from .multifidelity import estimator,per_example,shortlist
from experiments.hybrid_transfer.storage import configure_storage

@jax.jit
def sums(g,h):
    return (g.sum(0),h.sum(0),j.stack([j.vdot(g,g),j.vdot(h,h),j.vdot(g,h)]))

def main(k,seed):
    configure_storage();fs=fsspec.filesystem('s3')
    folder,meta=provenance(fs);m=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens']; rng=np.random.default_rng(seed)
    split=np.random.default_rng(8501).permutation(len(data))
    cal=j.asarray(data[rng.choice(split[:2048],1024,replace=False)])
    test=j.asarray(data[split[2048:]])
    ident=os.urandom(5).hex()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(shortlist=k,seed=seed,inputs=meta,calibration_documents=1024,test_docs=len(test),amortized_steps=10,
             device=jax.devices()[0].device_kind,proxy='shortlist log-normalizer and observed target logit')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='shortlist-multifidelity',
                   name=f'shortlist-{k}-s{seed}-{ident}',id=ident,mode='online',config=cfg)
    out=dict(config=cfg,wandb=run.url,rows=[])
    def save(stage):
        out['stage']=stage
        fs.pipe(ROOT+f'/shortlist/attempts/k{k}-s{seed}-{ident}.json',json.dumps(out,allow_nan=False).encode())
        print('SHORTLIST',stage,flush=True)
    ids=ready(shortlist(cal,k,128256))
    prep=[]
    for _ in range(40):
        start=time.perf_counter();ready(shortlist(cal,k,128256));prep.append(1000*(time.perf_counter()-start))
    out['preprocess_ms']=float(np.mean(prep))
    gs=hs=0.;ss=np.zeros(3)
    for i in range(0,len(test),4):
        t=test[i:i+4]
        g=per_example(m,t,ids,'full');h=per_example(m,t,ids,'cheap')
        a,c,v=sums(g,h);gs=gs+a;hs=hs+c;ss+=np.asarray(v,np.float64)
    gs,hs=gs/len(test),hs/len(test)
    vg,vh,cov=ss/len(test)-np.array([float(j.vdot(gs,gs)),float(j.vdot(hs,hs)),float(j.vdot(gs,hs))])
    ratio=(vg+vh-2*cov)/vg
    out['moments']=dict(full=vg,cheap=vh,cov=cov,unit_power_residual_fraction=ratio,oracle_power=cov/vh,
                        oracle_residual_fraction=1-cov*cov/(vg*vh))
    save('variance')
    run.log({'variance/residual_fraction':ratio,'variance/oracle_residual_fraction':out['moments']['oracle_residual_fraction']})
    # Select the core independently of the iid pool ordering by exchangeability.
    # Cases use identical real input IDs up to the relevant batch length.
    cases=[('SGD',b,b) for b in [4,8,12,16,24,32,48,64,96]]
    cases.extend(('MF',b,b*factor) for b in (2,4,8,16) for factor in (2,4,8))
    maxm=max(M for _,_,M in cases)
    pools=[]
    for _ in range(42):
        ind=rng.choice(split[2048:],maxm,replace=True)
        pools.append({M:j.asarray(data[ind[:M]]) for M in set(M for _,_,M in cases)})
    ready(pools)
    def measure(case,q):
        method,b,M=case
        return estimator(m,q[M],ids,b,'full' if method=='SGD' else 'corrected',1.)
    for case in cases:ready(measure(case,pools[0]));ready(measure(case,pools[1]))
    times=defaultdict(list)
    for pool in pools[2:]:
        for ix in rng.permutation(len(cases)):
            case=cases[ix];start=time.perf_counter();ready(measure(case,pool))
            times[case].append(1000*(time.perf_counter()-start))
    for case in cases:
        method,b,M=case
        v=vg/b if method=='SGD' else vg/M+(1/b-1/M)*vg*ratio
        out['rows'].append(dict(method=method,b=b,M=M,trace=v,ms=times[case],
             amortized_ms=np.mean(times[case])+(0 if method=='SGD' else out['preprocess_ms']/10)))
    save('complete')
    fs.pipe(ROOT+f'/shortlist/k{k}-s{seed}.json',json.dumps(out).encode())
    run.finish()
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

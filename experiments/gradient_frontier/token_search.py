"""Unbiased stratified target-token sampling retaining true causal context."""
import time,os,sys,json
from collections import defaultdict
import numpy as np
import jax
import jax.numpy as j
import equinox as eqx
import fsspec,wandb
from jax.flatten_util import ravel_pytree
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .multifidelity import activations,full_losses,estimator
from .kernels import restore
from .run import ROOT,provenance,ready

def token_loss(model,tok,key,k):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    h=activations(m,tok)
    n,t,_=h.shape
    assert t%k==0
    pos=j.arange(k)[None,:]*(t//k)+jax.random.randint(key,(n,k),0,t//k)
    x=j.take_along_axis(h,pos[:,:,None],axis=1)
    target=j.take_along_axis(tok[:,1:],pos,axis=1)
    return full_losses(x,m.lm_head.weight.array,target,1/m.config.width_multiplier).mean()
@eqx.filter_jit
def token_estimator(m,tokens,key,k):
    return ravel_pytree(eqx.filter_grad(token_loss)(m,tokens,key,k))[0]

def main(k,seed):
    configure_storage();fs=fsspec.filesystem('s3')
    folder,meta=provenance(fs);m=restore(folder/'weights.npz')
    data=np.load(folder/'windows.npz')['tokens']; rng=np.random.default_rng(seed)
    split=np.random.default_rng(8501).permutation(len(data))[2048:]
    ident=os.urandom(5).hex();os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='token-subsampling',
        name=f'positions-{k}-s{seed}-{ident}',id=ident,mode='online',config=dict(seed=seed,k=k,inputs=meta,paired_repeats=256))
    out=dict(k=k,seed=seed,wandb=run.url,rows=[],pairs=[])
    ids=j.zeros(1,j.int32)
    def calc(case,x,key):
        method,b=case
        if method=='SGD':return estimator(m,x,ids,b,'full',1.)
        return token_estimator(m,x,key,k)
    # Direct pairs at b=4; linear averaging across independent documents gives scaling for other b.
    for it in range(256):
        src=[j.asarray(data[rng.choice(split,4,replace=True)]) for _ in range(2)]
        keys=jax.random.split(jax.random.key(seed+it),2)
        g=calc(('token',4),src[0],keys[0])-calc(('token',4),src[1],keys[1])
        full=calc(('SGD',4),src[0],keys[0])-calc(('SGD',4),src[1],keys[1])
        out['pairs'].append([float(j.vdot(g,g))*.5,float(j.vdot(full,full))*.5])
    ratio=float(np.mean(np.array(out['pairs'])[:,0])/np.mean(np.array(out['pairs'])[:,1]))
    out['variance_ratio_at_fixed_b']=ratio
    print('TOKEN VARIANCE',k,ratio,flush=True)
    fs.pipe(ROOT+f'/positions/attempts/k{k}-s{seed}-{ident}.json',json.dumps(out).encode())
    run.log({'variance/relative':ratio})
    cases=[('SGD',b) for b in (4,8,12,16,24,32,48,64)] + [('token',b) for b in (8,16,32,48,64,96,128)]
    sizes=set(b for _,b in cases)
    samples=[]
    for i in range(42):
        indices=rng.choice(split,128,replace=True)
        samples.append(({b:j.asarray(data[indices[:b]]) for b in sizes},jax.random.key(seed+1000+i)))
    ready(samples)
    for c in cases:
        for pool,key in samples[:2]:ready(calc(c,pool[c[1]],key))
    times=defaultdict(list)
    for pool,key in samples[2:]:
        for q in rng.permutation(len(cases)):
            case=cases[q];x=pool[case[1]]
            t=time.perf_counter();ready(calc(case,x,key));times[case].append(1000*(time.perf_counter()-t))
    for (name,b),ms in times.items():out['rows'].append(dict(method=name,b=b,ms=ms,variance_vs_single=(ratio if name=='token' else 1)/b))
    fs.pipe(ROOT+f'/positions/k{k}-s{seed}.json',json.dumps(out).encode());run.finish()
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

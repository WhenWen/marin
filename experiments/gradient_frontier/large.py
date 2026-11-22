"""Shared-microbatch measurements for global batches through 256 sequences."""
from collections import defaultdict
import functools,hashlib,json,os,sys,time
from pathlib import Path
import equinox as eqx,fsspec,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .run import ROOT,ready
from .multifidelity import shortlist

@eqx.filter_jit
def step(state,data,ids,key,c,micro):
    count=data[0].shape[0]//micro
    vals=jax.tree.map(lambda x:x.reshape((count,micro,*x.shape[1:])),data)
    keys=jax.random.split(key,count)
    def grad(d,k):return eqx.filter_grad(objective)(state.model,d,ids,k,c)
    first=grad(jax.tree.map(lambda x:x[0],vals),keys[0])
    if count>1:
        def add(g,k):
            part=grad(jax.tree.map(lambda x:x[k],vals),keys[k])
            return jax.tree.map(j.add,g,part),None
        first,_=jax.lax.scan(add,first,j.arange(1,count))
        first=jax.tree.map(lambda x:x/count,first)
    return state.take_step(first,key=key)[0]

def main(s,freq,n,v,flash,micro):
    if flash:
        from .fused_attention import install
        install()
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(s,fs)
    pp=ROOT+'/inputs/native-packed';f=Path('/tmp/native-large.npz');fs.get_file(pp+'.npz',str(f))
    meta=json.loads(fs.cat_file(pp+'.json'));assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
    with np.load(f) as z:data=tuple(z[k] for k in ('tokens','weights','segments'))
    cal=j.asarray(data[0][:512]);hold=tuple(z[512:] for z in data)
    ids=ready(shortlist(cal,freq,128256));prep=[]
    for _ in range(24):
        t=time.perf_counter();ready(shortlist(cal,freq,128256));prep.append(1000*(time.perf_counter()-t))
    seed=24001+s+micro;rng=np.random.default_rng(seed)
    ident=os.urandom(5).hex();os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='large-global-batch',
        id=ident,name=f'{s}-k{freq}-n{n}-v{v}-flash{flash}-micro{micro}',config=dict(seed=seed,origin=src,micro=micro))
    cfg=(100000*v+freq,n);base=(0,0)
    sizes=[b for b in (16,24,32,48,64,80,96,112,128,160,192,224,256) if b%micro==0 and b>=micro]
    cases=[(c,b) for c in (base,cfg) for b in sizes]
    sam=[]
    for _ in range(18):
        idx=rng.integers(len(hold[0]),size=max(sizes))
        sam.append((tuple(j.asarray(a[idx]) for a in hold),jax.random.PRNGKey(int(rng.integers(2**30)))))
    ready(sam)
    def execute(c,b,pool,key):return step(state,jax.tree.map(lambda x:x[:b],pool),ids,key,c,micro)
    for c,b in cases:ready(execute(c,b,*sam[0]))
    times=defaultdict(list)
    for pool,key in sam[1:]:
        for i in rng.permutation(len(cases)):
            c,b=cases[i];t=time.perf_counter();ready(execute(c,b,pool,key))
            times[(c,b)].append(1000*(time.perf_counter()-t))
    out=dict(step=s,k=freq,n=n,v=v,flash=flash,micro=micro,prep_ms=prep,wandb=run.url,
        timing=[dict(method='base' if c==base else 'candidate',b=b,ms=ms) for (c,b),ms in times.items()])
    fs.pipe(ROOT+f'/large/k{freq}-n{n}-v{v}-f{flash}-m{micro}-s{s}.json',json.dumps(out).encode())
    run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

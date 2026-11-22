"""Test a gradient-prediction control across documents, coupling exact corrections to a nested subset."""
import json,os,sys,time
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .run import ROOT,ready
from .fused_attention import install


def gold(m,data,key,bank):
    return objective(m,data,bank,key,(113*100000+4096,64))


def cheap(m,data,key,bank,kind,k):
    return objective(m,data,bank,key,(kind*100000+k,0))


@eqx.filter_jit
def diagnose(model,a,b,key,high,low,kind,k):
    keys=jax.random.split(key)
    def get(d,r):
        return (eqx.filter_grad(gold)(model,d,r,high),
                eqx.filter_grad(cheap)(model,d,r,low,kind,k))
    x,y=get(a,keys[0]);xx,yy=get(b,keys[1])
    pairs=[]
    for ga,ha,gb,hb in zip(*map(jax.tree.leaves,(x,y,xx,yy))):
        dg=ga-gb;dh=ha-hb
        pairs.append(.5*j.stack((j.vdot(dg,dg),j.vdot(dh,dh),j.vdot(dg,dh))))
    return j.stack(pairs)


def estimate(model,data,key,high,low,kind,k,b,gamma):
    M=data[0].shape[0]
    small=jax.tree.map(lambda v:v[:b],data)
    extra=jax.tree.map(lambda v:v[b:],data)
    lam=gamma*(1-b/M)
    return gold(model,small,key,high)+lam*(cheap(model,extra,key,low,kind,k)-cheap(model,small,key,low,kind,k))


@eqx.filter_jit
def update(state,data,key,high,low,kind,k,b,gamma):
    g=eqx.filter_grad(estimate)(state.model,data,key,high,low,kind,k,b,gamma)
    return state.take_step(g,key=key)[0]


def main(step,kind,k,b,M,rep):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    from .group_regression import fit as fit_gold,levels as hg
    if kind<100:from .learned_adjoint import fit,levels
    elif kind<120:from .group_regression import fit,levels
    else:from .product_control import fit,levels
    initial=state if step==4001 else recover(4001,fs)[0]
    p=Path('/tmp/doc-control.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(p))
    with np.load(p) as z:train=tuple(j.asarray(z[a][:max(levels(kind)[0],hg(113)[0])]) for a in ('tokens','weights','segments'))
    def prepare():
        hi=fit_gold(initial.model,jax.tree.map(lambda v:v[:hg(113)[0]],train),4096,113)
        if (kind,k)==(113,4096):return hi,hi
        return hi,fit(initial.model,jax.tree.map(lambda v:v[:levels(kind)[0]],train),k,kind)
    high,low=ready(prepare());prep=[]
    for _ in range(8):
        t=time.perf_counter();ready(prepare());prep.append(1000*(time.perf_counter()-t))
    fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(p))
    with np.load(p) as z:hold=tuple(z[a] for a in ('tokens','weights','segments'))
    rng=np.random.default_rng(3613+step+rep)
    conf=dict(step=step,kind=kind,k=k,b=b,M=M,rep=rep,source=src)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='document-level-learned-controls',id=os.urandom(5).hex(),config=conf)
    pairs=[];ms=[]
    def draw(n):
        ix=rng.integers(len(hold[0]),size=n)
        return tuple(j.asarray(a[ix]) for a in hold)
    pars=[jax.tree_util.keystr(p) for p,_ in jax.tree_util.tree_leaves_with_path(eqx.filter(state.model,eqx.is_inexact_array))]
    for i in range(96):
        key=jax.random.key(2297+i+step);a,bb=draw(4),draw(4)
        pairs.append(np.asarray(diagnose(state.model,a,bb,key,high,low,kind,k)).tolist())
        data=ready(draw(M))
        if i==0:ready(update(state,data,key,high,low,kind,k,b,1.))
        t=time.perf_counter();ready(update(state,data,key,high,low,kind,k,b,1.));ms.append(1000*(time.perf_counter()-t))
    conf.update(device=jax.devices()[0].device_kind,prep=prep,pairs=pairs,ms=ms,params=pars,wandb=run.url)
    fs.pipe(ROOT+f'/doc-control/s{step}-v{kind}-k{k}-b{b}-M{M}-r{rep}.json',json.dumps(conf).encode())
    run.log({'median_ms':np.median(ms)});run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

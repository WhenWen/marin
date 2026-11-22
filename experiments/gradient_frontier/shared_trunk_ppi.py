"""A document-level PPI control and exact-position correction sharing one transformer forward and backward."""
import json
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .fused_attention import install
from .lowmem import objective as strong,stats
from .tail import trunk
from .run import ROOT,ready


def loss(model,data,key,b,k,mixing):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=trunk(m,data);w=m.get_lm_head().rearrange(('vocab','embed')).array
    tok,wt,_=data;y=j.roll(tok,-1,1);M,t=tok.shape
    # All terms depend on this same forward and their weighted adjoints take one backward pass.
    head=x@w[:k].T;pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z,_,_=stats(head,pos,y,.5);cheap=((z-pos)*wt).sum(1)/wt.sum(1)
    ref=strong(x[:b],w,y[:b],wt[:b],key,(61*100000+12288,64))
    return ref+mixing*(cheap.mean()-cheap[:b].mean())


@eqx.filter_jit
def evaluate(state,a,z,key,b,k,gamma):
    u,v=jax.random.split(key)
    g=eqx.filter_grad(loss)(state.model,a,u,b,k,gamma);h=eqx.filter_grad(loss)(state.model,z,v,b,k,gamma)
    return sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(g),jax.tree.leaves(h)))*.5


@eqx.filter_jit
def train_step(state,a,key,b,k,gamma):
    g=eqx.filter_grad(loss)(state.model,a,key,b,k,gamma)
    return state.take_step(g,key=key)[0]


def main(step,b,M,k):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    f=Path('/tmp/fused-doc-ppi.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:hold=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    seed=54616+step+b+k;rng=np.random.default_rng(seed)
    sizes=sorted({max(2,b//2),b,2*b,3*b,4*b,6*b,8*b})
    def draw():
        i=rng.integers(len(hold[0]),size=max(M,max(sizes)))
        return {B:tuple(j.asarray(v[i[:B]]) for v in hold) for B in set(sizes+[M])}
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,b=b,M=M,k=k,seed=seed,checkpoint=src)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='shared-trunk-document-PPI',id=os.urandom(5).hex(),config=cfg)
    base=[(m,B) for m in (61,62,7) for B in sizes];gammas=(.25,.5,.75,1.)
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in base]
    cand=[dict(gamma=g,ms=[],pair=[]) for g in gammas]
    out=dict(**cfg,device=jax.devices()[0].device_kind,wandb=run.url,rows=rows,candidates=cand)
    dst=ROOT+f'/shared-trunk-ppi/s{step}-b{b}-M{M}-k{k}.json';dummy=j.arange(1)
    for i in range(37):
        a,z=ready((draw(),draw()));key=jax.random.key(seed+i)
        for ix in rng.permutation(np.arange(-len(cand),len(rows))):
            start=time.perf_counter()
            if ix<0:ready(train_step(state,a[M],key,b,k,gammas[-ix-1]))
            else:
                m,B=base[ix];ready(update(state,a[B],key,dummy,m,12288 if m==61 else 16384,64))
            t=(time.perf_counter()-start)*1000
            if i:(cand[-ix-1] if ix<0 else rows[ix])['ms'].append(t)
        if not i:continue
        for r in cand:r['pair'].append(float(evaluate(state,a[M],z[M],key,b,k,r['gamma'])))
        for r,(m,B) in zip(rows,base):r['pair'].append(float(pair(state,a[B],z[B],key,dummy,m,12288 if m==61 else 16384,64)))
        if i%6==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':36});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

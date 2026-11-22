"""Use independent, unrevealed anchor panels for ten gradient corrections at changing model states."""
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
from .fused_attention import install
from .head_to_head import update
from .native_eval import objective,recover
from .run import ROOT,ready
from .tail import trunk


def cheap(model,d,k):
    m=jax.tree.map(lambda q:q.astype(j.bfloat16),model)
    h=trunk(m,d);w=m.get_lm_head().rearrange(('vocab','embed')).array;y=j.roll(d[0],-1,1)
    pos=j.einsum('btd,btd->bt',h,w[y]).astype(j.float32)
    logit=(h@w[:k].T).astype(j.float32)
    z=jax.nn.logsumexp(logit,-1)
    tail=jax.nn.logsumexp(logit[:,:,k//2:],-1)+j.log(1.5)
    z=j.logaddexp(j.logaddexp(z,tail),j.where(y>=k,pos,-j.inf))
    return ((z-pos)*d[1]).sum()/d[1].sum()


def anchor(model,d,k,rep):
    if k>0:return eqx.filter_grad(cheap)(model,d,k)
    return eqx.filter_grad(objective)(model,d,j.arange(1),rep,(61*100000+12288,64))


def gold(model,d,key):
    return eqx.filter_grad(objective)(model,d,j.arange(1),key,(61*100000+12288,64))


@eqx.filter_jit
def prepare(model,panels,k,keys):
    # Each group uses the SAME microbatch shape, including the group later revealed as the training draw.
    # Return exactly ten independent differences. The training loop reads only the slot for its current step.
    R,M,b,t=panels[0].shape
    def one(panel,key):
        ks=jax.random.split(key,M)
        def at(i):
            return anchor(model,jax.tree.map(lambda a:a[i],panel),k,ks[i])
        first=at(0)
        total=jax.lax.fori_loop(2,M,lambda i,s:jax.tree.map(j.add,s,at(i)),at(1))
        return jax.tree.map(lambda s,h:(s-(M-1)*h)/M,total,first)
    # lax.map bounds peak memory for the ten separate, dense 34M-parameter corrections.
    return jax.lax.map(lambda v:one(*v),(panels,keys))


@eqx.filter_jit
def prepare_fast(model,panels,k,keys,fast):
    R,M,b,t=panels[0].shape
    def one(panel,key):
        def get(d,ke):
            g=anchor(model,d,k,ke)
            return jax.tree.map(lambda a:j.empty((0,),j.float32) if a.size>2_000_000 else a.astype(j.bfloat16).astype(j.float32),g)
        gs=jax.vmap(get)(panel,jax.random.split(key,M))
        return jax.tree.map(lambda v:(v[1:].sum(0)-(M-1)*v[0])/M,gs)
    # Small frozen calibration moments can be acquired in parallel; all ten disjoint slots are charged.
    if fast==2:
        grouped=jax.tree.map(lambda x:x.reshape(R//2,2,*x.shape[1:]),(panels,keys))
        result=jax.lax.map(lambda v:jax.vmap(one)(*v),grouped)
        return jax.tree.map(lambda x:x.reshape(R,*x.shape[2:]),result)
    return jax.lax.map(lambda v:one(*v),(panels,keys))


@eqx.filter_jit
def timed(state,d,delta,key,gamma):
    g=gold(state.model,d,key)
    grad=jax.tree.map(lambda a,b:a if b.size==0 else a+gamma*b,g,delta)
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def stats(state,start,a,b,key,k,fast):
    ka,kb,kc,kd=jax.random.split(key,4)
    ga,gb=gold(state.model,a,ka),gold(state.model,b,kb)
    ha,hb=anchor(start.model,a,k,kc),anchor(start.model,b,k,kd)
    def val(x,y,u,v):
        g=x-y
        if fast and u.size>2_000_000:return .5*j.stack((j.vdot(g,g),j.array(0.),j.array(0.)))
        h=u.astype(j.bfloat16).astype(j.float32)-v.astype(j.bfloat16).astype(j.float32) if fast else u-v
        return .5*j.stack((j.vdot(g,g),j.vdot(h,h),j.vdot(g,h)))
    return sum(val(*row) for row in zip(*map(jax.tree.leaves,(ga,gb,ha,hb))))


def main(step,b,M,k,fast=0):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    start=state if step==4001 else recover(4001,fs)[0]
    tmp=Path('/tmp/panel-hold.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[name][512:] for name in ('tokens','weights','segments'))
    seed=42277+step+k+b;gen=np.random.default_rng(seed)
    def draw(shape):
        ix=gen.integers(len(pool[0]),size=shape)
        return tuple(j.asarray(v[ix]) for v in pool)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,b=b,M=M,k=k,fast=fast,checkpoint=src,seed=seed)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='ten-independent-frozen-panels',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,B) for m in (61,62,7) for B in (b//2,b,2*b,3*b,4*b,6*b)]
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in cases]
    out=dict(**cfg,wandb=run.url,device=jax.devices()[0].device_kind,rows=rows,components=[],ms=[],prep=[])
    dest=ROOT+f'/frozen-panel/s{step}-b{b}-ratio{M}-k{k}'+(f'-fast{fast}' if fast else '')+'.json'
    dummy=j.arange(1)
    for it in range(49):
        key=jax.random.key(seed+it)
        if it%8==0:
            panels=ready(draw((10,M,b)));keys=jax.random.split(key,10)
            t=time.perf_counter();all_delta=ready(prepare_fast(start.model,panels,k,keys,fast) if fast else prepare(start.model,panels,k,keys));elapsed=1000*(time.perf_counter()-t)
            if it:out['prep'].append(elapsed)
        slot=it%8
        # Stage cache access outside the clock once, then charge its real cost in the candidate operation below.
        left,right=ready((draw(6*b),draw(6*b)))
        l,r=ready(tuple({B:tuple(v[:B] for v in batch) for _,B in cases} for batch in (left,right)))
        for ix in gen.permutation(np.arange(-1,len(cases))):
            t=time.perf_counter()
            if ix==-1:
                d=jax.tree.map(lambda v:v[slot,0],panels);cor=jax.tree.map(lambda v:v[slot],all_delta)
                ready(timed(state,d,cor,key,1.))
            else:
                m,B=cases[ix];ready(update(state,l[B],key,dummy,m,12288 if m==61 else 16384,64))
            ms=1000*(time.perf_counter()-t)
            if it:
                if ix==-1:out['ms'].append(ms)
                else:rows[ix]['ms'].append(ms)
        if not it:continue
        out['components'].append(np.asarray(stats(state,start,l[b],r[b],key,k,fast)).tolist())
        for row,(m,B) in zip(rows,cases):
            row['pair'].append(float(pair(state,l[B],r[B],key,dummy,m,12288 if m==61 else 16384,64)))
        if it%8==0:fs.pipe(dest,json.dumps(out).encode())
    run.log({'panel_preparation_ms':np.median(out['prep'])});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

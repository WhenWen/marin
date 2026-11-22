"""Fuse MuonW with two independent inverse-probability corrections, each at its actual batch size."""
import json
import os
import sys
import time
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import debias, fused_attention
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT, ready
from .subspace_forecast import components, native_audit
from .temporal import momentum


def forecast(state, scale):
    return jax.tree.map(lambda x: -scale*x if x.size<2_000_000 else j.zeros_like(x),momentum(state))


@eqx.filter_jit
def regular(state,data,key,k,n):
    return debias.draw(state,data,key,k,n,61,0,j.arange(1))


@eqx.filter_jit
def residual(state,data,key,k,n,truth):
    _,avg=debias.audit(state,data,key,k,n,61,j.arange(1))
    return jax.tree.map(j.subtract,truth,avg)


@eqx.filter_jit
def advance(state,data,delta,key,k,n,scale,p,r):
    grad=forecast(state,scale)
    if data is not None:
        raw=debias.draw(state,data,key,k,n,61,0,j.arange(1))
        grad=jax.tree.map(lambda v,g:v+(g-v)/p,grad,raw)
    if delta is not None:
        grad=jax.tree.map(lambda v,d:v+d/r,grad,delta)
    return state.take_step(grad,key=key)[0]


def main(step,B,k,n,scale):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    pool_file=Path('/tmp/independent-audits.npz')
    fs.get_file(ROOT+'/inputs/native-packed.npz',str(pool_file))
    with np.load(pool_file) as z:pool=tuple(z[name][512:] for name in ('tokens','weights','segments'))
    original=Attention.__call__;fused_attention.install()
    def delta(data,key):
        Attention.__call__=original
        try:gold=native_audit(state.model,data,B)
        finally:fused_attention.install()
        return residual(state,data,key,k,n,gold)
    seed=738410+step+B+k+n+scale;rng=np.random.default_rng(seed)
    def sample(size):
        ix=rng.integers(len(pool[0]),size=size)
        return tuple(j.asarray(a[ix]) for a in pool)
    cases=[(m,kb,nb,b) for m,kb,nb in ((61,4096,16),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in (1,2,3,4,6,8,12,16,24,32)]
    rows=[dict(method=m,k=kb,n=nb,b=b,ms=[],pair=[]) for m,kb,nb,b in cases]
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,B=B,k=k,n=n,scale=scale,signed_momentum_coefficient=-scale,checkpoint=src,seed=seed,device=jax.devices()[0].device_kind)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='independent-precision-and-gradient-audits',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,times=[],components=[],rows=rows)
    path=ROOT+f'/independent-audits/s{step}-b{B}-k{k}-n{n}-c{scale}.json'
    pred=ready(forecast(state,scale));dummy=j.arange(1)
    for it in range(41):
        a,b,extra,extra2,u,v=ready(tuple(sample(s) for s in (B,B,B,B,32,32)))
        left,right=ready(tuple({bs:tuple(x[:bs] for x in d) for _,_,_,bs in cases} for d in (u,v)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(seed+it),4)
        times=[0.]*4
        def one(ix):
            if ix>=0:
                m,kv,nv,bs=cases[ix]
                return update(state,left[bs],ka,dummy,m,kv,nv)
            tag=-ix-1
            corr=delta(extra,kc) if tag&2 else None
            return advance(state,a if tag&1 else None,corr,ka,k,n,scale,.125,.015625)
        for ix in rng.permutation(np.arange(-4,len(cases))):
            start=time.perf_counter();ready(one(ix));elapsed=1000*(time.perf_counter()-start)
            if it:
                if ix<0:times[-ix-1]=elapsed
                else:rows[ix]['ms'].append(elapsed)
        if not it:continue
        risk=components(regular(state,a,ka,k,n),regular(state,b,kb,k,n),pred)
        error=norm_moments(delta(extra,kc),delta(extra2,kd))
        out['components'].append(np.asarray(j.concatenate((risk,error))).tolist());out['times'].append(times)
        for row,(m,kv,nv,bs) in zip(rows,cases):
            row['pair'].append(float(pair(state,left[bs],right[bs],ka,dummy,m,kv,nv)))
        if it%5==0:fs.pipe(path,json.dumps(out).encode())
    run.log({'cycles':40});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

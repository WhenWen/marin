"""Matched ablation with rare unmodified native full-vocabulary gradients and no precision audit."""
import functools
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
from .diagnostic_artifacts import publish
from . import fused_attention
from .contest import pair
from .independent_audits import forecast,donated_baseline,copy_state
from .native_eval import recover
from .subspace_forecast import components,native_audit
from .unbiased_check import native
from .run import ROOT,ready


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(args,state):
    d,key,scale,p=args
    pred=forecast(state,scale)
    if d is not None:
        gold=eqx.filter_grad(native)(state.model,d)
        pred=jax.tree.map(lambda x,g:x+(g-x)/p,pred,gold)
    return state.take_step(pred,key=key)[0]


def main(step,B,scale):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    original=Attention.__call__;fused_attention.install()
    tmp=Path('/tmp/pure-roulette.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[k][512:] for k in ('tokens','weights','segments'))
    rng=np.random.default_rng(163871+step+B)
    def sample():
        ix=rng.integers(len(pool[0]),size=max(64,B))
        return tuple(j.asarray(a[ix]) for a in pool)
    cases=[(m,k,n,b) for m,k,n in ((61,64,4),(61,256,8),(61,4096,32),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in (1,2,3,4,6,8,12,16,24,32,48,64)]
    rows=[dict(method=m,k=k,n=n,b=b,pair=[],ms=[]) for m,k,n,b in cases]
    cfg=dict(step=step,B=B,scale=scale,checkpoint=src,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='thinned-native-CE-donated',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,times=[],components=[]);dummy=j.arange(1)
    dst=ROOT+f'/exact-roulette/s{step}-B{B}-c{scale}.json'
    pred=ready(forecast(state,scale))
    for i in range(41):
        u,v=ready((sample(),sample()))
        a,b=ready(tuple({bs:tuple(x[:bs] for x in z) for bs in {B,*(c[-1] for c in cases)}} for z in (u,v)))
        key=jax.random.key(1321+i);ts=[0.,0.]
        for ix in rng.permutation(np.arange(-2,len(cases))):
            work=copy_state(state)
            if ix<0:Attention.__call__=original
            tick=time.perf_counter()
            if ix<0:ready(advance((a[B] if ix==-2 else None,key,scale,.2),work))
            else:
                m,k,n,bs=cases[ix];ready(donated_baseline((a[bs],key,dummy,m,k,n),work))
            ms=1000*(time.perf_counter()-tick)
            if ix<0:fused_attention.install()
            if i:
                if ix<0:ts[-ix-1]=ms
                else:rows[ix]['ms'].append(ms)
        if not i:continue
        Attention.__call__=original
        try:mm=components(native_audit(state.model,a[B],B),native_audit(state.model,b[B],B),pred)
        finally:fused_attention.install()
        out['components'].append(np.asarray(mm).tolist());out['times'].append(ts)
        for row,(m,k,n,bs) in zip(rows,cases):row['pair'].append(float(pair(state,a[bs],b[bs],key,dummy,m,k,n)))
        if i%5==0:publish(fs,dst,out)
    run.log({'cycles':40});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

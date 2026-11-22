"""Measure actual independent-coin selection and data gathering against native donated-buffer baselines."""
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
from . import fused_attention
from .independent_audits import copy_state,donated_baseline,residual
from .roulette_training import update
from .subspace_forecast import native_audit
from .native_eval import recover
from .contest import pair
from .run import ROOT,ready


def main(step,host):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    original=Attention.__call__;fused_attention.install()
    path=Path('/tmp/live-gate.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(path))
    with np.load(path) as z:pool=tuple(z[k] for k in ('tokens','weights','segments'))
    rng=np.random.default_rng(5528+step+host)
    def draw():
        ids=rng.integers(len(pool[0]),size=128);return tuple(j.asarray(v[ids]) for v in pool)
    cases=[(m,k,n,b) for m,k,n in ((61,64,4),(61,1024,16),(61,8192,32),(61,12288,64),(62,16384,64))
           for b in (1,2,3,4,6,8,12,16)]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    cfg=dict(step=step,host=host,B=32,p=.2,r=.002,seed=5528+step+host,checkpoint=src,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='end-to-end-native-gating-overhead',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,branches=[],live=[]);dummy=j.arange(1)
    dst=ROOT+f'/live-gate/s{step}-host{host}.json'
    p,r=.2,.002
    def candidate(work,data,opkey,force):
        # All selection, RNG work, dispatch and synchronization are in the timer.
        key,nextkey=jax.random.split(opkey)
        coin1,coin2,selection,gradkey,correctionkey=jax.random.split(key,5)
        if host:
            first,second=rng.random(2)<(p,r)
        else:
            first=bool(jax.random.bernoulli(coin1,p));second=bool(jax.random.bernoulli(coin2,r))
        if force is not None:first,second=bool(force&1),bool(force&2)
        ids=jax.random.permutation(selection,128) if first or second else None
        a=tuple(v[ids[:32]] for v in data) if first else None
        d=None
        if second:
            ext=tuple(v[ids[32:64]] for v in data)
            Attention.__call__=original
            try:truth=native_audit(work.model,ext,32)
            finally:fused_attention.install()
            d=residual(work,ext,correctionkey,12288,64,truth)
        ready(update((a,d,gradkey,nextkey,p,r,0),work))
        return int(first)+2*int(second)
    for i in range(65):
        u,v=ready((draw(),draw()))
        L,R=ready(tuple({bs:tuple(a[:bs] for a in d) for _,_,_,bs in cases} for d in (u,v)))
        times=[0.]*4
        for ix in rng.permutation(np.arange(-5,len(cases))):
            opkey=ready(jax.random.key(77631+(i+1)*170+int(ix)))
            work=copy_state(state)
            tic=time.perf_counter()
            if ix<0:branch=candidate(work,u,opkey,None if ix==-5 else -ix-1)
            else:
                m,k,n,bs=cases[ix]
                gkey,_=jax.random.split(opkey)
                ready(donated_baseline((L[bs],gkey,dummy,m,k,n),work))
            ms=1000*(time.perf_counter()-tic)
            if i:
                if ix==-5:out['live'].append([branch,ms])
                elif ix<0:times[-ix-1]=ms
                else:rows[ix]['ms'].append(ms)
        if not i:continue
        out['branches'].append(times)
        for row,(m,k,n,bs) in zip(rows,cases):
            row['pair'].append(float(pair(state,L[bs],R[bs],jax.random.key(77631+i),dummy,m,k,n)))
        if i%8==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':64});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

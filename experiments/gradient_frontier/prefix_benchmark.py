"""Randomized same-card benchmark of exact prefix packing, including original MuonW updates and donation."""
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
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .contest import pair
from .diagnostic_artifacts import publish
from .fused_attention import install
from .independent_audits import copy_state,donated_baseline
from .native_eval import recover
from . import prefix_panels,prefix_learning
from .run import ROOT,ready


@functools.partial(eqx.filter_jit,donate='all-except-first')
def packed(args,state):
    data,key,K,k,n,bank=args
    return prefix_panels.update(state,data,key,K,k,n) if bank is None else prefix_learning.update(state,data,key,K,k,n,bank)


def main(step,B,K,mode=0):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);install()
    file=Path('/tmp/prefix-panels.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(file))
    with np.load(file) as z:
        pool=tuple(z[name][512:] for name in ('tokens','weights','segments'))
        cal=tuple(j.asarray(z[name][:8]) for name in ('tokens','weights','segments')) if mode==2 else None
    bank=None;prep=[0.]
    if mode==1:bank=ready(prefix_learning.uniform(K))
    if mode==2:
        initial=state if step==4001 else recover(4001,fs)[0]
        bank=ready(prefix_learning.fit(initial.model,cal,K));prep=[]
        for _ in range(8):
            tic=time.perf_counter();bank=ready(prefix_learning.fit(initial.model,cal,K));prep.append(1000*(time.perf_counter()-tic))
        assert np.max(np.abs(np.asarray(bank).sum(0)/K-1))<1.e-4
    seed=811352+step+B+K
    gen=np.random.default_rng(seed)
    cases=[(1,k,n,B) for k,n in ((0,0),(2048,8),(4096,16),(8192,32),(12288,64))]
    cases.extend((m,k,n,b) for m,k,n in ((61,4096,32),(61,8192,32),(61,8192,64),(61,12288,64),(62,16384,64),(7,16384,64))
                 for b in sorted({max(1,B//8),B//4,B//2,3*B//4,B,B*5//4}))
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    cfg=dict(step=step,B=B,K=K,mode=mode,seed=seed,checkpoint=src,device=jax.devices()[0].device_kind,holdout='native-packed')
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='exact-causal-prefix-panels',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,prep=prep,bank=None if bank is None else np.asarray(bank).tolist())
    dest=ROOT+f'/prefix-panels/s{step}-b{B}-K{K}.json'
    if mode:dest=ROOT+f'/overlap-prefix/s{step}-b{B}-K{K}-mode{mode}.json'
    sizes=sorted({q[-1] for q in cases});dummy=j.arange(1)
    def sample():
        ii=gen.integers(len(pool[0]),size=max(sizes));return {b:tuple(j.asarray(v[ii[:b]]) for v in pool) for b in sizes}
    for i in range(37):
        a,b=ready((sample(),sample()));key=jax.random.key(seed+i)
        for ix in gen.permutation(len(cases)):
            m,k,n,sz=cases[ix]
            work=copy_state(state);tic=time.perf_counter()
            if m==1:ready(packed((a[sz],key,K,k,n,bank),work))
            else:ready(donated_baseline((a[sz],key,dummy,m,k,n),work))
            ms=1000*(time.perf_counter()-tic)
            if m==1:
                v=float(prefix_panels.measure(state,a[sz],b[sz],key,K,k,n) if bank is None else prefix_learning.measure(state,a[sz],b[sz],key,K,k,n,bank))
            else:v=float(pair(state,a[sz],b[sz],key,dummy,m,k,n))
            if i:rows[ix]['ms'].append(ms);rows[ix]['pair'].append(v)
        if i%6==0:publish(fs,dest,out)
    run.log({'cycles':36});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

"""Predict the document tail gradient by tensor and buy its residual with a positive probability."""
import functools
import os
import subprocess
import sys
import time

import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention,prepared_blocks as prep,segment_queries as seg
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .independent_audits import copy_state
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready
from .temporal import momentum


def block(model,data,layout,cut,k,ph,i):
    entry=layout[i]
    if entry is None:return jax.tree.map(j.zeros_like,eqx.filter(model,eqx.is_inexact_array))
    table,offset,at=entry
    v=seg.raw(model,data,table[at],offset[at],ph,1024 if i else cut,k,cut if i else 0)
    return seg.scale(v,len(table))


@eqx.filter_jit
def fit(state,a,b):
    def coef(p,g,h):
        n=j.vdot(p,p)
        return j.clip(.5*j.vdot(p,(g+h)*.5)/j.maximum(n,1e-12),-4.,4.)
    return jax.tree.map(coef,momentum(state),a,b)


@eqx.filter_jit
def risk(state,coef,g):
    return sum(j.vdot(d,d) for d in jax.tree.leaves(jax.tree.map(lambda x,c,p:x-c*p,g,coef,momentum(state))))


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    prefix,tail,coef,delta=inputs
    prior=jax.tree.map(j.multiply,coef,momentum(state))
    if tail is not None:prior=jax.tree.map(lambda p,t:p+(t-p)*4.,prior,tail)
    g=jax.tree.map(j.add,prefix,prior)
    if delta is not None:g=jax.tree.map(lambda a,b:a+b/.002,g,delta)
    return state.take_step(g,key=jax.random.split(state.training_key)[1])[0]


def main(step,cut,batch,k):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    state,src=recover(step,fs);fused_attention.install()
    board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg=dict(**meta,step=step,cut=cut,batch=batch,k=k,source=src,board=board,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=os.urandom(6).hex(),group='learned-document-tail',mode='online',config=cfg)
    phases=tuple(j.asarray(i,j.int32) for i in range(16))
    cases=[(m,kv,np0,b,0) for m,kv,np0 in ((61,4096,32),(61,12288,64),(62,16384,64)) for b in (1,2,4,8,12,16,24,32)]
    out=dict(**cfg,wandb=run.url,parts=[],prices=[],rows=[dict(config=c,pairs=[],ms=[]) for c in cases])
    for i in range(26):
        rng=np.random.default_rng(np.random.SeedSequence([418593,step,i]))
        host=[tuple(x[ids] for x in pool) for ids in rng.integers(len(pool[0]),size=(4,32))]
        panels=ready([tuple(j.asarray(v) for v in h0) for h0 in host])
        tic=time.perf_counter()
        plans=ready([seg.plan(h0[2],cut,batch,rng,True,True) for h0 in host])
        choices=[tuple(phases[int(rng.integers(16))] for _ in range(2)) for h0 in host]
        overhead=1000*(time.perf_counter()-tic)/4
        tic=time.perf_counter()
        tails=[ready(block(state.model,d,l,cut,k,ph[1],1)) for d,l,ph in zip(panels[:2],plans,choices)]
        if i<=1:
            co=ready(fit(state,*tails));out['cal_ms']=1000*(time.perf_counter()-tic)+2*overhead
        gs=[seg.plus(block(state.model,d,l,cut,k,ph[0],0),t) for d,l,ph,t in zip(panels,plans,choices,tails)]
        deltas=[ready(seg.correction(state.model,d,l,cut,k,phases,True)) for d,l in zip(panels[2:],plans[2:])]
        stats=np.asarray(norm_moments(*gs)).tolist()+np.asarray(norm_moments(*deltas)).tolist()
        stats.append(.5*sum(float(risk(state,co,t)) for t in tails))
        stats.append(float(norm_moments(*tails)[0]))
        sizes=sorted({c[3] for c in cases})
        left,right=ready([{n:tuple(v[:n] for v in d) for n in sizes} for d in panels[:2]])
        salt=j.asarray(i+63519,j.uint32);ms=np.zeros(4)
        for ix in rng.permutation(np.arange(-4,len(cases))):
            ix=int(ix);work=copy_state(state);t=time.perf_counter()
            if ix<0:
                tag=-ix-1
                pre=block(work.model,panels[0],plans[0],cut,k,choices[0][0],0)
                tail=block(work.model,panels[0],plans[0],cut,k,choices[0][1],1) if tag%2 else None
                corr=seg.correction(work.model,panels[2],plans[2],cut,k,phases,True) if tag>=2 else None
                ans=advance((pre,tail,co,corr),work)
            else:
                m,kv,n,b,_=cases[ix];ans=prep.baseline((left[b],salt,m,kv,n),work)
            ready(ans);elapsed=1000*(time.perf_counter()-t)
            if ix<0:ms[tag]=elapsed+overhead*(1+(tag>=2))
            elif i>=2:out['rows'][ix]['ms'].append(elapsed)
        if i>=2:
            out['parts'].append(stats);out['prices'].append(ms.tolist())
            for row,(m,kv,n,b,_) in zip(out['rows'],cases):row['pairs'].append(float(pair(state,left[b],right[b],jax.random.key(76139+i),j.arange(1),m,kv,n)))
            if i%2 or i==2:publish(fs,ROOT+f'/segment-tail-learning/s{step}-l{cut}-b{batch}-k{k}.json',out)
    run.log({'cycles':24});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

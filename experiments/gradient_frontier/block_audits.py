"""Spend separate gradient queries on the vocabulary readout and remaining parameters."""
import json
import functools
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
from . import debias,fused_attention
from .contest import pair
from .head_to_head import update
from .independent_audits import forecast,copy_state,donated_baseline
from .native_eval import recover
from .nested_forecast import norm_moments
from .subspace_forecast import native_audit
from .run import ROOT,ready


def readout(path):
    return '.lm_head.' in jax.tree_util.keystr(path)


def masked_loss(m,data,offset,key,k,n,part):
    m=jax.tree_util.tree_map_with_path(lambda path,x:x if readout(path)==(part==2) else jax.lax.stop_gradient(x),m)
    return debias.sampled(m,data,offset,key,k,n,61,j.arange(1))


def draw(state,data,key,k,n,part,shift):
    width=data[0].shape[1]//n;offset=jax.random.randint(key,(data[0].shape[0],n),0,width)
    return eqx.filter_grad(masked_loss)(state.model,data,(offset+shift)%width,key,k,n,part)


@eqx.filter_jit
def grad(state,d,key,k,n,part):return draw(state,d,key,k,n,part,0)


@eqx.filter_jit
def strict(state,d,key,k,n,gold):
    width=d[0].shape[1]//n
    def combined(i):
        a=draw(state,d,key,k,n,1,i);b=draw(state,d,key,k,n,2,i)
        return jax.tree.map(j.add,a,b)
    total=jax.lax.fori_loop(1,width,lambda i,a:jax.tree.map(j.add,a,combined(i)),combined(0))
    return jax.tree.map(lambda g,h:g-h/width,gold,total)


@eqx.filter_jit
def step(state,data,delta,key,k,n,part,scale,p,r):
    pred=forecast(state,scale)
    if part:
        raw=draw(state,data,key,k,n,part,0)
        pred=jax.tree_util.tree_map_with_path(lambda path,h,g:h+(g-h)/p if readout(path)==(part==2) else h,pred,raw)
    if delta is not None:pred=jax.tree.map(lambda h,d:h+d/r,pred,delta)
    return state.take_step(pred,key=key)[0]


@eqx.filter_jit
def component(x,y,base,part):
    variance=0.;risk=0.
    for (path,a),b,h in zip(jax.tree_util.tree_leaves_with_path(x),jax.tree.leaves(y),jax.tree.leaves(base)):
        if readout(path)!=(part==2):continue
        variance+=.5*j.vdot(a-b,a-b);risk+=j.vdot(a-h,b-h)
    return j.stack((variance,risk))


@functools.partial(eqx.filter_jit,donate='all-except-first')
def donated(args,state):
    return step(state,*args)


def main(st,B,k,n,scale,rep=0):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(st,fs)
    assert any(readout(path) for path,_ in jax.tree_util.tree_leaves_with_path(state.model))
    original=Attention.__call__;fused_attention.install()
    def delta(d,key):
        Attention.__call__=original
        try:g=native_audit(state.model,d,B)
        finally:fused_attention.install()
        return strict(state,d,key,k,n,g)
    f=Path('/tmp/block-audits.npz');pop='native-confirm3' if rep else 'native-packed'
    fs.get_file(ROOT+'/inputs/'+pop+'.npz',str(f))
    with np.load(f) as z:pool=tuple(z[a][0 if rep else 512:] for a in ('tokens','weights','segments'))
    rng=np.random.default_rng(48291+st+B+k)
    def data(size):
        ids=rng.integers(len(pool[0]),size=size);return tuple(j.asarray(v[ids]) for v in pool)
    cases=[(m,kv,nv,b) for m,kv,nv in ((61,256,8),(61,4096,32),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in (1,2,3,4,6,8,12,16,24,32)]
    if rep:
        cases.extend((m,kv,nv,b) for m,kv,nv in ((0,0,0),(61,64,4),(61,1024,16),(61,4096,64)) for b in (1,2,4,8))
        cases.extend((61,kv,nv,b) for kv,nv in ((4096,32),(8192,64),(12288,64)) for b in (40,48,64))
    largest=max(q[-1] for q in cases)
    rows=[dict(method=m,k=kv,n=nv,b=b,ms=[],pair=[]) for m,kv,nv,b in cases]
    cfg=dict(step=st,B=B,k=k,n=n,scale=scale,checkpoint=src,device=jax.devices()[0].device_kind,rep=rep,population=pop)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='readout-and-backbone-query-allocation',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,times=[],components=[])
    dest=ROOT+f'/block-audits/s{st}-B{B}-k{k}-n{n}-c{scale}.json';dummy=j.arange(1);base=ready(forecast(state,scale))
    if rep:dest=ROOT+f'/donated-block-audits/{jax.devices()[0].device_kind.split()[1]}-s{st}-B{B}-c{scale}.json'
    cycles=64 if rep else 32
    for i in range(cycles+1):
        a,b,c,d,U,V=ready(tuple(data(nq) for nq in (B,B,B,B,largest,largest)))
        left,right=ready(tuple({bs:tuple(v[:bs] for v in x) for _,_,_,bs in cases} for x in (U,V)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(735+i),4);ts=[0.]*6
        for ix in rng.permutation(np.arange(-6,len(cases))):
            work=copy_state(state) if rep else state
            tick=time.perf_counter()
            if ix>=0:
                m,kv,nv,bs=cases[ix]
                ready(donated_baseline((left[bs],ka,dummy,m,kv,nv),work) if rep else update(work,left[bs],ka,dummy,m,kv,nv))
            else:
                tag=-int(ix)-1;part=tag%3
                corr=delta(c,kc) if tag>=3 else None
                args=(a if part else None,corr,ka,k,n,part,scale,(.3 if part==1 else .075) if rep else (.2 if part==1 else .05),.002)
                ready(donated(args,work) if rep else step(work,*args))
            elapsed=1000*(time.perf_counter()-tick)
            if i:
                if ix<0:ts[-ix-1]=elapsed
                else:rows[ix]['ms'].append(elapsed)
        if not i:continue
        moments=[component(grad(state,a,ka,k,n,part),grad(state,b,kb,k,n,part),base,part) for part in (1,2)]
        moments.append(norm_moments(delta(c,kc),delta(d,kd)))
        out['components'].append(np.asarray(j.concatenate(moments)).tolist());out['times'].append(ts)
        for row,(m,kv,nv,bs) in zip(rows,cases):row['pair'].append(float(pair(state,left[bs],right[bs],ka,dummy,m,kv,nv)))
        if i%4==0:publish(fs,dest,out)
    run.log({'cycles':cycles});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

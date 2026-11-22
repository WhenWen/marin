"""Randomly audit a complete balanced family against a deterministic full-batch gradient."""
from functools import partial
import json,os,sys,time
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
from .run import ROOT,ready
from .tail import trunk
from .lowmem import objective
from .native_eval import recover
from .unbiased_check import native
from . import fused_attention

def sampled(model,data,offsets,key,k,n,v,bank):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=trunk(m,data)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    if v==113:
        from .group_regression import objective as grouped
        return grouped(x,w,j.roll(data[0],-1,axis=1),data[1],key,(100000*v+k,n),bank,offsets)
    return objective(x,w,j.roll(data[0],-1,axis=1),data[1],key,(100000*v+k,n),offsets)

def draw(state,data,key,k,n,v,offset,bank):
    width=data[0].shape[1]//n
    base=jax.random.randint(key,(data[0].shape[0],n),0,width)
    return eqx.filter_grad(sampled)(state.model,data,(base+offset)%width,key,k,n,v,bank)

@eqx.filter_jit
def common(state,data,key,k,n,v,bank):
    g=draw(state,data,key,k,n,v,0,bank)
    return state.take_step(g,key=key)[0]

@eqx.filter_jit
def grad0(state,data,key,k,n,v,bank):
    return eqx.filter_grad(sampled)(state.model,data,jax.random.randint(key,(data[0].shape[0],n),0,data[0].shape[1]//n),key,k,n,v,bank)

@eqx.filter_jit
def full(state,data):
    return eqx.filter_grad(native)(state.model,data)

@eqx.filter_jit
def audit(state,data,key,k,n,v,bank):
    width=data[0].shape[1]//n
    first=draw(state,data,key,k,n,v,0,bank)
    total=eqx.filter(first,eqx.is_array)
    def add(i,acc):
        g=draw(state,data,key,k,n,v,i,bank)
        return jax.tree.map(j.add,acc,g)
    total=jax.lax.fori_loop(1,width,add,total)
    return first,jax.tree.map(lambda x:x/width,total)

@eqx.filter_jit
def combine(state,first,mean,truth,p,key):
    g=jax.tree.map(lambda x,a,b:x+(b-a)/p,first,mean,truth)
    return state.take_step(g,key=key)[0]

@eqx.filter_jit
def measurements(first,mean,ref,other,othermean,otherref):
    def norm(g):return sum(j.vdot(x,x) for x in jax.tree.leaves(g))
    d=jax.tree.map(j.subtract,ref,mean)
    e=jax.tree.map(j.subtract,otherref,othermean)
    adjusted=jax.tree.map(lambda a,b,u,v:(a-b)+(u-v),first,other,d,e)
    raw=jax.tree.map(j.subtract,first,other)
    return j.array([.5*norm(adjusted),.5*(norm(d)+norm(e)),.5*norm(raw)])

def main(step,k,n,v,B,rep):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    f=Path('/tmp/strict-data.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(f))
    with np.load(f) as z:hold=tuple(z[a] for a in ('tokens','weights','segments'))
    bank=j.arange(1);prep=[]
    if v==113:
        from .group_regression import fit,levels
        path=Path('/tmp/group-strict-train.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(path))
        with np.load(path) as z:train=tuple(j.asarray(z[a][:levels(v)[0]]) for a in ('tokens','weights','segments'))
        initial=state if step==4001 else recover(4001,fs)[0]
        bank=ready(fit(initial.model,train,k,v))
        for _ in range(8):
            t=time.perf_counter();ready(fit(initial.model,train,k,v));prep.append(1000*(time.perf_counter()-t))
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='finite-precision-unbiased',id=os.urandom(5).hex(),config=dict(step=step,k=k,n=n,v=v,batch=B,rep=rep))
    rng=np.random.default_rng(step+55*rep);orig=Attention.__call__
    # Warm each implementation with the original attention active only for the reference function.
    def reference(data):
        Attention.__call__=orig
        try:return full(state,data)
        finally:fused_attention.install()
    def batch():return tuple(j.asarray(a[ix]) for a in hold)
    key=jax.random.PRNGKey(step);ix=rng.integers(len(hold[0]),size=B);data=batch()
    truth=ready(reference(data));first,mean=ready(audit(state,data,key,k,n,v,bank))
    ready(common(state,data,key,k,n,v,bank));ready(combine(state,first,mean,truth,.01,key))
    dst=ROOT+f'/strict/{jax.devices()[0].device_kind.split()[1]}-s{step}-k{k}-n{n}-v{v}-b{B}-r{rep}.json'
    out=dict(step=step,k=k,n=n,v=v,batch=B,rep=rep,source=src,device=jax.devices()[0].device_kind,wandb=run.url,prep=prep,pairs=[],timings=[])
    for ixrep in range(64):
        both=[]
        for jrep in range(2):
            ix=rng.integers(len(hold[0]),size=B);data=ready(batch());key=jax.random.PRNGKey(step+201*ixrep+jrep)
            def normal():
                start=time.perf_counter();ready(common(state,data,key,k,n,v,bank));return 1000*(time.perf_counter()-start)
            def gated():
                start=time.perf_counter()
                truth=reference(data);first,mean=audit(state,data,key,k,n,v,bank)
                ready(combine(state,first,mean,truth,.01,key))
                return (first,mean,truth),1000*(time.perf_counter()-start)
            if rng.random()<.5: ordinary=normal();triplet,rare=gated()
            else:triplet,rare=gated();ordinary=normal()
            both.extend(triplet);out['timings'].append([ordinary,rare])
        out['pairs'].append(np.asarray(measurements(*both)).tolist())
        if (ixrep+1)%8==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'audit/variance_component':np.mean(out['pairs'],0)[1]});run.finish()
if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

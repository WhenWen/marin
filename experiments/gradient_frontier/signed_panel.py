"""Estimate the native mean by revealing one leaf and adding conditionally centered signed controls."""
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
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .subspace_forecast import native_audit
from .tail import trunk
from . import fused_attention
from .run import ROOT,ready


def contrast_loss(model,data,k):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model);x=trunk(m,data)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    # The wide vocabulary parameters have little mean-gradient signal. Skip their backward path in the control.
    w=jax.lax.stop_gradient(w)
    y=j.roll(data[0],-1,1);pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    logits=(x@w[:k].T).astype(j.float32)
    z=j.logaddexp(jax.nn.logsumexp(logits,-1),jax.nn.logsumexp(logits[:,:,k//2:],-1)+j.log(1.5))
    loss=(j.logaddexp(z,j.where(y>=k,pos,-j.inf))-pos)*data[1]
    half=len(y)//2
    return .5*(loss[half:].sum()/data[1][half:].sum()-loss[:half].sum()/data[1][:half].sum())


@eqx.filter_jit
def control(model,panel,coins,k):
    M,b,t=panel[0].shape;offset=j.array(0,j.int32);total=None
    for depth in range(M.bit_length()-1):
        width=M//(2**depth)
        ds=jax.tree.map(lambda a:jax.lax.dynamic_slice_in_dim(a,offset,width).reshape(width*b,t),panel)
        # D depends only on the node and data. Its own branch sign is sampled independently.
        g=eqx.filter_grad(contrast_loss)(model,ds,k)
        g=jax.tree.map(lambda v:v if v.size<2_000_000 else j.empty((0,),j.float32),g)
        sign=1.-2.*coins[depth]
        total=jax.tree.map(lambda v:sign*v,g) if total is None else jax.tree.map(lambda a,v:a+sign*v,total,g)
        offset=offset+coins[depth]*(width//2)
    d=jax.tree.map(lambda a:jax.lax.dynamic_index_in_dim(a,offset,keepdims=False),panel)
    return total,d


@eqx.filter_jit
def corrected(g,d):
    return jax.tree.map(lambda v,h:v+h if h.size else v,g,d)


@eqx.filter_jit
def step(state,g,key):
    return state.take_step(g,key=key)[0]


@eqx.filter_jit
def distance(x,y):
    return .5*sum(j.vdot(a-b,a-b) for a,b in zip(jax.tree.leaves(x),jax.tree.leaves(y)))


def main(s,b,M,k):
    assert M>=2 and M&(M-1)==0
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(s,fs)
    p=Path('/tmp/signed-panel.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(p))
    with np.load(p) as z:hold=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    orig=Attention.__call__;fused_attention.install()
    def estimate(ds,key):
        coins=jax.random.bernoulli(key,.5,(M.bit_length()-1,)).astype(j.int32)
        d,leaf=control(state.model,ds,coins,k)
        Attention.__call__=orig
        try:g=native_audit(state.model,leaf,b)
        finally:fused_attention.install()
        return corrected(g,d)
    seed=1287+s+k+b;gen=np.random.default_rng(seed)
    def draw(shape):
        ix=gen.integers(len(hold[0]),size=shape);return tuple(j.asarray(a[ix]) for a in hold)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=s,b=b,M=M,k=k,checkpoint=src,seed=seed)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='exact-native-signed-panels',id=os.urandom(5).hex(),config=cfg)
    sizes=sorted(set((1,2,b,2*b,4*b,8*b,12*b,16*b)))
    cases=[(m,B) for m in (61,62,7) for B in sizes]
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in cases]
    out=dict(**cfg,wandb=run.url,device=jax.devices()[0].device_kind,ms=[],pair=[],rows=rows)
    dst=ROOT+f'/signed-panel/s{s}-b{b}-M{M}-k{k}.json';dummy=j.arange(1)
    for i in range(37):
        a,z,ref1,ref2=ready((draw((M,b)),draw((M,b)),draw(max(sizes)),draw(max(sizes))))
        l,r=ready(tuple({B:tuple(v[:B] for v in d) for B in sizes} for d in (ref1,ref2)))
        k1,k2=jax.random.split(jax.random.key(seed+i))
        for ix in gen.permutation(np.arange(-1,len(cases))):
            start=time.perf_counter()
            if ix==-1:ready(step(state,estimate(a,k1),k1))
            else:
                m,B=cases[ix];ready(update(state,l[B],k1,dummy,m,12288 if m==61 else 16384,64))
            ms=1000*(time.perf_counter()-start)
            if i:
                if ix==-1:out['ms'].append(ms)
                else:rows[ix]['ms'].append(ms)
        if not i:continue
        out['pair'].append(float(distance(estimate(a,k1),estimate(z,k2))))
        for row,(m,B) in zip(rows,cases):
            row['pair'].append(float(pair(state,l[B],r[B],k1,dummy,m,12288 if m==61 else 16384,64)))
        if i%6==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'variance':np.mean(out['pair'])});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

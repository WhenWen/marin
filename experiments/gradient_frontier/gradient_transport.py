"""Learn a left/right linear map from a cheap transformer gradient to a full training gradient."""
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
from .frozen_panel import cheap,gold
from .native_eval import recover
from .head_to_head import update
from .contest import pair
from .fused_attention import install
from .run import ROOT,ready


def matrices(a):
    return a.ndim>=3 and a.size<2_000_000 and a.shape[-1]<=512


def reshape(a):
    return a.reshape(a.shape[0],-1,a.shape[-1])


def observe(model,left,right,key,k):
    ka,kb=jax.random.split(key)
    g=jax.tree.map(j.subtract,gold(model,left,ka),gold(model,right,kb))
    h=jax.tree.map(j.subtract,eqx.filter_grad(cheap)(model,left,k),eqx.filter_grad(cheap)(model,right,k))
    return g,h


def covariance(g,h):
    if not matrices(g):
        return (j.reshape(j.vdot(g,h),(1,)),j.reshape(j.vdot(h,h),(1,)),j.empty((0,)),j.empty((0,)))
    v,u=reshape(g),reshape(h);tr=lambda x:x.swapaxes(-1,-2)
    return (v@tr(u),u@tr(u),tr(u)@v,tr(u)@u)


def solve(cross,gram,reg,identity):
    n=gram.shape[-1];eye=j.eye(n)
    ridge=reg*j.maximum(1.e-14,j.trace(gram,axis1=-2,axis2=-1)/n)
    # Solve A K = C with a shrinkage prior around the chosen scalar identity.
    left=cross+identity*ridge[:,None,None]*eye
    return j.linalg.solve(gram+ridge[:,None,None]*eye,left.swapaxes(-1,-2)).swapaxes(-1,-2)


@eqx.filter_jit
def fit(model,samples,k,key,mode):
    params=jax.tree.leaves(eqx.filter(model,eqx.is_inexact_array))
    def at(i):
        left,right=(jax.tree.map(lambda v:v[i,t],samples) for t in (0,1))
        g,h=observe(model,left,right,jax.random.fold_in(key,i),k)
        return tuple(covariance(x,y) for x,y in zip(jax.tree.leaves(g),jax.tree.leaves(h)))
    sums=jax.lax.fori_loop(1,len(samples[0]),lambda i,s:jax.tree.map(j.add,s,at(i)),at(0))
    bank=[]
    for p,(c,l,d,r) in zip(params,sums):
        if matrices(p):
            A=solve(c,l,.35,1.)
            B=solve(d.swapaxes(-1,-2),r,.35,1.).swapaxes(-1,-2)
            bank.append((A,B))
        else:bank.append((j.clip(c/j.maximum(l,1e-20),-3,3),j.empty((0,))))
    return tuple(bank)


def transport(tree,bank,mode):
    leaves,td=jax.tree.flatten(tree);vals=[]
    for v,(a,b) in zip(leaves,bank):
        if matrices(v):
            h=reshape(v);q=a@h
            if mode:q=.5*(q+h@b)
            vals.append(q.reshape(v.shape))
        else:vals.append(a[0]*v)
    return jax.tree.unflatten(td,vals)


@eqx.filter_jit
def timed(state,panel,bank,k,mode,key):
    M,b,t=panel[0].shape
    def d(i):return jax.tree.map(lambda v:v[i],panel)
    def h(i):return eqx.filter_grad(cheap)(state.model,d(i),k)
    initial=h(0)
    extra=jax.lax.fori_loop(2,M,lambda i,s:jax.tree.map(j.add,s,h(i)),h(1))
    delta=transport(jax.tree.map(lambda a,b:(b-(M-1)*a)/M,initial,extra),bank,mode)
    grad=jax.tree.map(j.add,gold(state.model,d(0),key),delta)
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def metrics(state,left,right,key,k,bank,mode):
    dg,dh=observe(state.model,left,right,key,k);pred=transport(dh,bank,mode)
    def val(x,h,y):
        return .5*j.stack((j.vdot(x,x),j.vdot(y,y),j.vdot(x,y),j.vdot(h,h),j.vdot(x,h)))
    return sum(val(*c) for c in zip(*map(jax.tree.leaves,(dg,dh,pred))))


def main(step,b,M,k,ncal,mode):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    origin=state if step==4001 else recover(4001,fs)[0]
    tmp=Path('/tmp/transport-data.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:
        hold=tuple(z[name][512:] for name in ('tokens','weights','segments'))
        train=tuple(j.asarray(z[name][:2*ncal].reshape(ncal,2,1,-1)) for name in ('tokens','weights','segments'))
    key=jax.random.key(1883);bank=ready(fit(origin.model,train,k,key,mode));prep=[]
    for _ in range(4):
        t=time.perf_counter();bank=ready(fit(origin.model,train,k,key,mode));prep.append(1000*(time.perf_counter()-t))
    seed=91597+step+k;gen=np.random.default_rng(seed)
    def draw(shape):
        ix=gen.integers(len(hold[0]),size=shape)
        return tuple(j.asarray(v[ix]) for v in hold)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,b=b,M=M,k=k,ncal=ncal,mode=mode,checkpoint=src,seed=seed)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='learned-gradient-transport',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,B) for m in (61,62,7) for B in (b//2,b,2*b,3*b,4*b,6*b)]
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in cases]
    out=dict(**cfg,wandb=run.url,device=jax.devices()[0].device_kind,rows=rows,components=[],ms=[],prep=prep)
    dst=ROOT+f'/gradient-transport/s{step}-b{b}-M{M}-k{k}-c{ncal}-m{mode}.json'
    dummy=j.arange(1)
    for i in range(49):
        a,z,panel=ready((draw(6*b),draw(6*b),draw((M,b))))
        l,r=ready(tuple({B:tuple(v[:B] for v in d) for _,B in cases} for d in (a,z)))
        key=jax.random.key(seed+i)
        for ix in gen.permutation(np.arange(-1,len(cases))):
            t=time.perf_counter()
            if ix==-1:ready(timed(state,panel,bank,k,mode,key))
            else:
                m,B=cases[ix];ready(update(state,l[B],key,dummy,m,12288 if m==61 else 16384,64))
            ms=1000*(time.perf_counter()-t)
            if i:
                if ix==-1:out['ms'].append(ms)
                else:rows[ix]['ms'].append(ms)
        if not i:continue
        out['components'].append(np.asarray(metrics(state,l[b],r[b],key,k,bank,mode)).tolist())
        for row,(m,B) in zip(rows,cases):
            row['pair'].append(float(pair(state,l[B],r[B],key,dummy,m,12288 if m==61 else 16384,64)))
        if i%8==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'time_ms':np.median(out['ms'])});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

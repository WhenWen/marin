"""Audit the trained cross-fit control using cyclic panels of the identical emitted gradient executable."""
import functools
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as j
import numpy as np
import haliax as h
import fsspec
import wandb
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention,online_adjoint
from .contest import pair
from .diagnostic_artifacts import publish
from .group_regression import fit
from .independent_audits import copy_state,donated_baseline
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit
from .tail import trunk


def loss(model,data,key,k,n,bank,at):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model);x=trunk(m,data)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    return online_adjoint.objective(x,w,j.roll(data[0],-1,axis=1),data[1],key,(17200000+k,n),bank,at,uniform=True)


@eqx.filter_jit
def draw(model,data,key,k,n,bank,base,shift):
    width=data[0].shape[1]//n
    return eqx.filter_grad(loss)(model,data,key,k,n,bank,(base+shift)%width)


@eqx.filter_jit
def offsets(key,B,n,T):return jax.random.randint(key,(2,B*n//2),0,T//n)


@eqx.filter_jit
def add(a,b):return jax.tree.map(j.add,a,b)


@eqx.filter_jit
def subtract_mean(native,total,length):return jax.tree.map(lambda a,b:a-b/length,native,total)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(args,state):
    h,d,r,key=args
    if d is not None:h=jax.tree.map(lambda x,y:x+y/r,h,d)
    return state.take_step(h,key=key)[0]


@eqx.filter_jit
def metrics(h,g,d,e):
    diff=lambda u,v:sum(j.vdot(a-b,a-b) for a,b in zip(jax.tree.leaves(u),jax.tree.leaves(v)))/2
    prod=lambda u,v:sum(j.vdot(a,b) for a,b in zip(jax.tree.leaves(u),jax.tree.leaves(v)))
    return j.stack((diff(h,g),(prod(d,d)+prod(e,e))/2,prod(d,e)))


def main(step,B,k,n,width=128,depth=4):
    configure_storage();fs=fsspec.filesystem('s3')
    def checkpoint(index):
        if width==128 and depth==4:return recover(index,fs)
        from .scale_probe import recover_scale
        return recover_scale(width,depth,fs,index)[:2]
    state,src=checkpoint(step);original=Attention.__call__;fused_attention.install()
    file=Path('/tmp/strict-crossfit.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(file))
    with np.load(file) as z:
        pool=tuple(z[key][512:] for key in ('tokens','weights','segments'))
        cal=tuple(j.asarray(z[key][:4]) for key in ('tokens','weights','segments'))
    origin=state if step==4001 else checkpoint(4001)[0]
    bank=ready(fit(origin.model,cal,k,113));prep=[]
    for _ in range(8):
        tick=time.perf_counter();bank=ready(fit(origin.model,cal,k,113));prep.append(1000*(time.perf_counter()-tick))
    L=pool[0].shape[1]//n;phase=tuple(j.asarray(i,j.int32) for i in range(L))
    seed=163940+step+B+k+n;gen=np.random.default_rng(seed)
    def g(data,key,at,i):return draw(state.model,data,key,k,n,bank,at,phase[i])
    def residual(data,key,at):
        Attention.__call__=original
        try:gold=native_audit(state.model,data,B)
        finally:fused_attention.install()
        summed=g(data,key,at,0)
        for i in range(1,L):summed=add(summed,g(data,key,at,i))
        return subtract_mean(gold,summed,L)
    cases=[(m,kk,nn,b) for m,kk,nn in ((3,32768,0),(4,4096,0),(201,4096,0),(202,8192,0),(206,1024,0),(61,4096,32),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in sorted({max(2,B//4),B//2,3*B//4,B,5*B//4,2*B})]
    cfg=dict(step=step,B=B,k=k,n=n,width=width,depth=depth,checkpoint=src,seed=seed,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='precision-repaired-crossfit',id=os.urandom(5).hex(),config=cfg)
    rows=[dict(method=m,k=kk,n=nn,b=b,ms=[],pair=[]) for m,kk,nn,b in cases]
    out=dict(**cfg,wandb=run.url,rows=rows,prep=prep,components=[],times=[])
    dst=ROOT+f'/strict-crossfit/{cfg["device"].split()[1]}-s{step}-b{B}-k{k}-n{n}.json'
    if width!=128 or depth!=4:dst=dst.replace('/strict-crossfit/',f'/strict-crossfit-scale/w{width}-d{depth}-')
    sizes={q[-1] for q in cases};dummy=j.arange(1)
    def sample():
        ix=gen.integers(len(pool[0]),size=max(sizes));return {b:tuple(j.asarray(a[ix[:b]]) for a in pool) for b in sizes}
    for it in range(41):
        a,b,c,d=ready(tuple(sample() for _ in range(4)));keys=jax.random.split(jax.random.key(seed+it),4)
        parts=[a[B],b[B],c[B],d[B]];ats=[offsets(key,B,n,pool[0].shape[1]) for key in keys];ts=[0.,0.]
        for ix in gen.permutation(np.arange(-2,len(cases))):
            work=copy_state(state);tick=time.perf_counter()
            if ix<0:
                # Include both RNG generation and the same materialized gradient kernel used in the precision audit.
                at=offsets(keys[0],B,n,pool[0].shape[1]);raw=g(a[B],keys[0],at,0)
                corr=residual(c[B],keys[2],offsets(keys[2],B,n,pool[0].shape[1])) if ix==-2 else None
                ready(advance((raw,corr,.015625,keys[0]),work))
            else:
                m,kk,nn,bs=cases[ix];ready(donated_baseline((a[bs],keys[0],dummy,m,kk,nn),work))
            elapsed=1000*(time.perf_counter()-tick)
            if it:
                if ix<0:ts[-ix-1]=elapsed
                else:rows[ix]['ms'].append(elapsed)
        if not it:continue
        h1,h2=[g(parts[i],keys[i],ats[i],0) for i in (0,1)];d1,d2=[residual(parts[i],keys[i],ats[i]) for i in (2,3)]
        out['components'].append(np.asarray(metrics(h1,h2,d1,d2)).tolist());out['times'].append(ts)
        for row,(m,kk,nn,bs) in zip(rows,cases):row['pair'].append(float(pair(state,a[bs],b[bs],keys[0],dummy,m,kk,nn)))
        if it%5==0:publish(fs,dst,out)
    run.log({'cycles':40});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(*map(int,sys.argv[1:]))

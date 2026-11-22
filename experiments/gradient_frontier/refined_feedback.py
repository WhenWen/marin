"""Forecast the current mean from past native feedback in small parameter blocks."""
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .causal_feedback import packet,decay
from .native_eval import recover,objective
from .fused_attention import install
from .run import ROOT


def columns(cur,prev,older,origin,groups):
    def make(m,m1,m2,w,w1,w2,w0):
        n=m.size
        if not n:return j.empty((1,9,0),j.float32)
        count=groups if n%groups==0 and n>=groups*16 else 1
        def block(a):return a.reshape(count,-1)
        m,m1,m2,w,w1,w2,w0=map(block,(m,m1,m2,w,w1,w2,w0))
        rms=lambda x:j.sqrt(j.maximum(1.e-14,j.mean(x*x,axis=1,keepdims=True)))
        scale=j.clip(j.abs(m)/rms(m),0.,5.)
        d=w-w1
        return j.stack((m,m1,m2,d,w1-w2,w-w0,m*scale,d*scale,w*rms(m)/rms(w)),axis=1)
    return jax.tree.map(make,cur[0],prev[0],older[0],cur[1],prev[1],older[1],origin[1])


@eqx.filter_jit
def summarize(cur,previous,older,origin,following,groups):
    b=columns(cur,previous,older,origin,groups)
    out=[]
    for z,m,n,rate in zip(jax.tree.leaves(b),jax.tree.leaves(cur[0]),jax.tree.leaves(following[0]),decay(cur[0])):
        g=((n-rate*m)/(1-rate)).reshape(z.shape[0],-1)
        out.append((z@z.swapaxes(-1,-2),j.einsum('gdl,gl->gd',z,g)))
    return tuple(out)


@eqx.filter_jit
def makepred(state,cur,prev,older,origin,sums,step,groups):
    cols=jax.tree.leaves(columns(cur,prev,older,origin,groups))
    leaves,td=jax.tree.flatten(eqx.filter(state.model,eqx.is_inexact_array))
    vals=[]
    for z,v,(gram,cross) in zip(cols,leaves,sums):
        if not z.size:vals.append(j.zeros_like(v));continue
        if step<=4002:pred=-10.*z[:,0]
        else:
            sd=j.sqrt(j.maximum(1e-15,j.diagonal(gram,axis1=1,axis2=2)))
            r=gram/sd[:,:,None]/sd[:,None,:]+.16*j.eye(z.shape[1])
            prior=j.zeros_like(sd).at[:,0].set(-10.*sd[:,0])
            a=j.linalg.solve(r,(cross/sd+.16*prior)[:,:,None])[:,:,0]/sd
            a=j.where(j.diagonal(gram,axis1=1,axis2=2)>1e-14,a,0.)
            pred=j.einsum('gd,gdl->gl',a,z)
        vals.append(pred.reshape(v.shape))
    return jax.tree.unflatten(td,vals)


@eqx.filter_jit
def evaluate(state,a,b,pred,key):
    fa,fb=eqx.filter_grad(objective)(state.model,a,j.arange(1),key,(0,0)),eqx.filter_grad(objective)(state.model,b,j.arange(1),key,(0,0))
    return sum(j.vdot(x-p,y-p) for x,y,p in zip(*map(jax.tree.leaves,(fa,fb,pred))))


def main(step,groups):
    configure_storage();fs=fsspec.filesystem('s3');install()
    # Only these preceding states affect the fitted coefficients. The evaluation draws are separate.
    packets={}
    for t in sorted({4001,*range(max(4001,step-6),step+1)}):
        s,_=recover(t,fs);packets[t]=packet(s)
    state,_=recover(step,fs);history=[]
    for t in range(max(4001,step-4),step):
        history.append(summarize(packets[t],packets[max(4001,t-1)],packets[max(4001,t-2)],packets[4001],packets[t+1],groups))
    if history:sums=jax.tree.map(lambda *x:sum(x),*history)
    else:
        cur=packets[4001];sums=jax.tree.map(j.zeros_like,summarize(cur,cur,cur,cur,cur,groups))
    pred=makepred(state,packets[step],packets[max(4001,step-1)],packets[max(4001,step-2)],packets[4001],sums,step,groups)
    tmp=Path('/tmp/feedback-test.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    rng=np.random.default_rng(3852+step);risk=[]
    for i in range(96):
        a,b=[tuple(j.asarray(v[ix]) for v in pool) for ix in rng.integers(len(pool[0]),size=(2,8))]
        risk.append(float(evaluate(state,a,b,pred,jax.random.key(i))))
    fs.pipe(ROOT+f'/refined-feedback/s{step}-g{groups}.json',json.dumps(dict(step=step,groups=groups,risk=risk)).encode())


if __name__=='__main__':main(*map(int,sys.argv[1:]))

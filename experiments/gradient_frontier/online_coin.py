"""A causal two-feature ridge predictor updated from the estimator actually returned to MuonW."""
import equinox as eqx
import jax
import jax.numpy as j
import numpy as np
from .causal_feedback import slim
from .temporal import momentum
from . import debias


def prepare(state,previous,history,step):
    prev=slim(eqx.filter(previous.model,eqx.is_inexact_array))
    length=len(jax.tree.leaves(prev));gram=np.zeros((length,2,2));cross=np.zeros((length,2))
    columns=[0,3]
    for record in history[:step-4001]:
        gram=.72*gram+np.asarray(record['gram'])[:,columns,:][:,:,columns]
        cross=.72*cross+np.asarray(record['feedback_cross'])[:,columns]
    return prev,j.asarray(gram,j.float32),j.asarray(cross,j.float32)


def solve(gram,cross):
    sd=j.sqrt(j.maximum(1e-16,j.diagonal(gram,axis1=1,axis2=2)))
    prior=j.zeros_like(sd).at[:,0].set(-8.*sd[:,0])
    beta=j.linalg.solve(gram/sd[:,:,None]/sd[:,None,:]+.16*j.eye(2),(cross/sd+.16*prior)[:,:,None])[:,:,0]/sd
    return j.clip(beta,-50.,50.)


def inspect(state,previous,gram,cross):
    beta=solve(gram,cross);mom=jax.tree.leaves(momentum(state));old=jax.tree.leaves(previous)
    params,td=jax.tree.flatten(eqx.filter(state.model,eqx.is_inexact_array))
    fit=[];xs=[]
    for c,m,w,pr in zip(beta,mom,params,old):
        if pr.size:
            x=j.stack((m.reshape(-1),(w-pr).reshape(-1)))
            fit.append((c@x).reshape(w.shape));xs.append(x)
        else:
            fit.append(j.zeros_like(w));xs.append(j.empty((2,0),j.float32))
    return jax.tree.unflatten(td,fit),xs


@eqx.filter_jit
def prediction(state,previous,gram,cross):
    return inspect(state,previous,gram,cross)[0]


@eqx.filter_jit
def step(state,data,delta,key,k,n,p,r,previous,gram,cross,next_key=None):
    pred,xs=inspect(state,previous,gram,cross)
    g=pred
    if data is not None:
        raw=debias.draw(state,data,key,k,n,61,0,j.arange(1))
        g=jax.tree.map(lambda a,b:a+(b-a)/p,g,raw)
    if delta is not None:g=jax.tree.map(lambda a,b:a+b/r,g,delta)
    # Both statistics are available from the step just taken and enter only the next forecast.
    aa=.72*gram+j.stack([x@x.T for x in xs])
    bb=.72*cross+j.stack([x@v.reshape(-1) if x.size else j.zeros(2) for x,v in zip(xs,jax.tree.leaves(g))])
    return state.take_step(g,key=key if next_key is None else next_key)[0],aa,bb

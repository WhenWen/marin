"""Fit a temporal predictor from the optimizer's own already-observed gradient feedback."""
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover
from .temporal import momentum
from .run import ROOT


def slim(tree):
    return jax.tree.map(lambda x:x if x.size<2_000_000 else j.empty((0,),j.float32),tree)


def packet(state):
    return slim(momentum(state)),slim(eqx.filter(state.model,eqx.is_inexact_array))


def basis(cur,prev,older,origin,step):
    def f(m,m1,m2,w,w1,w2,w0):
        if not m.size:return j.empty((6,0),j.float32)
        return j.stack((m,m1,m2,w-w1,w1-w2,w-w0)).reshape(6,-1)
    return jax.tree.map(f,cur[0],prev[0],older[0],cur[1],prev[1],older[1],origin[1])


def decay(tree):
    return [(.95 if 'proj.weight' in jax.tree_util.keystr(path) else .9) for path,x in jax.tree_util.tree_leaves_with_path(tree)]


@eqx.filter_jit
def statistics(cur,prev,older,origin,nxt,step):
    Bs=jax.tree.leaves(basis(cur,prev,older,origin,step));m=jax.tree.leaves(cur[0]);nextm=jax.tree.leaves(nxt[0])
    gram=[];cross=[]
    for z,a,b,beta in zip(Bs,m,nextm,decay(cur[0])):
        clipped=(b-beta*a)/(1-beta)
        gram.append(z@z.T);cross.append(z@clipped.reshape(-1))
    return j.stack(gram),j.stack(cross)


def coefficients(history,mode):
    # User-visible training feedback for a past step is already available when fitting its successor.
    if not history:return None
    cov=np.mean([h[0] for h in history[-3:]],0)
    cross=np.mean([h[1] for h in history[-3:]],0)
    sd=np.sqrt(np.maximum(1e-16,np.diagonal(cov,axis1=1,axis2=2)))
    norm=cov/sd[:,:,None]/sd[:,None,:]
    A=np.linalg.solve(norm+.12*np.eye(6),(cross/sd)[:,:,None])[:,:,0]/sd
    return np.where(np.diagonal(cov,axis1=1,axis2=2)>1e-14,A,0.)


def main():
    configure_storage();fs=fsspec.filesystem('s3');pools={};hist=[];result=[]
    for t in range(4001,4012):
        pools[t]=packet(recover(t,fs)[0])
    for t in range(4001,4011):
        z=json.loads(fs.cat_file(ROOT+f'/history-prediction/s{t}.json'))
        g=np.array(z['gram']);y=np.array(z['pairs'])[:,:,2:];signal=np.array(z['pairs'])[:,:,0].sum(1).mean()
        A=coefficients(hist,0)
        if A is None:A=np.zeros((len(g),6));A[:,0]=-10.
        # Drop the two vocabulary-sized predictions in the fast implementation.
        A[-2:]=0
        risk=signal-2*np.sum(A*y.mean(0))+np.einsum('pi,pij,pj->',A,g,A)
        stats=statistics(pools[t],pools[max(4001,t-1)],pools[max(4001,t-2)],pools[4001],pools[t+1],t)
        gram,cross=[np.asarray(q) for q in stats]
        result.append(dict(step=t,risk=float(risk),signal=float(signal),A=A.tolist(),gram=gram.tolist(),feedback_cross=cross.tolist()))
        hist.append((gram,cross))
    fs.pipe(ROOT+'/causal-prediction/feedback.json',json.dumps(result).encode())
    print([(r['step'],round(r['risk'],4)) for r in result],flush=True)


if __name__=='__main__':main()

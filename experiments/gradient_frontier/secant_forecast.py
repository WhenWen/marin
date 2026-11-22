"""Use past MuonW updates to interpolate the gradient field in local parameter blocks."""
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
from .fused_attention import install
from .native_eval import recover,objective
from .run import ROOT

SETTINGS=((.03,0.),(.25,0.),(1.,0.),(.03,.5),(.25,.5),(1.,.5))


@eqx.filter_jit
def predictors(model,packets,groups):
    weights=[jax.tree.leaves(p[1]) for p in packets];mom=[jax.tree.leaves(p[0]) for p in packets]
    betas=decay(packets[-1][0])
    leaves,td=jax.tree.flatten(eqx.filter(model,eqx.is_inexact_array));allpred=[]
    for i,v in enumerate(leaves):
        w=weights[-1][i]
        if not w.size:
            allpred.append(j.empty((len(SETTINGS),0),j.float32));continue
        n=groups if w.size%groups==0 and w.size>=groups*16 else 1
        now=w.reshape(n,-1)
        design=j.stack([q[i].reshape(n,-1)-now for q in weights[:-1]],axis=1)
        target=j.stack([((a[i]-betas[i]*b[i])/(1.-betas[i])).reshape(n,-1) for a,b in zip(mom[1:],mom[:-1])],axis=1)
        C=design@design.swapaxes(-1,-2)
        # Local linear extrapolation weights sum to one. Each output vector uses only past observed gradients.
        scale=j.maximum(1e-15,j.diagonal(C,axis1=-2,axis2=-1).mean(-1))
        eye=j.eye(len(weights)-1)
        pred=[]
        for reg,mix in SETTINGS:
            system=C/scale[:,None,None]+reg*eye
            ww=j.linalg.solve(system,j.ones((*system.shape[:2],1)))[:,:,0]
            ww=ww/j.maximum(ww.sum(1,keepdims=True),1e-12)
            fld=j.einsum('gi,gid->gd',ww,target).reshape(-1)
            pred.append((1.-mix)*fld+mix*(-10.*mom[-1][i].reshape(-1)))
        allpred.append(j.stack(pred))
    return jax.tree.unflatten(td,allpred)


@eqx.filter_jit
def probe(model,a,b,pred,key):
    def g(d):return eqx.filter_grad(objective)(model,d,j.arange(1),key,(0,0))
    ga,gb=g(a),g(b)
    errors=j.zeros(len(SETTINGS));signal=0.
    for x,y,P in zip(jax.tree.leaves(ga),jax.tree.leaves(gb),jax.tree.leaves(pred)):
        signal+=j.vdot(x,y)
        if P.size:
            errors+=j.sum(P*P,axis=1)-P@(x.reshape(-1)+y.reshape(-1))
    return errors+signal


def main(step,groups):
    assert step>4001
    configure_storage();fs=fsspec.filesystem('s3');install()
    packets=[]
    for i in range(max(4001,step-7),step+1):
        state,src=recover(i,fs);packets.append(packet(state))
    models=predictors(state.model,packets,groups)
    path=Path('/tmp/secant-data.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(path))
    with np.load(path) as z:pool=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    rng=np.random.default_rng(21981+step);risk=[]
    for i in range(128):
        a,b=[tuple(j.asarray(v[ix]) for v in pool) for ix in rng.integers(len(pool[0]),size=(2,8))]
        risk.append(np.asarray(probe(state.model,a,b,models,jax.random.key(i))).tolist())
    out=dict(step=step,groups=groups,settings=SETTINGS,risk=risk,source=src)
    fs.pipe(ROOT+f'/secant-forecast/s{step}-g{groups}.json',json.dumps(out).encode())


if __name__=='__main__':main(*map(int,sys.argv[1:]))

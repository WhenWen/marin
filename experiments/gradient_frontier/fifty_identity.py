"""Enumerate every emitted phase and audit outcome at fixed real native model states."""
import json
import os
import sys
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention,strict_crossfit
from .diagnostic_artifacts import publish
from .fifty_crossover import native_call,read_np
from .group_regression import fit
from .native_eval import recover
from .subspace_forecast import native_audit
from .run import ROOT,ready


@eqx.filter_jit
def audit_branch(h,d,den):
    return jax.tree.map(lambda a,b:a+b*den,h,d)


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);origin,_=recover(4001,fs);fused_attention.install()
    cal=tuple(v[:4] for v in read_np(fs,ROOT+'/inputs/native-packed.npz'))
    bank=ready(fit(origin.model,cal,4096,113));pool=read_np(fs,ROOT+'/inputs/native-confirm3.npz')
    B,k,n,den=32,4096,128,32;L=1024//n;r=1./den
    seed=73850+step;gen=np.random.default_rng(seed)
    cfg=dict(step=step,source=src,B=B,k=k,n=n,r=r,device=jax.devices()[0].device_kind,seed=seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fifty-conditional-gradient-enumeration',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,outcomes_per_batch=2*L,rows=[])
    for panel in range(6):
        ids=gen.integers(len(pool[0]),size=B);data=tuple(v[ids] for v in pool)
        key=jax.random.key(seed+panel);off=strict_crossfit.offsets(key,B,n,1024)
        g=ready(native_call(native_audit,state.model,data,B))
        grads=[strict_crossfit.draw(state.model,data,key,k,n,bank,off,j.asarray(i)) for i in range(L)]
        total=grads[0]
        for v in grads[1:]:total=strict_crossfit.add(total,v)
        delta=strict_crossfit.subtract_mean(g,total,L)
        hit=[audit_branch(v,delta,den) for v in grads]
        parts=[jax.tree.leaves(v) for v in grads+hit]
        sq=ref=absolute=0.
        for leaf,truth in enumerate(jax.tree.leaves(g)):
            current=np.zeros(truth.shape,np.float64)
            for i,gt in enumerate(parts):current+=np.asarray(gt[leaf],np.float64)*((1-r) if i<L else r)/L
            tr=np.asarray(truth,np.float64);err=(current-tr).ravel()
            sq+=np.dot(err,err);ref+=np.dot(tr.ravel(),tr.ravel());absolute=max(absolute,np.max(np.abs(err)))
        row=dict(panel=panel,sample_ids=ids.tolist(),relative_l2=np.sqrt(sq/ref),maximum_coordinate=absolute)
        out['rows'].append(row);publish(fs,ROOT+f'/fifty-identity-v1/s{step}.json',out)
        run.log({'native_relative_l2':row['relative_l2'],'maximum_coordinate':absolute})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(int(sys.argv[1]))

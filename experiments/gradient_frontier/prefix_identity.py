"""Enumerate all balanced target-panel phases, comparing native, full fused, and packed gradients on identical documents."""
import json
import os
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .diagnostic_artifacts import publish
from .fused_attention import install
from .native_eval import recover,objective
from .prefix_panels import loss
from .subspace_forecast import native_audit
from .run import ROOT,ready


@eqx.filter_jit
def average(state,d,key,K):
    def pick(phase):return eqx.filter_grad(loss)(state.model,d,key,K,0,0,phase)
    total=jax.lax.fori_loop(1,K,lambda i,v:jax.tree.map(j.add,v,pick(i)),pick(0))
    return jax.tree.map(lambda v:v/K,total)


@eqx.filter_jit
def levels(grad,ref):
    diff=jax.tree.map(j.subtract,grad,ref)
    sq=sum(j.vdot(a,a) for a in jax.tree.leaves(diff));base=sum(j.vdot(a,a) for a in jax.tree.leaves(ref))
    return j.array([j.sqrt(sq/base),j.max(j.stack([j.max(j.abs(a)) for a in jax.tree.leaves(diff)])),sq])


@eqx.filter_jit
def fused_reference(state,d,key):
    return eqx.filter_grad(objective)(state.model,d,j.arange(1),key,(0,0))


def main(step,K):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    from levanter.layers.attention import Attention
    original=Attention.__call__
    path=Path('/tmp/prefix-identity.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(path))
    with np.load(path) as z:pool=tuple(z[k][512:] for k in ('tokens','weights','segments'))
    B=max(16,2*K);seed=191682+step
    rng=np.random.default_rng(seed)
    cfg=dict(step=step,K=K,B=B,source=src,seed=seed,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='exact-prefix-phase-enumeration',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[]);dest=ROOT+f'/prefix-identity/s{step}-k{K}.json'
    for it in range(16):
        ids=rng.integers(len(pool[0]),size=B);d=ready(tuple(j.asarray(v[ids]) for v in pool))
        key=jax.random.key(seed+it)
        Attention.__call__=original
        try:gold=ready(native_audit(state.model,d,B))
        finally:install()
        fused=fused_reference(state,d,key);avg=average(state,d,key,K)
        out['rows'].append([np.asarray(levels(avg,fused)).tolist(),np.asarray(levels(avg,gold)).tolist(),np.asarray(levels(fused,gold)).tolist()])
        if (it+1)%4==0:publish(fs,dest,out)
    run.log({'batches':16});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

"""Enumerate the full expected raw segment gradient and bound rounding in its final additions."""
import os
import sys

import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from experiments.hybrid_transfer.storage import configure_storage
from rigging.secrets import require_secret
from . import segment_queries as seg,fused_attention
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import native_call
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit


def main(step,cut,batch,k,balanced):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    state,src=recover(step,fs);fused_attention.install()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(**meta,step=step,cut=cut,batch=batch,k=k,length_balanced=bool(balanced),source=src)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=os.urandom(6).hex(),group='attention-segment-identity',mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[])
    phases=tuple(j.asarray(i,j.int32) for i in range(16))
    rng=np.random.default_rng(48137+step)
    for panel in range(4):
        ix=rng.integers(len(pool[0]),size=32)
        host=tuple(x[ix] for x in pool);d=ready(tuple(j.asarray(x) for x in host))
        layout=seg.plan(host[2],cut,batch,rng,bool(balanced))
        gold=ready(native_call(native_audit,state.model,d,32))
        corr=ready(seg.correction(state.model,d,layout,cut,k,phases))
        refs=[np.asarray(a,np.float64) for a in jax.tree.leaves(gold)]
        acc=[np.asarray(a,np.float64).copy() for a in jax.tree.leaves(corr)]
        magnitude=[np.abs(a) for a in acc]
        groups=[]
        for length,entry in zip((cut,1024),layout):
            if entry is None:groups.append(0);continue
            table,offsets,_=entry;groups.append(len(table))
            for group,off in zip(table,offsets):
                for phase in phases:
                    # This is the exact materialized summand shipped by the routine branch.
                    raw=seg.scale(seg.raw(state.model,d,group,off,phase,length,k),len(table))
                    prob=1./(len(table)*len(phases))
                    for a,m,v in zip(acc,magnitude,jax.tree.leaves(raw)):
                        x=np.asarray(v,np.float64)*prob
                        a+=x;m+=np.abs(x)
        norm=lambda xs:np.sqrt(sum(np.square(a).sum() for a in xs))
        err=norm([a-b for a,b in zip(acc,refs)])/norm(refs)
        # Simple triangle bound for ordinary FP32 addition, inverse-probability scaling, and audit addition.
        # Gradients and scaled residuals are finite. The tiny absolute term also covers subnormal rounding/FTZ.
        guard=(8*np.finfo(np.float32).eps*norm(magnitude)+np.sqrt(sum(x.size for x in refs))*np.finfo(np.float32).tiny*4096)/norm(refs)
        out['rows'].append(dict(panel=panel,groups=groups,phase_count=16,enumerated_relative_l2=err,final_rounding_bound=guard,bounded_relative_l2=err+guard))
        publish(fs,ROOT+f'/segment-identity/s{step}-l{cut}-b{batch}-k{k}-bal{balanced}.json',out)
        run.log({'enumerated_relative_l2':err,'final_rounding_bound':guard})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

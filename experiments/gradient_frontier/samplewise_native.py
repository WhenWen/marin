"""Measure samplewise native audits and vectorized controls against the exact B1 executable."""
import os
import sys
import time
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fifty_blocks_strict as fixed,fused_attention
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import native_call
from .native_batch_reduction import effective_gradient
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready
from .subspace_forecast import native_audit
from .unbiased_check import native


@eqx.filter_jit
def vectorized(model,data,group):
    batches=jax.tree.map(lambda x:x.reshape(-1,group,x.shape[-1]),data)
    def one(d):
        return eqx.filter_grad(native)(model,jax.tree.map(lambda a:a[None,:],d))
    def group_sum(d):
        grads=jax.vmap(one)(d)
        return jax.tree.map(lambda g:g.sum(0,dtype=j.float32),grads)
    first=group_sum(jax.tree.map(lambda a:a[0],batches))
    def body(i,acc):
        return jax.tree.map(j.add,acc,group_sum(jax.tree.map(lambda a:a[i],batches)))
    sums=jax.lax.fori_loop(1,len(batches[0]),body,first)
    return jax.tree.map(lambda v:v/data[0].shape[0],sums)


@eqx.filter_jit
def divide32(g):return jax.tree.map(lambda v:v/32,g)


@eqx.filter_jit
def minus(a,b):return jax.tree.map(j.subtract,a,b)


def serial(model,d):return divide32(effective_gradient(model,d,1))


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs);state,src=recover(step,fs);fused_attention.install()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(**meta,step=step,source=src,device=jax.devices()[0].device_kind,groups=[1,4,8,16,32])
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='samplewise-native-audit-kernels',id=os.urandom(6).hex(),mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[])
    for i in range(17):
        rng=np.random.default_rng(np.random.SeedSequence([91627,step,i]))
        ab=[ready(tuple(j.asarray(z[ix]) for z in pool)) for ix in rng.integers(len(pool[0]),size=(2,32))]
        target=[ready(native_call(serial,state.model,d)) for d in ab]
        buf=[];base=[]
        for j0,d in enumerate(ab):
            key=jax.random.key(step*197+i*2+j0)
            td=fixed.cyclic_sum(state,d,key,1);rd=fixed.cyclic_sum(state,d,key,2)
            buf.append((td,rd))
            base.append(ready(fixed.correction(target[j0],td,rd,16)))
        for group in rng.permutation(cfg['groups']):
            group=int(group);t=time.perf_counter()
            approx=[ready(native_call(vectorized,state.model,d,group)) for d in ab]
            ms=1000*(time.perf_counter()-t)/2
            discrepancy=[ready(minus(t,a)) for t,a in zip(target,approx)]
            proxy=[fixed.correction(g,*s,16) for g,s in zip(approx,buf)]
            x,y,z=(np.asarray(norm_moments(*pair)).tolist() for pair in (discrepancy,proxy,base))
            if i:out['rows'].append(dict(pair=i,group=group,ms=ms,samplewise_repair=x,fast_correction=y,serial_correction=z))
        if i:publish(fs,ROOT+f'/samplewise-native-v1/s{step}.json',out)
    run.log({'pairs':16});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(int(sys.argv[1]))

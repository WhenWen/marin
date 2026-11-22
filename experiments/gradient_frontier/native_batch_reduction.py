"""Compare native B32 backpropagation to the mean of actually executed samplewise and microbatch gradients."""
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
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .native_eval import recover
from .nested_forecast import norm_moments
from .subspace_forecast import native_audit
from .run import ROOT,ready


@eqx.filter_jit
def add(a,b):return jax.tree.map(j.add,a,b)


@eqx.filter_jit
def delta(a,ref,n):
    return jax.tree.map(lambda x,z:x/n-z,a,ref)


def effective_gradient(model,data,micro):
    g=native_audit(model,tuple(x[:micro] for x in data),micro)
    for pos in range(micro,32,micro):
        g=add(g,native_audit(model,tuple(x[pos:pos+micro] for x in data),micro))
    return g


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs);state,src=recover(step,fs)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(**meta,step=step,source=src,device=jax.devices()[0].device_kind)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='native-batch-reduction',id=os.urandom(6).hex(),mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[])
    for i in range(13):
        rng=np.random.default_rng(np.random.SeedSequence([751386,step,i]))
        stacks=[tuple(j.asarray(v[ix]) for v in pool) for ix in rng.integers(len(pool[0]),size=(2,32))]
        ref=[ready(native_audit(state.model,b,32)) for b in stacks]
        for micro in (1,4,8):
            t=time.perf_counter();d=[ready(delta(effective_gradient(state.model,b,micro),g,32//micro)) for b,g in zip(stacks,ref)]
            elapsed=1000*(time.perf_counter()-t)/2
            norms=np.asarray(norm_moments(*d));targets=np.asarray(norm_moments(*ref))
            if i:out['rows'].append(dict(pair=i,micro=micro,ms=elapsed,mean_squared_difference=norms[0].item(),difference_cross=norms[1].item(),mean_squared_native=targets[0].item(),native_cross=targets[1].item()))
        if i:publish(fs,ROOT+f'/native-batch-reduction/s{step}.json',out)
    run.log({'paired_draws':12});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(int(sys.argv[1]))

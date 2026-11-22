"""Measure coupled telescoping corrections from the B32 native target to the single-example native target."""
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
from .run import ROOT,ready
from .subspace_forecast import native_audit


@eqx.filter_jit
def contrast(a,b,c):return jax.tree.map(lambda x,y,z:(x+y)*.5-z,a,b,c)


def pair_level(model,data,m):
    left=tuple(v[:m] for v in data);right=tuple(v[m:] for v in data)
    return contrast(native_audit(model,left,m),native_audit(model,right,m),native_audit(model,data,2*m))


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs);state,src=recover(step,fs)
    cfg=dict(**meta,step=step,source=src,device=jax.devices()[0].device_kind,levels=[1,2,4,8,16])
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='samplewise-native-multilevel',id=os.urandom(6).hex(),mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[])
    for i in range(25):
        rng=np.random.default_rng(np.random.SeedSequence([524193,step,i]))
        for m in rng.permutation(cfg['levels']):
            m=int(m)
            arrays=ready([tuple(j.asarray(v[idx]) for v in pool) for idx in rng.integers(len(pool[0]),size=(2,2*m))])
            t=time.perf_counter();dif=[ready(pair_level(state.model,a,m)) for a in arrays];ms=1000*(time.perf_counter()-t)/2
            stats=np.asarray(norm_moments(*dif))
            if i:out['rows'].append(dict(pair=i,level=m,ms=ms,second_moment=float(stats[0]),squared_mean_est=float(stats[1])))
        if i%4==0 and i:publish(fs,ROOT+f'/native-reduction-levels/s{step}.json',out)
    run.log({'pairs_per_level':24});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(int(sys.argv[1]))

"""Compare gradient variance and actual one-step MuonW direction at native and sparse training states."""
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
from levanter.checkpoint import load_checkpoint
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention,fifty_blocks_strict as fixed,muon_safe_prediction as forecast
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import eval_ce,native_call,read_np
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit
import json


@eqx.filter_jit
def exact_update(state,g):
    return state.take_step(g,key=jax.random.split(state.training_key)[1])[0].model


def main(step,trajectory):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    initial,_=recover(4001,fs);fused_attention.install()
    if trajectory and step>4001:
        origin=ROOT+'/fifty-crossover-v1/GB200-p6-s856102/main/commits/'+str(step)+'.json'
        receipt=json.loads(fs.cat_file(origin));src=receipt['checkpoints']['402-12288-64-32-500']
        template=eqx.filter_eval_shape(lambda:initial)
        state=load_checkpoint(template,src,allow_partial=False,axis_mapping={},mesh=jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model')))
        assert int(state.step)==step
    else:state,src=recover(step,fs)
    assert len(forecast.matrix_paths(state))>0
    cal=read_np(fs,ROOT+'/inputs/native-packed.npz')
    a,b=ready([tuple(x[lo:lo+32] for x in cal) for lo in (0,32)])
    co=ready(forecast.calibrate(initial,a,b));tic=time.perf_counter();co=ready(forecast.calibrate(initial,a,b));cal_ms=1000*(time.perf_counter()-tic)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(**meta,step=step,trajectory=trajectory,source=src,muon_groups=len(forecast.matrix_paths(state)),cal_ms=cal_ms,seed=34568)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=os.urandom(6).hex(),group='muon-nesterov-forecast-drift',mode='online',config=cfg)
    scales=(0,4,8,9,10,-1)
    models={s:ready(forecast.next_model(state,co,s)) for s in scales}
    priors={s:ready(forecast.predict(state,co,s)) for s in scales}
    out=dict(**cfg,wandb=run.url,coefficients=[float(y) for y in co],rows=[])
    for i in range(24):
        rng=np.random.default_rng(np.random.SeedSequence([34568,step,i]))
        a,b,e=ready([tuple(j.asarray(v[ids]) for v in pool) for ids in rng.integers(len(pool[0]),size=(3,32))])
        gold=native_call(native_audit,state.model,a,32);ref=ready(exact_update(state,gold))
        loss0=float(native_call(eval_ce,state.model,e));loss_ref=float(native_call(eval_ce,ref,e))
        mom={s:[] for s in scales}
        for part in (1,2):
            g,h0=[fixed.raw_block(state,d,jax.random.key(7171+step+i*2+t),part,j.asarray(rng.integers(16),j.int32)) for t,d in enumerate((a,b))]
            for s in scales:mom[s].extend(np.asarray(fixed.moments(g,h0,priors[s],part)).tolist())
        for s in scales:
            stats=np.asarray(forecast.agreement(state,ref,models[0],models[s])).tolist()
            ce=float(native_call(eval_ce,models[s],e))
            out['rows'].append(dict(pair=i,scale=s,block_moments=mom[s],matrix_update_moments=stats,before_ce=loss0,after_ce=ce,exact_ce=loss_ref))
        if (i+1)%4==0:publish(fs,ROOT+f'/forecast-drift/s{step}-t{trajectory}.json',out)
    run.log({'pairs':24});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

"""Use matched draws to evaluate new forecasts at all fifty states, reusing the identical independent native correction."""
import json
import os
import sys

import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fifty_blocks_strict as fixed,fused_attention,muon_safe_prediction as forecast
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import read_np
from .native_eval import recover
from .run import ROOT,ready


def main(block,rep):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs,True)
    initial,_=recover(4001,fs);fused_attention.install()
    cal=read_np(fs,ROOT+'/inputs/native-packed.npz');a,b=ready([tuple(v[n:n+32] for v in cal) for n in (0,32)])
    co=ready(forecast.calibrate(initial,a,b))
    cfg=dict(**meta,block=block,rep=rep,device=jax.devices()[0].device_kind,coefficients=[float(x) for x in co])
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=os.urandom(6).hex(),group='fifty-constrained-forecast-variance',mode='online',config=cfg)
    for step in range(4001+block*5,4006+block*5):
        dst=ROOT+f'/fifty-forecast-variance-v1/GB200-s{step}-r{rep}.json'
        if fs.exists(dst) and json.loads(fs.cat_file(dst))['complete']:continue
        donor=ROOT+f'/fifty-block-strict-variance-v1/GB200-s{step}-r{rep}.json'
        src=json.loads(fs.cat_file(donor));assert src['complete'] and src['holdout']==meta['holdout'] and src['paired_seed']==653927
        state,ckpt=recover(step,fs);assert src['checkpoint']==ckpt
        priors={s:ready(forecast.predict(state,co,s)) for s in (8,-1)}
        out=dict(**cfg,step=step,source=ckpt,correction_source=donor,wandb=run.url,components={'420':[],'422':[]},complete=False)
        for i in range(16):
            rng=np.random.default_rng(np.random.SeedSequence([653927,step,rep,i]))
            ids=rng.integers(len(pool[0]),size=(4,32))
            ab=ready([tuple(j.asarray(x[q]) for x in pool) for q in ids[:2]])
            keys=jax.random.split(jax.random.fold_in(jax.random.key(402731+rep),step*100+i))
            stats={s:[] for s in priors}
            for part in (1,2):
                def raw(d,key):
                    k,pk=jax.random.split(key);ph=jax.random.randint(pk,(),0,16,dtype=j.int32)
                    return fixed.raw_block(state,d,k,part,ph)
                v,w=[raw(d,k) for d,k in zip(ab,keys)]
                for s,p in priors.items():stats[s].extend(np.asarray(fixed.moments(v,w,p,part)).tolist())
            for s,method in ((8,'420'),(-1,'422')):
                out['components'][method].append(stats[s]+src['components'][i][-2:])
            if (i+1)%4==0:
                out['complete']=i==15
                publish(fs,dst,out)
        run.log({'state':step,'pairs':16})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

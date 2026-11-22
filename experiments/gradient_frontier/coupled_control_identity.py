"""Check emitted nested short-gradient controls by marginalizing every phase, branch and precision coin."""
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
from . import coupled_block_control as ctl,prepared_blocks as prep,fused_attention
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import native_call
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit


def main(step,length,k):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    state,src=recover(step,fs);fused_attention.install()
    rng=np.random.default_rng(91592+step)
    plan,phases,_=prep.prepare(15513+step)
    def data():
        ids=rng.integers(len(pool[0]),size=128)
        return tuple(j.asarray(x[ids]) for x in pool)
    def starts():return j.asarray(rng.integers(1024//length,size=32),j.int32)
    # Fit on separate calibration text. Freeze the learned coefficients during each enumeration.
    cal=data();coef={}
    s=plan[0];start=starts()
    for part in (1,2):
        l=ctl.cheap(state.model,cal,s,start,part,length,k);h0=ctl.high(state.model,cal,s,part)
        mom=np.asarray(ctl.moments(state,l,h0,part))
        coef[part]=j.asarray(np.clip(mom[:,2]/np.maximum(1e-12,mom[:,0]),-3,3),j.float32)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(**meta,step=step,length=length,k=k,origin=src)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=os.urandom(6).hex(),group='nested-multifidelity-identity',mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,rows=[],outcomes_per_pair=320,coefficients={part:np.asarray(v).tolist() for part,v in coef.items()})
    for i in range(4):
        ds=[data(),data()]
        slots=[(a,a,c,c,e) for a,_,c,_,e in plan[1+2*i:3+2*i]]
        gold=[ready(native_call(native_audit,state.model,tuple(x[s[0]] for x in d),32)) for d,s in zip(ds,slots)]
        corrections=[ready(prep.discrepancy(state,d,s,phases)) for d,s in zip(ds,slots)]
        ref=[(np.asarray(g,np.float64)+np.asarray(hh,np.float64))*.5 for g,hh in zip(*map(jax.tree.leaves,gold))]
        avg=[np.zeros_like(x) for x in ref]
        for side,(d,s) in enumerate(zip(ds,slots)):
            u=starts()
            for kind,prob in enumerate((.2,.3,.3,.125,.075)):
                part=0 if kind==0 else 1 if kind<=2 else 2
                low=ctl.cheap(state.model,d,s,u,part,length,k) if part else None
                require_hi=kind in (2,4)
                for phase in (phases if require_hi else [None]):
                    hi=prep.raw_block(state.model,d,s[0],s[2],phase,part) if require_hi else None
                    for corr,w in ((None,.998),(corrections[1-side],.002)):
                        value=ctl.emitted((low,hi,part,coef.get(part),corr),state)
                        weight=.5*prob*w/(16 if require_hi else 1)
                        for x,g in zip(avg,jax.tree.leaves(value)):x+=np.asarray(g,np.float64)*weight
        diff=[(x-y).ravel() for x,y in zip(avg,ref)]
        relative=float(np.sqrt(sum(x@x for x in diff)/sum(x.ravel()@x.ravel() for x in ref)))
        mx=max(float(np.max(np.abs(x))) for x in diff)
        out['rows'].append(dict(panel=i,relative_l2=relative,max_coordinate=mx))
        publish(fs,ROOT+f'/coupled-control-identity/s{step}-l{length}-k{k}.json',out)
        run.log({'relative_l2':relative,'max_coordinate':mx})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

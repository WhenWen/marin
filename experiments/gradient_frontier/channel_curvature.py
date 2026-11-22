"""Fit channelwise local gradient drift at the starting checkpoint using paired random perturbations."""
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
from . import debias, fifty_blocks_strict as fixed, fused_attention
from .diagnostic_artifacts import publish
from .fifty_crossover import read_np
from .independent_audits import forecast
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready


@eqx.filter_jit
def fit(origin,cal,key):
    params,tree=jax.tree.flatten(eqx.filter(origin.model,eqx.is_inexact_array))
    keys=jax.random.split(key,len(params)+1)
    eps=[.02*j.maximum(j.sqrt(j.mean(w*w)),1.e-4) if w.size<2_000_000 else None for w in params]
    signs=[jax.random.rademacher(k,w.shape,dtype=j.float32) if e is not None else None for w,e,k in zip(params,eps,keys)]
    def probe(sign):
        weights=[w+sign*e*v if e is not None else w for w,e,v in zip(params,eps,signs)]
        model=eqx.combine(jax.tree.unflatten(tree,weights),eqx.filter(origin.model,lambda v:not eqx.is_inexact_array(v)))
        d=eqx.tree_at(lambda s:s.model,origin,model)
        return jax.tree.leaves(debias.draw(d,cal,keys[-1],4096,128,61,0,j.arange(1)))
    lo,hi=probe(-1),probe(1)
    base=jax.tree.leaves(forecast(origin,10))
    out=[]
    for w,e,s,a,b,m in zip(params,eps,signs,lo,hi,base):
        if e is None:
            out.append((j.empty(0),j.empty(0),j.empty(0)));continue
        anchor=.5*(a+b)-m
        diag=(b-a)*s/(2*e)
        if diag.ndim>=2:
            # Pool input coordinates; stacked layer and output-channel axes retain separate slopes.
            diag=diag.mean(-1,keepdims=True)
        else:diag=diag.mean()
        out.append((w,anchor,diag))
    return tuple(out)


def predictor(state,bank,mix,drift):
    m,td=jax.tree.flatten(forecast(state,10))
    params=jax.tree.leaves(eqx.filter(state.model,eqx.is_inexact_array))
    out=[]
    for w,b,(w0,a,hess) in zip(params,m,bank):
        if not a.size or not mix:out.append(b);continue
        change=(w-w0)*hess if drift else 0.
        if drift:
            ceiling=j.sqrt((j.mean(a*a)+j.mean(b*b)+1.e-15)/(j.mean(change*change)+1.e-15))
            change=change*j.minimum(1.,ceiling)
        out.append(b+mix*(a+change))
    return jax.tree.unflatten(td,out)


evaluate=eqx.filter_jit(predictor)


def main(step,cal_size):
    configure_storage();fs=fsspec.filesystem('s3')
    origin,_=recover(4001,fs);state,source=recover(step,fs);fused_attention.install()
    pool=read_np(fs,ROOT+'/inputs/native-packed.npz')
    cal=tuple(z[:cal_size] for z in pool);test=tuple(z[512:] for z in pool)
    settings=((0.,False),(.25,False),(.5,False),(.25,True),(.5,True),(1.,True))
    key=jax.random.key(85161)
    bank=ready(fit(origin,cal,key));prep=[]
    for _ in range(8):
        t=time.perf_counter();bank=ready(fit(origin,cal,key));prep.append(1000*(time.perf_counter()-t))
    pred=[ready(evaluate(state,bank,*s)) for s in settings]
    cfg=dict(step=step,source=source,calibration=cal_size,device=jax.devices()[0].device_kind,settings=settings,independent_population='native-packed rows 512 onward',origin_step=4001)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='channelwise-curvature-discovery',id=os.urandom(6).hex(),mode='online',config=cfg)
    dest=ROOT+f'/channel-curvature-v1/s{step}-cal{cal_size}.json'
    out=dict(**cfg,prep=prep,rows=[],wandb=run.url)
    for i in range(48):
        rng=np.random.default_rng(np.random.SeedSequence([61298,step,i]))
        ids=rng.integers(len(test[0]),size=(2,32))
        a,b=[tuple(z[ix] for z in test) for ix in ids]
        ka,kb,pa,pb=jax.random.split(jax.random.key(step*97+i),4)
        ga=fixed.raw_block(state,a,ka,1,jax.random.randint(pa,(),0,16,dtype=j.int32))
        gb=fixed.raw_block(state,b,kb,1,jax.random.randint(pb,(),0,16,dtype=j.int32))
        metrics=np.array([np.asarray(fixed.moments(ga,gb,p,1)) for p in pred])
        out['rows'].append(metrics.tolist())
        if (i+1)%8==0:publish(fs,dest,out)
    run.log({'prep_ms':np.mean(prep),'paired_draws':48});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

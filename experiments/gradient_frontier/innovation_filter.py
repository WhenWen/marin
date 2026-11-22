"""Predict the momentum forecast error from only past purchased block gradients."""
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
from . import block_audits,fifty_blocks_strict as fixed,fused_attention
from .diagnostic_artifacts import publish
from .fifty_crossover import read_np
from .independent_audits import forecast
from .native_eval import recover
from .run import ROOT,ready


@eqx.filter_jit
def observed(state,data,key):
    base,shift=jax.random.split(key)
    phase=jax.random.randint(shift,(),0,16,dtype=j.int32)
    raw=fixed.raw_block(state,data,base,1,phase)
    prior=[x for path,x in jax.tree_util.tree_leaves_with_path(forecast(state,10)) if not block_audits.readout(path)]
    return tuple(x-p for x,p in zip(raw,prior) if p.size<2_000_000)


@eqx.filter_jit
def initialize(label):
    fast=tuple(.5*x for x in label); slow=tuple(.25*x for x in label)
    return fast,slow,j.zeros((len(label),2,2)),j.zeros((len(label),2))


def slopes(gram,cross):
    scale=j.maximum(j.diagonal(gram,axis1=1,axis2=2).sum(-1),1.e-15)
    reg=.15*scale
    prior=j.broadcast_to(j.array([.5,.5]),cross.shape)
    fit=j.linalg.solve(gram+reg[:,None,None]*j.eye(2),(cross+reg[:,None]*prior)[:,:,None])[:,:,0]
    return j.clip(fit,-1.,2.)


@eqx.filter_jit
def score(aux,a,b):
    fast,slow,gram,cross=aux
    beta=slopes(gram,cross)
    # For independent observations, this is the exact paired change in squared prediction error.
    risk=j.zeros(4)
    for i,(f,s,x,y) in enumerate(zip(fast,slow,a,b)):
        xx=j.stack((f.ravel(),s.ravel()))
        covariance=xx@xx.T
        target=xx@(x+y).ravel()
        coeff=j.stack((j.array([1.,0.]),j.array([0.,1.]),j.array([.5,.5]),beta[i]))
        risk+=j.einsum('ri,ij,rj->r',coeff,covariance,coeff)-coeff@target
    return risk


@eqx.filter_jit
def advance(aux,label):
    fast,slow,gram,cross=aux
    if label is None:return tuple(.68*z for z in fast),tuple(.90*z for z in slow),gram,cross
    aa=[];bb=[]
    for f,s,y in zip(fast,slow,label):
        x=j.stack((f.ravel(),s.ravel()))
        aa.append(x@x.T);bb.append(x@y.ravel())
    return (tuple(.68*(.4*f+.6*y) for f,y in zip(fast,label)),tuple(.90*(.75*s+.25*y) for s,y in zip(slow,label)),
            .8*gram+j.stack(aa),.8*cross+j.stack(bb))


def main(seed):
    configure_storage();fs=fsspec.filesystem('s3');fused_attention.install()
    pool=read_np(fs,ROOT+'/inputs/native-packed.npz');cal=tuple(v[:512] for v in pool);test=tuple(v[512:] for v in pool)
    cfg=dict(seed=seed,algorithm='two causal filters on purchased gradient innovations, shrunk least squares per tensor',hardware=jax.devices()[0].device_kind,pt=.3,calibration=32,split=512)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='causal-purchased-gradient-filter',id=os.urandom(6).hex(),mode='online',config=cfg)
    out=dict(**cfg,wandb=run.url,states=[],settings=['fast','slow','half','adaptive'])
    def draw(seq,indices):return tuple(z[indices] for z in seq)
    for step in range(4001,4051):
        gen=np.random.default_rng(np.random.SeedSequence([seed,step,136715]))
        key=jax.random.key(seed+step*1103);state,src=recover(step,fs)
        if step==4001:
            data=draw(cal,gen.integers(len(cal[0]),size=32));aux=ready(initialize(observed(state,data,key)))
        obs=[]
        for i in range(8):
            a,b=gen.integers(len(test[0]),size=(2,32))
            ka,kb=jax.random.split(jax.random.fold_in(key,1+i))
            obs.append(np.asarray(score(aux,observed(state,draw(test,a),ka),observed(state,draw(test,b),kb))).tolist())
        gate=gen.random()<.3
        # A single independent training observation becomes available for the next step if the trunk was queried.
        feedback=observed(state,draw(cal,gen.integers(len(cal[0]),size=32)),jax.random.fold_in(key,97)) if gate else None
        aux=ready(advance(aux,feedback))
        out['states'].append(dict(step=step,checkpoint=src,trunk_event=bool(gate),paired_risk_change=obs))
        publish(fs,ROOT+f'/innovation-filter-v1/s{seed}.json',out)
        run.log(dict(step_index=step,adaptive_change_in_full_variance=np.mean(obs,0)[-1]*(1/.3-1)))
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(int(sys.argv[1]))

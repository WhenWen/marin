"""Exact conditional meeting-time tail and readout variance bound for a lagged independent-MH softmax gradient."""
import json
import os
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .diagnostic_artifacts import publish
from .fused_attention import install
from .native_eval import recover
from .proposal_chain import fit
from .run import ROOT,ready
from .tail import trunk


@eqx.filter_jit
def hidden(model,data):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    return trunk(m,data)


@eqx.filter_jit
def probe(x,w,target,bank,K):
    centers,table,counts=bank
    logits=(x@w.T).astype(j.float32)
    true=jax.nn.softmax(logits,axis=1)
    weight_scale=j.sum(x.astype(j.float32)**2,axis=1)
    V=w.shape[0];ids=j.arange(V);row=j.arange(len(x))
    base=j.log1p(1./(1.+ids))/j.log1p(float(V))
    region=j.argmax(jax.nn.standardize(x.astype(j.float32),axis=-1).astype(j.bfloat16)@centers.T,-1)
    learned=counts[region].astype(j.float32)/256.
    target_dist=jax.nn.one_hot(target,V)
    if K:
        short=j.zeros_like(true).at[:,:K].set(jax.nn.softmax(logits[:,:K],axis=1))
        proposal=.12*base+.12*target_dist+.16*learned+.60*short
    else:proposal=.35*base+.20*target_dist+.45*learned
    proposal/=proposal.sum(1,keepdims=True)
    importance=true/proposal;order=j.argsort(importance,axis=1)
    weights=j.take_along_axis(importance,order,1)
    ps=j.take_along_axis(true,order,1);qs=j.take_along_axis(proposal,order,1)
    cum=j.pad(j.cumsum(ps[:,:-1],axis=1),((0,0),(1,0)))
    a=j.clip(cum/j.maximum(weights,1e-30)+j.cumsum(qs[:,::-1],axis=1)[:,::-1],1.e-20,1.)
    at_y=j.argmax(order==target[:,None],axis=1)
    pair_accept=j.minimum(1.,weights/j.maximum(importance[row,target,None],1e-30))
    entry=qs*pair_accept*(order!=target[:,None])
    meet=j.minimum(a,a[row,at_y,None])
    # Unmet after the first proposal: unrounded signed gradient places +/-N on the heavier token.
    # Its competitor never occupies that token up to the first common accept.
    mean_n=j.sum(entry/meet,axis=1)
    n2=j.sum(entry*(2.-meet)/meet**2,axis=1)
    signal=j.sum((true-target_dist)**2,axis=1)
    bound=weight_scale*j.maximum(0.,n2-signal)
    tails=j.stack([j.sum(entry*(1-meet)**m,axis=1) for m in (0,1,4,16,64,256,1024)],axis=1)
    return j.concatenate((mean_n[:,None],n2[:,None],bound[:,None],(weight_scale*signal)[:,None],tails),axis=1)


def main(step,K,clusters):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    start=state if step==4001 else recover(4001,fs)[0]
    tmp=Path('/tmp/coupled-softmax-positions.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:
        train=tuple(j.asarray(z[k][:4]) for k in ('tokens','weights','segments'))
        pool=tuple(z[k][512:] for k in ('tokens','weights','segments'))
    seed=901318+step
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,K=K,clusters=clusters,seed=seed,source=src,device=jax.devices()[0].device_kind)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='unbiased-coupled-metropolis-softmax',id=os.urandom(5).hex(),config=cfg)
    bank=ready(fit(start.model,train,clusters,250))
    gen=np.random.default_rng(seed);out=dict(**cfg,wandb=run.url,rows=[],tail_depths=[0,1,4,16,64,256,1024])
    w=state.model.get_lm_head().rearrange(('vocab','embed')).array.astype(j.bfloat16)
    dest=ROOT+f'/coupled-softmax/s{step}-k{K}-c{clusters}.json'
    for i in range(96):
        ids=gen.integers(len(pool[0]),size=8);d=tuple(j.asarray(x[ids]) for x in pool)
        pos=gen.integers(0,1023,size=(8,8))
        xx=hidden(state.model,d)[j.arange(8)[:,None],pos].reshape(-1,w.shape[1])
        y=d[0][j.arange(8)[:,None],pos+1].reshape(-1)
        metrics=np.asarray(probe(xx,w,y,bank,K))
        assert np.isfinite(metrics).all()
        out['rows'].append(metrics.tolist())
        if (i+1)%8==0:publish(fs,dest,out)
    run.log({'mean_additional_transitions':np.mean(np.asarray(out['rows'])[:,:,0]),
             'mean_lower_bound_readout_variance':np.mean(np.asarray(out['rows'])[:,:,2])});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

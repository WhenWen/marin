"""Price always-on, heterogeneous minibatches for parameter groups sharing a single transformer forward."""
import os
import subprocess
import sys
import time

import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.layers.attention_mask import AttentionMask
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from functools import partial
from .independent_audits import copy_state, donated_baseline
from .lowmem import objective
from .native_eval import recover
from .run import ROOT, ready


def suffix_loss(model,data,key,depth):
    """Include the readout, final normalization, and the last depth transformer layers."""
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    tokens,weights,segments=data
    axes=(h.Axis('batch',tokens.shape[0]),h.Axis('position',tokens.shape[1]))
    mask=AttentionMask.causal().with_segment_ids(h.named(segments,axes))
    x=m.embeddings.embed(h.named(tokens,axes))
    layers=m.transformer.layers.unstacked()
    for i,layer in enumerate(layers):
        x=layer(x,mask)
        if i==len(layers)-depth-1:x=jax.lax.stop_gradient(x)
    hidden=m.transformer.norm(x).rearrange(('batch','position','embed')).array
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    offsets=jax.random.randint(key,(len(tokens),64),0,tokens.shape[1]//64)
    return objective(hidden,w,j.roll(tokens,-1,axis=1),weights,key,(6112288,64),offsets)


def pool_leaf(path,small,extra,depth,alpha,axes):
    name=jax.tree_util.keystr(path)
    if '.lm_head.' in name or '.transformer.norm.' in name:return small+alpha*(extra-small)
    if '.layers.stacked.' in name:
        axis=axes[name]
        blend=(j.arange(small.shape[axis])>=4-depth).reshape(tuple(small.shape[axis] if i==axis else 1 for i in range(small.ndim)))
        return small+j.where(blend,alpha*(extra-small),0.)
    return small


def estimate(model,base,extra,key,depth):
    a,b=jax.random.split(key)
    raw=eqx.filter_grad(suffix_loss)(model,base,a,4)
    if depth<0:return raw
    other=eqx.filter_grad(suffix_loss)(model,extra,b,depth)
    # Haliax's named layer dimension may occupy a different physical axis across leaves.
    axes={}
    for path,t in jax.tree_util.tree_leaves_with_path(model,is_leaf=lambda v:isinstance(v,h.NamedArray)):
        name=jax.tree_util.keystr(path)
        if '.layers.stacked.' in name and isinstance(t,h.NamedArray):
            axes[name+'[<flat index 0>]']=next(i for i,ax in enumerate(t.axes) if ax.name=='layer')
    alpha=extra[0].shape[0]/(base[0].shape[0]+extra[0].shape[0])
    return jax.tree_util.tree_map_with_path(lambda p,x,y:pool_leaf(p,x,y,depth,alpha,axes),raw,other)


@partial(eqx.filter_jit,donate='all-except-first')
def advance(args,state):
    base,extra,key,depth=args
    return state.take_step(estimate(state.model,base,extra,key,depth),key=jax.random.split(state.training_key)[1])[0]


@eqx.filter_jit
def variance(state,a,b,c,d,ka,kb,depth):
    left=estimate(state.model,a,b,ka,depth);right=estimate(state.model,c,d,kb,depth)
    return .5*sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(left),jax.tree.leaves(right)))


def main(step,base_size,extra_size):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);pool,meta=population(fs)
    fused_attention.install()
    cfg=dict(**meta,step=step,source=src,base_size=base_size,extra_size=extra_size,device=jax.devices()[0].device_kind,
             board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='depth-minibatch-allocation',id=os.urandom(6).hex(),mode='online',config=cfg)
    cases=[(61,12288,64,b) for b in (2,4,8,16,32)]+[(62,16384,64,b) for b in (2,4,8,16,32)]
    out=dict(**cfg,wandb=run.url,arms=[dict(depth=d,ms=[],pairs=[]) for d in (-1,0,1,2,3)],controls=[dict(config=c,ms=[],pairs=[]) for c in cases])
    rng=np.random.default_rng(np.random.SeedSequence([54132,step,base_size,extra_size]));dummy=j.arange(1)
    for i in range(25):
        a,b,c,d=ready([tuple(j.asarray(v[idx]) for v in pool) for idx in rng.integers(len(pool[0]),size=(4,max(32,extra_size,base_size)))])
        cut=lambda batch,n:tuple(v[:n] for v in batch)
        ab,ex,cb,ef=ready((cut(a,base_size),cut(b,extra_size),cut(c,base_size),cut(d,extra_size)))
        data=ready({n:(cut(a,n),cut(c,n)) for _,_,_,n in cases})
        ka,kb=jax.random.split(jax.random.key(i+91521))
        for choice in rng.permutation(len(cases)+5):
            s=copy_state(state);tic=time.perf_counter()
            if choice<5:
                row=out['arms'][choice];z=advance((ab,ex,ka,row['depth']),s)
            else:
                row=out['controls'][choice-5];m,k,n,B=row['config'];z=donated_baseline((data[B][0],ka,dummy,m,k,n),s)
            ready(z);ms=1000*(time.perf_counter()-tic)
            if i:row['ms'].append(ms)
        if not i:continue
        for row in out['arms']:row['pairs'].append(float(variance(state,ab,ex,cb,ef,ka,kb,row['depth'])))
        for row in out['controls']:
            m,k,n,B=row['config'];row['pairs'].append(float(pair(state,*data[B],ka,dummy,m,k,n)))
        if i%4==0:publish(fs,ROOT+f'/depth-minibatch/s{step}-b{base_size}-e{extra_size}.json',out)
    run.log({'pairs':24});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

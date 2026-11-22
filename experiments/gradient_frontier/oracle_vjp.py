"""Measure the attainable budget of a full trunk VJP with a free, precomputed exact readout adjoint."""
import functools
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .independent_audits import copy_state,donated_baseline
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit
from .tail import trunk


@eqx.filter_jit
def references(model,data):
    m=jax.tree.map(lambda v:v.astype(j.bfloat16),model);x=trunk(m,data)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(data[0],-1,axis=1);weight=data[1]/data[1].sum()
    def block(args):
        xx,yy,wt=args
        p=jax.nn.softmax((xx@w.T).astype(j.float32),axis=-1)
        p=p.at[j.arange(len(xx)),yy].add(-1.)
        return (p*wt[:,None]).astype(w.dtype)@w
    return jax.lax.map(block,(x.reshape(-1,128,x.shape[-1]),y.reshape(-1,128),weight.reshape(-1,128))).reshape(x.shape)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(args,state):
    data,key,adj,read=args
    def term(model):
        m=jax.tree.map(lambda v:v.astype(j.bfloat16),model)
        xx=trunk(m,data)
        return j.sum(xx.astype(j.float32)*adj.astype(j.float32))
    g=eqx.filter_grad(term)(state.model)
    g=eqx.tree_at(lambda v:v.lm_head.weight.array,g,read)
    return state.take_step(g,key=key)[0]


@eqx.filter_jit
def difference(a,b):
    return sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))/2


def main(step,band):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    original=Attention.__call__;fused_attention.install()
    file=Path('/tmp/oracle-vjp.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(file))
    with np.load(file) as z:pool=tuple(z[name][512:] for name in ('tokens','weights','segments'))
    sizes=(4,8,12,16,24,32,48,64) if not band else (32,48,64,80,96,128,160,192)
    ref=8;seed=38927+step+band;rng=np.random.default_rng(seed)
    def sample():
        ix=rng.integers(len(pool[0]),size=max(sizes))
        return {b:tuple(j.asarray(s[ix[:b]]) for s in pool) for b in {*sizes,ref}}
    def native(data):
        Attention.__call__=original
        try:return native_audit(state.model,data,ref)
        finally:fused_attention.install()
    dummy=j.arange(1);example=sample();adj={b:ready(references(state.model,example[b])) for b in sizes}
    read=ready(native(example[ref]).lm_head.weight.array)
    cases=[(m,k,n,b) for m,k,n in ((3,32768,0),(4,4096,0),(201,4096,0),(202,8192,0),(206,1024,0),(61,4096,32),(61,8192,64),(61,12288,64),(62,16384,64),(7,16384,64)) for b in sizes]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases];views=[dict(b=b,ms=[]) for b in sizes]
    cfg=dict(step=step,band=band,seed=seed,reference_batch=ref,checkpoint=src,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='free-readout-full-trunk-oracle',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,views=views,reference=[])
    path=ROOT+f'/oracle-vjp/{cfg["device"].split()[1]}-s{step}-r{band}.json'
    for it in range(41):
        a,b=ready((sample(),sample()));key=jax.random.key(seed+it)
        for ix in rng.permutation(np.arange(-len(sizes),len(cases))):
            work=copy_state(state);tic=time.perf_counter()
            if ix<0:
                sz=sizes[-ix-1];ready(advance((a[sz],key,adj[sz],read),work))
            else:
                m,k,n,sz=cases[ix];ready(donated_baseline((a[sz],key,dummy,m,k,n),work))
            ms=1000*(time.perf_counter()-tic)
            if it:(views[-ix-1] if ix<0 else rows[ix])['ms'].append(ms)
        if not it:continue
        out['reference'].append(float(difference(native(a[ref]),native(b[ref]))))
        for row,(m,k,n,sz) in zip(rows,cases):row['pair'].append(float(pair(state,a[sz],b[sz],key,dummy,m,k,n)))
        if it%5==0:publish(fs,path,out)
    run.log({'cycles':40});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(*map(int,sys.argv[1:]))

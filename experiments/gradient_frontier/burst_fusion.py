"""Measure consecutive native MuonW updates with predictable gradients, checking the full carried state."""
import functools
import json
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
from . import fused_attention,online_coin
from .causal_feedback import slim
from .independent_audits import copy_state,forecast
from .native_eval import recover
from .run import ROOT,ready


def one(state,aux,learn):
    key,nextkey=jax.random.split(state.training_key)
    if learn:
        prev,gram,cross=aux
        nxt,aa,bb=online_coin.step(state,None,None,key,12288,64,.2,.002,prev,gram,cross,next_key=nextkey)
        return nxt,(slim(eqx.filter(state.model,eqx.is_inexact_array)),aa,bb)
    return state.take_step(forecast(state,0),key=nextkey)[0],aux


@functools.partial(eqx.filter_jit,donate='all-except-first')
def burst(settings,work):
    length,learn,barrier=settings
    state,aux=work
    for i in range(length):
        state,aux=one(state,aux,learn)
        if barrier and i+1<length:
            def protect(x):
                if not eqx.is_array(x) or (barrier==1 and x.size>=2_000_000):return x
                return jax.lax.optimization_barrier(x)
            state,aux=jax.tree.map(protect,(state,aux))
    return state,aux


@eqx.filter_jit
def error(left,right):
    vals=[]
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        if not eqx.is_array(a):continue
        if a.size==0:continue
        if jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key):a,b=jax.random.key_data(a),jax.random.key_data(b)
        floating=j.issubdtype(a.dtype,j.inexact)
        word={2:j.uint16,4:j.uint32,8:j.uint64}.get(a.dtype.itemsize,j.uint8)
        aa,bb=(jax.lax.bitcast_convert_type(a,word),jax.lax.bitcast_convert_type(b,word)) if floating else (a,b)
        delta=a.astype(j.float32)-b.astype(j.float32)
        vals.append(j.stack([j.count_nonzero(aa!=bb).astype(j.float32),j.max(j.abs(delta)),
                             j.sum(delta*delta) if floating else 0.,j.sum(a.astype(j.float32)**2) if floating else 0.]))
    return j.stack(vals)


def main(step,learn,barrier=0):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);fused_attention.install()
    if learn:
        previous=state if step==4001 else recover(step-1,fs)[0]
        hist=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
        aux=ready(online_coin.prepare(state,previous,hist,step))
    else:aux=None
    dev=jax.devices()[0].device_kind;seed=45709+step+learn
    gen=np.random.default_rng(seed);lengths=(1,2,3,4,6,8,10)
    cfg=dict(step=step,learn=learn,barrier=barrier,source=src,device=dev,seed=seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fused-consecutive-native-updates',id=os.urandom(5).hex(),config=cfg)
    dst=ROOT+'/burst-fusion/'+f'{dev.split()[1]}-s{step}-learn{learn}'+(f'-bar{barrier}' if barrier else '')+'.json'
    rows=[dict(length=L,mode=mode,ms=[]) for L in lengths for mode in ('separate','fused')]
    out=dict(**cfg,wandb=run.url,rows=rows,checks={},cold=[])
    def start():return copy_state((state,aux))
    def execute(L,mode,work):
        if mode=='fused':return ready(burst((L,bool(learn),barrier),work))
        for _ in range(L):work=ready(burst((1,bool(learn),barrier),work))
        return work
    for L in lengths:
        lhs=execute(L,'separate',start())
        work=start();tic=time.perf_counter();rhs=execute(L,'fused',work);out['cold'].append(1000*(time.perf_counter()-tic))
        check=np.asarray(error(lhs,rhs));assert np.isfinite(check).all()
        out['checks'][L]=dict(different=int(check[:,0].sum()),max_abs=float(check[:,1].max()),
                              relative_l2=float(np.sqrt(check[:,2].sum()/max(1.,check[:,3].sum()))))
        if L==10:
            names=[jax.tree_util.keystr(p) for p,a in jax.tree_util.tree_leaves_with_path(lhs) if eqx.is_array(a) and a.size]
            assert len(names)==len(check)
            out['leaf_checks']=[dict(path=name,different=int(q[0]),max_abs=float(q[1])) for name,q in zip(names,check)]
        assert int(rhs[0].step)==step+L
        del lhs,rhs
        publish(fs,dst,out)
    for i in range(48):
        for ix in gen.permutation(len(rows)):
            q=rows[ix];work=start();tic=time.perf_counter();execute(q['length'],q['mode'],work)
            q['ms'].append(1000*(time.perf_counter()-tic))
        if (i+1)%6==0:publish(fs,dst,out)
    run.log({'cycles':48});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

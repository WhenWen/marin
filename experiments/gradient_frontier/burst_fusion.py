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
    length,learn=settings
    state,aux=work
    for _ in range(length):state,aux=one(state,aux,learn)
    return state,aux


@eqx.filter_jit
def error(left,right):
    vals=[]
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        if not eqx.is_array(a):continue
        if a.size==0:continue
        if jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key):a,b=jax.random.key_data(a),jax.random.key_data(b)
        delta=a.astype(j.float32)-b.astype(j.float32)
        vals.append(j.stack([j.count_nonzero(a!=b).astype(j.float32),j.max(j.abs(delta)),j.sum(delta*delta),j.sum(a.astype(j.float32)**2)]))
    return j.stack(vals)


def main(step,learn):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);fused_attention.install()
    if learn:
        previous=state if step==4001 else recover(step-1,fs)[0]
        hist=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
        aux=ready(online_coin.prepare(state,previous,hist,step))
    else:aux=None
    dev=jax.devices()[0].device_kind;seed=45709+step+learn
    gen=np.random.default_rng(seed);lengths=(1,2,3,4,6,8,10)
    cfg=dict(step=step,learn=learn,source=src,device=dev,seed=seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fused-consecutive-native-updates',id=os.urandom(5).hex(),config=cfg)
    dst=ROOT+'/burst-fusion/'+f'{dev.split()[1]}-s{step}-learn{learn}.json'
    rows=[dict(length=L,mode=mode,ms=[]) for L in lengths for mode in ('separate','fused')]
    out=dict(**cfg,wandb=run.url,rows=rows,checks={},cold=[])
    def start():return copy_state((state,aux))
    def execute(L,mode,work):
        if mode=='fused':return ready(burst((L,bool(learn)),work))
        for _ in range(L):work=ready(burst((1,bool(learn)),work))
        return work
    for L in lengths:
        lhs=execute(L,'separate',start())
        work=start();tic=time.perf_counter();rhs=execute(L,'fused',work);out['cold'].append(1000*(time.perf_counter()-tic))
        check=np.asarray(error(lhs,rhs));assert np.isfinite(check).all()
        out['checks'][L]=dict(different=int(check[:,0].sum()),max_abs=float(check[:,1].max()),
                              relative_l2=float(np.sqrt(check[:,2].sum()/max(1.,check[:,3].sum()))))
        assert int(rhs[0].step)==step+L
        del lhs,rhs
        fs.pipe(dst,json.dumps(out).encode())
    for i in range(48):
        for ix in gen.permutation(len(rows)):
            q=rows[ix];work=start();tic=time.perf_counter();execute(q['length'],q['mode'],work)
            q['ms'].append(1000*(time.perf_counter()-tic))
        if (i+1)%6==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':48});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

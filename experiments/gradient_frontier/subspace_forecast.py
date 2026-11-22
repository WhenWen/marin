"""Project a cheap current gradient onto the subspace of actually observed optimizer states.

The predictor is independent of the fresh native audit draw. No past evaluation labels enter its fit.
"""
import json
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
from .head_to_head import negative_loss, update
from .contest import pair
from .native_eval import recover
from .run import ROOT, ready
from .temporal import momentum
from .unbiased_check import native


def trunk_leaf(a):
    return a.size < 2_000_000


def features(state, prev, older, origin):
    cur, lag, lag2 = map(momentum, (state, prev, older))
    models=[eqx.filter(s.model, eqx.is_inexact_array) for s in (state, prev, older, origin)]
    # Retain trunk vectors only. Both vocabulary matrices together have <3% of the observed mean squared norm.
    def stack(m, m1, m2, w, w1, w2, w0):
        if not trunk_leaf(w):
            return j.empty((0,), j.float32)
        return j.stack((m,m1,m2,w-w1,w1-w2,w-w0)).reshape(6,-1)
    return jax.tree.map(stack,cur,lag,lag2,*models)


def predict(model, calibration, basis, key):
    # The current calibration draw can use a biased loss because it only estimates projection coefficients.
    cheap=eqx.filter_grad(negative_loss)(model,calibration,key,2048,4)
    def regress(g,f):
        if not trunk_leaf(g):
            return j.zeros_like(g)
        scales=j.sqrt(j.maximum(j.mean(f*f,axis=1,keepdims=True),1e-16))
        z=f/scales
        gram=z@z.T/f.shape[1]
        cross=z@g.reshape(-1)/f.shape[1]
        coef=jax.scipy.linalg.solve(gram+.06*j.eye(6),cross,assume_a='pos')
        return (coef@z).reshape(g.shape)
    return jax.tree.map(regress,cheap,basis)


@eqx.filter_jit
def make_prediction(state,cal,prev,older,origin,key):
    return predict(state.model,cal,features(state,prev,older,origin),key)


@eqx.filter_jit
def native_audit(model,data,micro):
    """Average independent native full-softmax gradients at a declared reference microbatch size."""
    K,t=data[0].shape
    if K==micro:
        return eqx.filter_grad(native)(model,data)
    assert K%micro==0
    groups=jax.tree.map(lambda x:x.reshape(-1,micro,t),data)
    def grad_at(i):
        return eqx.filter_grad(native)(model,jax.tree.map(lambda x:x[i],groups))
    def add(i,total):
        return jax.tree.map(j.add,total,grad_at(i))
    total=jax.lax.fori_loop(1,K//micro,add,grad_at(0))
    return jax.tree.map(lambda x:x/(K//micro),total)


@eqx.filter_jit
def correct(state,pred,actual,p,key):
    grad=jax.tree.map(lambda x,y:x+(y-x)/p,pred,actual)
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def common(state,pred,key):
    return state.take_step(pred,key=key)[0]


@eqx.filter_jit
def components(a,b,pred):
    delta=jax.tree.map(j.subtract,a,b)
    v=sum(j.vdot(v,v) for v in jax.tree.leaves(delta))*.5
    er=sum(j.vdot(x-h,y-h) for x,y,h in zip(*map(jax.tree.leaves,(a,b,pred))))
    return j.stack((v,er))


def main(step,K,micro,cal):
    configure_storage()
    fs=fsspec.filesystem('s3')
    state,src=recover(step,fs)
    prev,older,origin=[recover(s,fs)[0] for s in (max(4001,step-1),max(4001,step-2),4001)]
    native_attention=Attention.__call__
    tmp=Path('/tmp/subspace-pool.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as pack:
        pool=tuple(pack[k][512:] for k in ('tokens','weights','segments'))
    seed=9002+step+23*K+cal
    rng=np.random.default_rng(seed)
    def draw(B):
        idx=rng.integers(len(pool[0]),size=B)
        return tuple(j.asarray(v[idx]) for v in pool)
    def audit(d):
        Attention.__call__=native_attention
        try:
            return native_audit(state.model,d,micro)
        finally:
            fused_attention.install()
    fused_attention.install()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,K=K,micro=micro,cal=cal,checkpoint=src,seed=seed)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='online-subspace-native-audit',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,k,n,b) for m,k,n in ((3,32768,0),(201,4096,0),(206,1024,0),(61,12288,64)) for b in (1,2,4,8,12,16,24,32)]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    out=dict(**cfg,wandb=run.url,components=[],times=[],rows=rows,device=jax.devices()[0].device_kind)
    dst=ROOT+f'/subspace/{step}-K{K}-micro{micro}-cal{cal}.json'
    dummy=j.arange(1)
    for i in range(1+48):
        C,A,B,D=ready((draw(cal),draw(micro),draw(micro),draw(K)))
        ref1,ref2=ready((draw(32),draw(32)))
        key=jax.random.key(seed+i)
        def execute(case):
            if case>=0:
                m,k,n,b=cases[case]
                return update(state,tuple(x[:b] for x in ref1),key,dummy,m,k,n)
            pred=make_prediction(state,C,prev,older,origin,key)
            if case==-1:
                return common(state,pred,key)
            return correct(state,pred,audit(D),.25,key)
        # Compile and warm every executable once before collecting time.
        times=[None,None]
        for case in rng.permutation(np.arange(-2,len(cases))):
            start=time.perf_counter();ready(execute(case));ms=1e3*(time.perf_counter()-start)
            if i:
                if case<0: times[-case-1]=ms
                else:rows[case]['ms'].append(ms)
        if not i: continue
        h0=make_prediction(state,C,prev,older,origin,key)
        out['components'].append(np.asarray(components(audit(A),audit(B),h0)).tolist())
        out['times'].append(times)
        for row,(m,k,n,b) in zip(rows,cases):
            row['pair'].append(float(pair(state,tuple(x[:b] for x in ref1),tuple(x[:b] for x in ref2),key,dummy,m,k,n)))
        if i%8==0:
            fs.pipe(dst,json.dumps(out).encode())
    run.log({'prediction/error2':np.mean(out['components'],0)[1]});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

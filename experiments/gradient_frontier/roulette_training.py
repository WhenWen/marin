"""Train with independent routine and precision audits while preserving the original optimizer and RNG sequence."""
import functools
import hashlib
import io
import json
import time
import equinox as eqx
import jax
import jax.numpy as j
import numpy as np
from levanter.trainer import StepInfo
from . import debias, fused_attention
from .independent_audits import forecast,residual
from .subspace_forecast import native_audit
from .unbiased_check import native


@eqx.filter_jit
def true_loss(model,data):
    return native(model,data)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def update(inputs,state):
    data,delta,gradkey,next_key,p,r,scale=inputs
    g=forecast(state,scale)
    if data is not None:
        raw=debias.draw(state,data,gradkey,12288,64,61,0,j.arange(1))
        g=jax.tree.map(lambda h,x:h+(x-h)/p,g,raw)
    if delta is not None:g=jax.tree.map(lambda h,d:h+d/r,g,delta)
    return state.take_step(g,key=next_key)[0]


def initial_forecast(state):
    from .causal_feedback import slim
    previous=slim(eqx.filter(state.model,eqx.is_inexact_array))
    count=len(jax.tree.leaves(previous))
    return previous,j.zeros((count,2,2)),j.zeros((count,2))


def save_forecast(fs,checkpoint,aux):
    arrays=[np.asarray(a) for a in jax.tree.leaves(aux)]
    digest=hashlib.sha256()
    for a in arrays:
        digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    mem=io.BytesIO();np.savez(mem,**{f'v{i}':a for i,a in enumerate(arrays)})
    raw=mem.getvalue();dest=checkpoint.removeprefix('s3://')+'/forecast'
    meta=dict(bytes_sha=hashlib.sha256(raw).hexdigest(),logical_sha=digest.hexdigest(),arrays=len(arrays))
    fs.pipe(dest+'.npz',raw);fs.pipe(dest+'.json',json.dumps(meta).encode())
    return meta['logical_sha']


def restore_forecast(fs,checkpoint,state):
    stem=checkpoint.removeprefix('s3://')+'/forecast'
    meta=json.loads(fs.cat_file(stem+'.json'));raw=fs.cat_file(stem+'.npz')
    assert hashlib.sha256(raw).hexdigest()==meta['bytes_sha']
    arr,td=jax.tree.flatten(initial_forecast(state))
    assert len(arr)==meta['arrays']
    with np.load(io.BytesIO(raw)) as archive:
        values=[archive[f'v{i}'] for i in range(len(arr))]
    assert all(x.shape==y.shape and x.dtype==y.dtype for x,y in zip(values,arr))
    return jax.tree.unflatten(td,[j.asarray(a) for a in values])


@functools.partial(eqx.filter_jit,donate='all-except-first')
def learned_update(inputs,state):
    from .online_coin import step
    from .causal_feedback import slim
    a,delta,gkey,nextkey,p,r,aux=inputs
    nxt,aa,bb=step(state,a,delta,gkey,12288,64,p,r,*aux,next_key=nextkey)
    return nxt,(slim(eqx.filter(state.model,eqx.is_inexact_array)),aa,bb)


def with_audits(state,batch,p,r,scale,original_attention,aux=None):
    from levanter.layers.attention import Attention
    key,next_key=jax.random.split(state.training_key)
    primary,precision,select,gradkey,correction_key=jax.random.split(key,5)
    raw=tuple(v.rearrange(('batch','position')).array for v in (batch.tokens,batch.loss_weight,batch.attn_mask.segment_ids[0]))
    B=32
    select=jax.random.permutation(select,raw[0].shape[0])
    def subset(ids):return tuple(v[ids] for v in raw)
    # Report actual original CE on a fixed monitoring subset separately from the update timer.
    Attention.__call__=original_attention
    loss=float(true_loss(state.model,tuple(x[:4] for x in raw)))
    fused_attention.install()
    start=time.perf_counter()
    use_primary=bool(jax.random.bernoulli(primary,p))
    use_precision=bool(jax.random.bernoulli(precision,r))
    a=subset(select[:B]) if use_primary else None
    delta=None
    if use_precision:
        d=subset(select[B:2*B])
        Attention.__call__=original_attention
        try:truth=native_audit(state.model,d,B)
        finally:fused_attention.install()
        delta=residual(state,d,correction_key,12288,64,truth)
    if aux is None:nxt=update((a,delta,gradkey,next_key,p,r,scale),state)
    else:
        # A checkpoint at the start of the window may expose the same buffer as both current and previous weights.
        # Keep the forecast independent of the donated native state on every path.
        previous,gram,cross=aux
        independent=(jax.tree.map(lambda x:x.copy(),previous),gram,cross)
        nxt,aux=learned_update((a,delta,gradkey,next_key,p,r,independent),state)
    jax.block_until_ready(nxt)
    elapsed=time.perf_counter()-start
    event=dict(primary_draw=use_primary,precision_draw=use_precision,primary_probability=p,
               precision_probability=r,negative_momentum_scale=scale,reference_microbatch=B,
               train_loss_monitor_examples=4,estimator_update_ms=1000*elapsed)
    return StepInfo(nxt,loss,elapsed),event,aux

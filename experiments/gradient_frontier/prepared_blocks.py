"""Precompute independent sample panels and position offsets, fuse key advancement, store BF16 block gradients."""
import functools
import time
import equinox as eqx
import jax
import jax.numpy as j
import numpy as np
from . import block_audits,fifty_blocks_strict as strict
from .fifty_blocks import B,K,N,PT,PR,R
from .fifty_crossover import native_call
from .native_eval import objective
from .run import ready
from .subspace_forecast import native_audit


def prepare(seed):
    tic=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([seed,9138127]))
    slots=[]
    for _ in range(50):
        ids=rng.permutation(128)[:64]
        raw,extra=rng.integers(16,size=(2,B,N),dtype=np.int32)
        slots.append((j.asarray(ids[:B],j.int32),j.asarray(ids[B:],j.int32),j.asarray(raw),j.asarray(extra),j.asarray(rng.integers(16),j.int32)))
    phases=tuple(j.asarray(i,j.int32) for i in range(16))
    ready((slots,phases))
    return slots,phases,1000*(time.perf_counter()-tic)


@eqx.filter_jit
def raw_block(model,full,ids,base,phase,part):
    data=tuple(a[ids] for a in full)
    g=eqx.filter_grad(block_audits.masked_loss)(model,data,(base+phase)%16,jax.random.key(0),K,N,part)
    return tuple(v.astype(j.bfloat16) for path,v in jax.tree_util.tree_leaves_with_path(g) if block_audits.readout(path)==(part==2))


@eqx.filter_jit
def fp32(raw):return jax.tree.map(lambda a:a.astype(j.float32),raw)


@eqx.filter_jit
def add(total,raw):return jax.tree.map(lambda x,y:x+y.astype(j.float32),total,raw)


def discrepancy(state,data,slot,phases):
    _,ids,_,offset,_=slot
    gold=native_call(native_audit,state.model,tuple(x[ids] for x in data),B)
    sums=[]
    for part in (1,2):
        total=fp32(raw_block(state.model,data,ids,offset,phases[0],part))
        for phase in phases[1:]:total=add(total,raw_block(state.model,data,ids,offset,phase,part))
        sums.append(total)
    return strict.correction(gold,*sums,16)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    value,part,corr=inputs
    nxt=jax.random.split(state.training_key)[1]
    return state.take_step(strict.prediction(state,value,part,corr),key=nxt)[0]


def update(state,data,slot,phases,part,hit):
    ids,_,offset,_,phase=slot
    value=raw_block(state.model,data,ids,offset,phase,part) if part else None
    corr=discrepancy(state,data,slot,phases) if hit else None
    return advance((value,part,corr),state),hit


@functools.partial(eqx.filter_jit,donate='all-except-first')
def baseline(inputs,state):
    data,seed,m,k,n=inputs
    draw,nextkey=jax.random.split(state.training_key)
    g=eqx.filter_grad(objective)(state.model,data,j.arange(1),jax.random.fold_in(draw,seed),(100000*m+k,n))
    return state.take_step(g,key=nextkey)[0]

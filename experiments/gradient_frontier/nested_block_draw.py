"""Nest readout queries inside trunk queries and share the forward and backward when both are purchased."""
import functools
import equinox as eqx
import jax
import jax.numpy as j
from . import block_audits,debias,fifty_blocks_strict as fixed
from .fifty_blocks import PT,PR,R,B,K,N
from .fifty_crossover import native_call
from .independent_audits import forecast
from .subspace_forecast import native_audit


@eqx.filter_jit
def both(state,data,key,phase):
    return debias.draw(state,data,key,K,N,61,phase,j.arange(1))


def separate(tree,readout):
    return tuple(v for path,v in jax.tree_util.tree_leaves_with_path(tree) if block_audits.readout(path)==readout)


@eqx.filter_jit
def make_delta(native,partial,total,n):
    split=iter(partial);w=PR/PT
    return jax.tree_util.tree_map_with_path(lambda path,g,f:g-f/n if block_audits.readout(path) else g-((1-w)*next(split)+w*f)/n,native,total)


def discrepancy(state,data,key):
    width=data[0].shape[1]//N
    g=native_call(native_audit,state.model,data,B)
    t=fixed.cyclic_sum(state,data,key,1)
    fa=both(state,data,key,j.asarray(0,j.int32))
    for i in range(1,width):
        fa=fixed.add(fa,both(state,data,key,j.asarray(i,j.int32)))
    return make_delta(g,t,fa,width)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    mode,value,delta,nxt=inputs
    if mode==2:
        base=forecast(state,10)
        g=jax.tree_util.tree_map_with_path(lambda path,h,a:h+(a-h)/(PR if block_audits.readout(path) else PT),base,value)
        if delta is not None:g=jax.tree.map(lambda h,d:h+d/R,g,delta)
    else:g=fixed.prediction(state,value,mode,delta)
    return state.take_step(g,key=nxt)[0]


def update(state,data,gate,seed):
    part,hit,indices=gate
    key,nxt=jax.random.split(state.training_key)
    k,ph,dk=jax.random.split(jax.random.fold_in(key,seed),3)
    value=None
    if part:
        sample=tuple(v[indices[:B]] for v in data)
        phase=jax.random.randint(ph,(),0,sample[0].shape[1]//N,dtype=j.int32)
        value=both(state,sample,k,phase) if part==2 else fixed.raw_block(state,sample,k,1,phase)
    delta=discrepancy(state,tuple(v[indices[B:]] for v in data),dk) if hit else None
    return advance((part,value,delta,nxt),state),hit


def measurement(state,data,key):
    k,ph,coin=jax.random.split(key,3)
    phase=jax.random.randint(ph,(),0,data[0].shape[1]//N,dtype=j.int32)
    full=both(state,data,k,phase)
    trunk=separate(full,False) if bool(jax.random.bernoulli(coin,PR/PT)) else fixed.raw_block(state,data,k,1,phase)
    return trunk,separate(full,True)

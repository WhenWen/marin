"""Use identical materialized raw-gradient executables for routine draws and native audits."""
import functools
import equinox as eqx
import jax
import jax.numpy as jnp
from . import block_audits
from .fifty_blocks import B,K,N,PT,PR,R
from .fifty_crossover import native_call
from .independent_audits import forecast
from .subspace_forecast import native_audit


@eqx.filter_jit
def raw_block(state,data,key,part,phase):
    g=block_audits.draw(state,data,key,K,N,part,phase)
    return tuple(v for path,v in jax.tree_util.tree_leaves_with_path(g) if block_audits.readout(path)==(part==2))


@eqx.filter_jit
def add(x,y):
    return jax.tree.map(jnp.add,x,y)


def cyclic_sum(state,data,key,part):
    width=data[0].shape[1]//N
    total=raw_block(state,data,key,part,jnp.asarray(0,jnp.int32))
    for i in range(1,width):
        total=add(total,raw_block(state,data,key,part,jnp.asarray(i,jnp.int32)))
    return total


@eqx.filter_jit
def correction(gold,trunk,head,width):
    blocks=iter(trunk),iter(head)
    return jax.tree_util.tree_map_with_path(lambda path,v:v-next(blocks[int(block_audits.readout(path))])/width,gold)


def discrepancy(state,data,key):
    gold=native_call(native_audit,state.model,data,B)
    a=cyclic_sum(state,data,key,1)
    b=cyclic_sum(state,data,key,2)
    return correction(gold,a,b,data[0].shape[1]//N)


def prediction(state,raw,part,delta):
    prior=forecast(state,10)
    if part:
        seq=iter(raw)
        p=PT if part==1 else PR
        prior=jax.tree_util.tree_map_with_path(lambda path,x:x+(next(seq)-x)/p if block_audits.readout(path)==(part==2) else x,prior)
    if delta is not None:
        prior=jax.tree.map(lambda x,d:x+d/R,prior,delta)
    return prior


emitted=eqx.filter_jit(prediction)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    raw,part,delta,nextkey=inputs
    return state.take_step(prediction(state,raw,part,delta),key=nextkey)[0]


def update(state,data,gate,seed,repair=discrepancy):
    part,hit,indices=gate
    stepkey,nextkey=jax.random.split(state.training_key)
    gkey,phasekey,dkey=jax.random.split(jax.random.fold_in(stepkey,seed),3)
    value=None
    if part:
        batch=tuple(v[indices[:B]] for v in data)
        phase=jax.random.randint(phasekey,(),0,batch[0].shape[1]//N,dtype=jnp.int32)
        value=raw_block(state,batch,gkey,part,phase)
    delta=repair(state,tuple(v[indices[B:]] for v in data),dkey) if hit else None
    return advance((value,part,delta,nextkey),state),hit


@eqx.filter_jit
def moments(a,b,prior,part):
    targets=[x for path,x in jax.tree_util.tree_leaves_with_path(prior) if block_audits.readout(path)==(part==2)]
    return jnp.stack((sum(jnp.vdot(x-y,x-y) for x,y in zip(a,b))*.5,sum(jnp.vdot(x-p,y-p) for x,y,p in zip(a,b,targets))))

"""Return the raw cotangent from the same executable that performs an ordinary MuonW update."""
import functools

import equinox as eqx
import jax
import jax.numpy as j

from . import block_audits,fifty_blocks_strict as strict,prepared_blocks as prep
from .fifty_blocks import B,K,N
from .fifty_crossover import native_call
from .independent_audits import copy_state
from .subspace_forecast import native_audit

prepare=prep.prepare
baseline=prep.baseline
fp32=prep.fp32


@functools.partial(eqx.filter_jit,donate='all-except-first')
def ordinary(inputs,state):
    full,ids,base,phase,part=inputs
    val=None
    if part:
        data=tuple(v[ids] for v in full)
        g=eqx.filter_grad(block_audits.masked_loss)(state.model,data,(base+phase)%16,jax.random.key(0),K,N,part)
        val=tuple(v.astype(j.bfloat16) for path,v in jax.tree_util.tree_leaves_with_path(g) if block_audits.readout(path)==(part==2))
    key=jax.random.split(state.training_key)[1]
    nxt=state.take_step(strict.prediction(state,val,part,None),key=key)[0]
    return nxt,val


def raw_from_state(state,full,ids,base,phase,part):
    # Same executable and donated argument layout are used in routine and correction paths.
    unused,val=ordinary((full,ids,base,phase,part),copy_state(state))
    jax.block_until_ready(unused)
    return val


def discrepancy(state,data,slot,phases):
    _,ids,_,offset,_=slot
    gold=native_call(native_audit,state.model,tuple(x[ids] for x in data),B)
    sums=[]
    for part in (1,2):
        total=None
        for phase in phases:
            value=raw_from_state(state,data,ids,offset,phase,part)
            total=fp32(value) if total is None else prep.add(total,value)
        sums.append(total)
    return strict.correction(gold,*sums,16)


def update(state,data,slot,phases,part,hit):
    ids,_,offset,_,phase=slot
    if not hit:
        nxt,val=ordinary((data,ids,offset,phase,part),state)
        jax.block_until_ready(val)
        return nxt,False
    delta=discrepancy(state,data,slot,phases)
    val=raw_from_state(state,data,ids,offset,phase,part) if part else None
    return prep.advance((val,part,delta),state),True

"""Vary only the two block purchase probabilities along the strict estimator's cost–variance curve."""
import functools
import equinox as eqx
import jax
import jax.numpy as jnp
from . import block_audits,fifty_blocks_strict as strict
from .fifty_blocks import B,N,R
from .independent_audits import forecast

RATES={411:(.15,.0375),412:(.225,.05625),402:(.30,.075),413:(.45,.1125),414:(.60,.15)}


@functools.partial(eqx.filter_jit,donate='all-except-first')
def step(inputs,state):
    raw,part,delta,nextkey,pt,pr=inputs
    g=forecast(state,10)
    if part:
        seq=iter(raw);prob=pt if part==1 else pr
        g=jax.tree_util.tree_map_with_path(lambda path,v:v+(next(seq)-v)/prob if block_audits.readout(path)==(part==2) else v,g)
    if delta is not None:g=jax.tree.map(lambda v,d:v+d/R,g,delta)
    return state.take_step(g,key=nextkey)[0]


def update(state,data,gate,seed,method):
    part,hit,indices=gate
    seedkey,nxt=jax.random.split(state.training_key)
    key,phase_key,dkey=jax.random.split(jax.random.fold_in(seedkey,seed),3)
    value=None
    if part:
        batch=tuple(x[indices[:B]] for x in data)
        phase=jax.random.randint(phase_key,(),0,batch[0].shape[1]//N,dtype=jnp.int32)
        value=strict.raw_block(state,batch,key,part,phase)
    delta=strict.discrepancy(state,tuple(x[indices[B:]] for x in data),dkey) if hit else None
    return step((value,part,delta,nxt,*RATES[method]),state),hit

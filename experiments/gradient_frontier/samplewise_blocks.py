"""Anchor the rare block correction to 32 actual single-sequence native backward calls."""
import jax
import numpy as np
from . import fifty_blocks_strict as strict
from .fifty_crossover import native_call
from .samplewise_native import serial
from .subspace_forecast import native_audit


def discrepancy(state,data,key):
    g=native_call(serial,state.model,data)
    return strict.correction(g,strict.cyclic_sum(state,data,key,1),strict.cyclic_sum(state,data,key,2),16)


def host_mean(model,data):
    acc=None
    for i in range(data[0].shape[0]):
        g=native_call(native_audit,model,tuple(x[i:i+1] for x in data),1)
        if acc is None:acc=[np.asarray(x,np.float64).copy() for x in jax.tree.leaves(g)]
        else:
            for a,x in zip(acc,jax.tree.leaves(g)):a+=np.asarray(x,np.float64)
    for a in acc:a/=data[0].shape[0]
    return acc

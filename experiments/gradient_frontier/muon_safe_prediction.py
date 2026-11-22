"""Constrain calibrated momentum forecasts to preserve the direction of the Muon Nesterov input."""
import equinox as eqx
import jax
import jax.numpy as j
from . import block_audits,fifty_blocks_strict as fixed
from .temporal import momentum


@eqx.filter_jit
def calibrate(state,data1,data2):
    b=momentum(state)
    raw={}
    for part in (1,2):
        a=fixed.raw_block(state,data1,jax.random.key(7331),part,j.asarray(1,j.int32))
        c=fixed.raw_block(state,data2,jax.random.key(62374),part,j.asarray(9,j.int32))
        raw[part]=iter(zip(a,c))
    coefficients=[]
    for path,v in jax.tree_util.tree_leaves_with_path(b):
        a,c=next(raw[1+int(block_audits.readout(path))])
        dot=.5*(j.vdot(a,v)+j.vdot(c,v))
        coef=dot/j.maximum(j.vdot(v,v),1e-12)
        # Bound the forecast for a Muon matrix to leave at least 10% of beta² momentum before orthogonalization.
        coefficients.append(j.clip(coef,-.9*.95**2/(.05*1.95),16.) if v.size<2000000 else j.zeros((),j.float32))
    return tuple(coefficients)


def predict(state,coef,scale):
    b=momentum(state)
    if scale>=0:
        return jax.tree.map(lambda x:-scale*x if x.size<2000000 else j.zeros_like(x),b)
    leaves,td=jax.tree.flatten(b)
    return jax.tree.unflatten(td,[a*x if x.size<2000000 else j.zeros_like(x) for a,x in zip(coef,leaves)])


@eqx.filter_jit
def next_model(state,coef,scale):
    return state.take_step(predict(state,coef,scale),key=jax.random.split(state.training_key)[1])[0].model


def matrix_paths(state):
    return {jax.tree_util.keystr(p).split('.momentum_buffer',1)[1] for p,_ in jax.tree_util.tree_leaves_with_path(state.opt_state) if '.momentum_buffer.' in jax.tree_util.keystr(p)}


@eqx.filter_jit
def agreement(state,reference,zero,test):
    names=matrix_paths(state)
    out=j.zeros(6,j.float32)
    for (path,old),a,b,c in zip(jax.tree_util.tree_leaves_with_path(state.model),*map(jax.tree.leaves,(reference,zero,test))):
        if jax.tree_util.keystr(path) not in names:continue
        ref=a-old;no=b-old;new=c-old
        out=out+j.stack([j.vdot(ref,ref),j.vdot(no,no),j.vdot(new,new),j.vdot(ref,new),j.vdot(no,new),j.vdot(ref,no)])
    return out

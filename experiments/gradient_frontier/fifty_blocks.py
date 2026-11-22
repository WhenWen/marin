"""Fixed, predictable block sampling with a fifty-step gate schedule and an independent native precision audit."""
import functools
import time
import equinox as eqx
import jax
import jax.numpy as j
import numpy as np
from levanter.layers.attention import Attention
from . import block_audits,fused_attention
from .independent_audits import forecast
from .run import ready
from .subspace_forecast import native_audit
ORIGINAL=Attention.__call__
B,K,N=32,12288,64
PT,PR,R=.30,.075,.002


def schedule(seed):
    """The scheduled coins are used only at their own step, independently of model and data."""
    tic=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,518319]))
    u=rng.random(50);parts=np.where(u<PT,1,np.where(u<PT+PR,2,0));audit=rng.random(50)<R
    indices=ready(tuple(j.asarray(rng.permutation(128)[:2*B],j.int32) for _ in range(50)))
    return tuple((int(a),bool(b),c) for a,b,c in zip(parts,audit,indices)),(time.perf_counter()-tic)*1000


@functools.partial(eqx.filter_jit,donate='all-except-first')
def apply(inputs,state):
    data,indices,part,gkey,nextkey,delta=inputs
    pred=forecast(state,10)
    if part:
        small=tuple(v[indices[:B]] for v in data)
        g=block_audits.draw(state,small,gkey,K,N,part,0)
        p=PT if part==1 else PR
        pred=jax.tree_util.tree_map_with_path(lambda path,a,v:a+(v-a)/p if block_audits.readout(path)==(part==2) else a,pred,g)
    if delta is not None:pred=jax.tree.map(lambda x,v:x+v/R,pred,delta)
    return state.take_step(pred,key=nextkey)[0]


def update(state,data,gate,seed):
    part,hit,indices=gate
    key,nextkey=jax.random.split(state.training_key);gkey,dkey=jax.random.split(jax.random.fold_in(key,seed))
    delta=None
    if hit:
        extra=tuple(a[indices[B:]] for a in data)
        Attention.__call__=ORIGINAL
        try:gold=native_audit(state.model,extra,B)
        finally:fused_attention.install()
        delta=block_audits.strict(state,extra,dkey,K,N,gold)
    return apply((data,indices,part,gkey,nextkey,delta),state),hit

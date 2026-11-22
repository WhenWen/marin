"""Attribute the trace of the gradient covariance to parameter blocks on actual packed minibatches."""
import json,os,sys
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .fused_attention import install
from .run import ROOT

@eqx.filter_jit
def measure(model,a,b,k):
    ga=eqx.filter_grad(objective)(model,a,j.arange(1),k,(0,0))
    gb=eqx.filter_grad(objective)(model,b,j.arange(1),k,(0,0))
    vals=[]
    for x,y in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
        d=x-y
        vals.append(.5*j.vdot(d,d))
    return j.stack(vals)
def main(step,b):
    configure_storage();f=fsspec.filesystem('s3');install();state,src=recover(step,f)
    tmp=Path('/tmp/variance-map.npz');f.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:hold=tuple(z[a][512:] for a in ('tokens','weights','segments'))
    keys=[jax.tree_util.keystr(p) for p,v in jax.tree_util.tree_leaves_with_path(eqx.filter(state.model,eqx.is_inexact_array))]
    rng=np.random.default_rng(733+step);samples=[]
    for i in range(128):
        left,right=[tuple(j.asarray(a[idx]) for a in hold) for idx in rng.integers(len(hold[0]),size=(2,b))]
        samples.append(np.asarray(measure(state.model,left,right,jax.random.key(step+i))).tolist())
    assert len(keys)==len(samples[0])
    f.pipe(ROOT+f'/variance-map/step{step}-b{b}.json',json.dumps(dict(step=step,batch=b,parameters=keys,pairs=samples,source=src)).encode())
if __name__=='__main__':main(*map(int,sys.argv[1:]))

"""Compress predictive Gram matrices for raw-gradient forecasts from previously observed optimizer states."""
import json,sys
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .run import ROOT,ready
from .native_eval import recover,objective
from .fused_attention import install
from .temporal import momentum


def sub(x,y):return jax.tree.map(j.subtract,x,y)
def model(s):return eqx.filter(s.model,eqx.is_inexact_array)


@eqx.filter_jit
def compress(basis):
    columns=[jax.tree.leaves(x) for x in basis];grams=[]
    for i in range(len(columns[0])):
        vectors=[v[i].reshape(-1) for v in columns]
        grams.append(j.stack([j.stack([j.vdot(x,y) for y in vectors]) for x in vectors]))
    return j.stack(grams)


@eqx.filter_jit
def labels(state,d1,d2,basis,key):
    k1,k2=jax.random.split(key)
    g=eqx.filter_grad(objective)(state.model,d1,j.arange(1),k1,(0,0))
    g2=eqx.filter_grad(objective)(state.model,d2,j.arange(1),k2,(0,0))
    cols=list(map(jax.tree.leaves,basis));ret=[]
    for i,(u,v) in enumerate(zip(jax.tree.leaves(g),jax.tree.leaves(g2))):
        yz=j.stack([j.vdot(u,cc[i]) for cc in cols])
        ret.append(j.concatenate((j.stack((j.vdot(u,v),.5*j.vdot(u-v,u-v))),yz)))
    return j.stack(ret)


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');install()
    state=recover(step,fs)[0];prev=recover(max(4001,step-1),fs)[0];older=recover(max(4001,step-2),fs)[0];origin=recover(4001,fs)[0]
    # All state features are available prior to the current gradient draw.
    curm,prevm,oldm=map(momentum,(state,prev,older))
    basis=(curm,prevm,oldm,sub(model(state),model(prev)),sub(model(prev),model(older)),sub(model(state),model(origin)))
    gram=np.asarray(compress(basis));names=[jax.tree_util.keystr(p) for p,_ in jax.tree_util.tree_leaves_with_path(model(state))]
    f=Path('/tmp/history-pool.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:hold=tuple(z[a][512:] for a in ('tokens','weights','segments'))
    raw=[];rng=np.random.default_rng(6110+step)
    for i in range(128):
        a,b=[tuple(j.asarray(z[ix]) for z in hold) for ix in rng.integers(len(hold[0]),size=(2,8))]
        raw.append(np.asarray(labels(state,a,b,basis,jax.random.key(step+i))).tolist())
    fs.pipe(ROOT+f'/history-prediction/s{step}.json',json.dumps(dict(step=step,gram=gram.tolist(),names=names,pairs=raw)).encode())
if __name__=='__main__':main(int(sys.argv[1]))

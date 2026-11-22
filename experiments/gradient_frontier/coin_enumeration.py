"""Exhaust the two coin tosses and every cyclic vocabulary correction on real native batches."""
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
from levanter.layers.attention import Attention
from experiments.hybrid_transfer.storage import configure_storage
from . import debias, fused_attention
from .independent_audits import forecast
from .native_eval import recover
from .nested_forecast import difference
from .subspace_forecast import native_audit
from .run import ROOT,ready


@eqx.filter_jit
def mean_coins(state,data,delta,key,scale,p,r):
    pred=forecast(state,scale)
    def conditional(i):
        raw=debias.draw(state,data,key,12288,64,61,i,j.arange(1))
        def one(P,H,D):
            base=P+(H-P)/p
            # Materialize each floating-point estimator just as on a realized event.
            branches=(P,base,P+D/r,base+D/r)
            pr=((1-p)*(1-r),p*(1-r),(1-p)*r,p*r)
            return sum(v*q for v,q in zip(branches,pr))
        return jax.tree.map(one,pred,raw,delta)
    total=jax.lax.fori_loop(1,16,lambda i,a:jax.tree.map(j.add,a,conditional(i)),conditional(0))
    return jax.tree.map(lambda a:a/16,total)


@eqx.filter_jit
def compare(x,y,a,b):
    err=jax.tree.map(lambda x,y,a,b:.5*((x+y)-(a+b)),x,y,a,b)
    ref=jax.tree.map(lambda a,b:.5*(a+b),a,b)
    norm=lambda z:sum(j.vdot(v,v) for v in jax.tree.leaves(z))
    return j.stack((j.sqrt(norm(err)/norm(ref)),j.max(j.stack([j.max(j.abs(v)) for v in jax.tree.leaves(err)]))))


def main(step,B,scale):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    tmp=Path('/tmp/coin-check.npz');fs.get_file(ROOT+'/inputs/native-confirm2.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[k] for k in ('tokens','weights','segments'))
    rng=np.random.default_rng(75103+step)
    original=Attention.__call__
    def native(d):
        Attention.__call__=original
        try:return native_audit(state.model,d,B)
        finally:fused_attention.install()
    metrics=[]
    for i in range(6):
        a,b=ready(tuple(tuple(j.asarray(v[ix]) for v in pool) for ix in rng.integers(len(pool[0]),size=(2,B))))
        ka,kb=jax.random.split(jax.random.key(941+i))
        ga,gb=native(a),native(b)
        _,ha=debias.audit(state,a,ka,12288,64,61,j.arange(1));_,hb=debias.audit(state,b,kb,12288,64,61,j.arange(1))
        # Each ordered pair uses the other batch for the numerical audit.
        # Averaging over which batch is routine gives the mean native gradient of the two-panel population.
        left=mean_coins(state,a,difference(gb,hb),ka,scale,.2,.002)
        right=mean_coins(state,b,difference(ga,ha),kb,scale,.2,.002)
        metrics.append(np.asarray(compare(left,right,ga,gb)).tolist())
    out=dict(step=step,B=B,p=.2,r=.002,scale=scale,pairs=len(metrics),outcomes_per_pair=2*16*4,metrics=metrics,source=src)
    fs.pipe(ROOT+f'/coin-enumeration/s{step}-b{B}-c{scale}.json',json.dumps(out).encode())


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

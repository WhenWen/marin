"""Enumerate native gradients and every signed control path on actual held-out text."""
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
import haliax as h
from levanter.layers.attention import Attention
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention
from .native_eval import recover
from .signed_panel import control,corrected
from .subspace_forecast import native_audit
from .run import ROOT


@eqx.filter_jit
def add(x,y):
    return jax.tree.map(j.add,x,y)


@eqx.filter_jit
def errors(estimate,reference,correction,M):
    def norm(x):return sum(j.vdot(a,a) for a in jax.tree.leaves(x))
    r=jax.tree.map(lambda x:x/M,reference);e=jax.tree.map(lambda x,y:(x-y)/M,estimate,reference)
    largest=[j.max(j.abs(v)) for v in jax.tree.leaves(e)]
    return j.stack((j.sqrt(norm(e)/norm(r)),j.max(j.stack(largest)),j.sqrt(norm(correction))/M,j.sqrt(norm(r))))


def main(step,b,M,k):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    tmp=Path('/tmp/signed-test.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[n] for n in ('tokens','weights','segments'))
    orig=Attention.__call__;fused_attention.install()
    def true(d):
        Attention.__call__=orig
        try:return native_audit(state.model,d,b)
        finally:fused_attention.install()
    rng=np.random.default_rng(1824+step);results=[]
    for trial in range(8):
        ix=rng.integers(len(pool[0]),size=(M,b));a=tuple(j.asarray(v[ix]) for v in pool)
        for leaf in range(M):
            bits=j.asarray([(leaf>>t)&1 for t in reversed(range(M.bit_length()-1))],j.int32)
            d,selected=control(state.model,a,bits,k);original=jax.tree.map(lambda v:v[leaf],a)
            assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(selected,original))
            g=true(selected);ref=true(original);est=corrected(g,d)
            cor=jax.tree.map(lambda x,v:x if x.size else j.zeros_like(v),d,g)
            if leaf==0:total=est;gold=ref;csum=cor
            else:total=add(total,est);gold=add(gold,ref);csum=add(csum,cor)
        results.append(np.asarray(errors(total,gold,csum,M)).tolist())
    fs.pipe(ROOT+f'/signed-check/s{step}-b{b}-M{M}-k{k}.json',json.dumps(dict(step=step,b=b,M=M,k=k,checkpoint=src,paths=M,panels=8,metrics=results)).encode())
    print('max relative discrepancy',np.max(results,axis=0)[0],flush=True)


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

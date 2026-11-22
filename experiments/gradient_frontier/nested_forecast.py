"""Online projection, large debiased audits and sparse finite-precision corrections to the native gradient."""
import json
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import debias,fused_attention
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import make_prediction,native_audit,common,correct,components


@eqx.filter_jit
def large_control(state,data,key,micro):
    arrays=jax.tree.map(lambda x:x.reshape((-1,micro,x.shape[-1])),data)
    n=len(arrays[0]);keys=jax.random.split(key,n)
    def one(i):
        d=jax.tree.map(lambda x:x[i],arrays)
        return debias.draw(state,d,keys[i],12288,64,61,0,j.arange(1))
    total=jax.lax.fori_loop(1,n,lambda i,a:jax.tree.map(j.add,a,one(i)),one(0))
    return jax.tree.map(lambda x:x/n,total)


@eqx.filter_jit
def difference(a,b):
    return jax.tree.map(j.subtract,a,b)


@eqx.filter_jit
def add_control(x,d,r):
    return jax.tree.map(lambda v,w:v+w/r,x,d)


@eqx.filter_jit
def norm_moments(x,y):
    def inner(a,b):return sum(j.vdot(v,w) for v,w in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
    return j.stack((.5*(inner(x,x)+inner(y,y)),inner(x,y)))


def main(step,K,micro,cal):
    configure_storage();fs=fsspec.filesystem('s3');state,source=recover(step,fs)
    prev,older,start=[recover(s,fs)[0] for s in (max(4001,step-1),max(4001,step-2),4001)]
    tmp=Path('/tmp/nested-input.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[k][512:] for k in ('tokens','weights','segments'))
    original=Attention.__call__;fused_attention.install();seed=3299+step+K*cal;rng=np.random.default_rng(seed)
    def native(d):
        Attention.__call__=original
        try:return native_audit(state.model,d,micro)
        finally:fused_attention.install()
    def discrepancy(d,key):
        _,avg=debias.audit(state,d,key,12288,64,61,j.arange(1))
        return difference(native(d),avg)
    def draw(b):
        ix=rng.integers(len(pool[0]),size=b)
        return tuple(j.asarray(z[ix]) for z in pool)
    cfg=dict(step=step,K=K,micro=micro,cal=cal,checkpoint=source,seed=seed,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='strict-two-audit-prediction',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,k,n,b) for m,k,n in ((61,12288,64),(62,16384,64),(7,16384,64)) for b in (1,2,4,8,16,24,32)]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    out=dict(**cfg,wandb=run.url,components=[],times=[],rows=rows)
    dst=ROOT+f'/nested-forecast/s{step}-K{K}-m{micro}-c{cal}.json';dummy=j.arange(1)
    for i in range(33):
        C,A,B,large,extra,extra2,ref1,ref2=ready(tuple(draw(n) for n in (cal,micro,micro,K,micro,micro,32,32)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(seed+i),4)
        # Materialize the baseline views ahead of the clocks, exactly as for the candidate arrays.
        L,R=ready(tuple({b:tuple(v[:b] for v in d) for _,_,_,b in cases} for d in (ref1,ref2)))
        def execute(c):
            if c>=0:
                m,k,n,b=cases[c];return update(state,L[b],ka,dummy,m,k,n)
            p=make_prediction(state,C,prev,older,start,kc)
            if c==-1:return common(state,p,ka)
            control=large_control(state,large,kb,micro)
            if c==-3:control=add_control(control,discrepancy(extra,kd),.125)
            return correct(state,p,control,.25,ka)
        times=[0.]*3
        for c in rng.permutation(np.arange(-3,len(cases))):
            t=time.perf_counter();ready(execute(c));ms=1000*(time.perf_counter()-t)
            if i:
                if c<0:times[-c-1]=ms
                else:rows[c]['ms'].append(ms)
        if not i:continue
        p=make_prediction(state,C,prev,older,start,kc)
        # Paired native gradients are independent of the fitted predictor, and share it within this risk estimate.
        err=components(native(A),native(B),p)[1]
        v=components(large_control(state,A,ka,micro),large_control(state,B,kb,micro),p)[0]
        d=norm_moments(discrepancy(extra,kc),discrepancy(extra2,kd))
        out['components'].append(np.asarray(j.concatenate((j.stack((v,err)),d))).tolist())
        out['times'].append(times)
        for r,(m,k,n,b) in zip(rows,cases):r['pair'].append(float(pair(state,L[b],R[b],ka,dummy,m,k,n)))
        if i%4==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'native_error2':np.mean(out['components'],axis=0)[1]});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

"""Keep the history predictor inside the native MuonW update, with exact native audits."""
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
from .causal_feedback import basis,packet
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .nested_forecast import large_control,difference,add_control,norm_moments
from .subspace_forecast import native_audit,components
from .run import ROOT,ready


def prediction(state,history,coef):
    prev,older,start=history
    B=jax.tree.leaves(basis(packet(state),prev,older,start,0))
    target=eqx.filter(state.model,eqx.is_inexact_array);leaves,td=jax.tree.flatten(target)
    vals=[(c@z).reshape(v.shape) if z.size else j.zeros_like(v) for v,z,c in zip(leaves,B,coef)]
    return jax.tree.unflatten(td,vals),B


@eqx.filter_jit
def getpred(state,history,coef):
    return prediction(state,history,coef)[0]


@eqx.filter_jit
def advance(state,history,coef,pastG,pastC,observed,p,key,adapt):
    pred,features=prediction(state,history,coef)
    grad=pred if observed is None else jax.tree.map(lambda x,y:x+(y-x)/p,pred,observed)
    nxt=state.take_step(grad,key=key)[0]
    if not adapt:return nxt
    # Charge the update of the online fit even on cheap rounds. The next round can consume these coefficients.
    grams=j.stack([z@z.T for z in features])
    crosses=j.stack([z@g.reshape(-1) if z.size else j.zeros(6) for z,g in zip(features,jax.tree.leaves(grad))])
    g=grams+pastG;c=crosses+pastC
    s=j.sqrt(j.maximum(1e-16,j.diagonal(g,axis1=1,axis2=2)))
    r=g/s[:,:,None]/s[:,None,:]+.12*j.eye(6)
    newcoef=j.linalg.solve(r,(c/s)[:,:,None])[:,:,0]/s
    return nxt,newcoef,grams,crosses


def main(step,K,micro,adaptive):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    history=[packet(recover(t,fs)[0]) for t in (max(4001,step-1),max(4001,step-2),4001)]
    if adaptive:
        learned=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
        ix=step-4001;coef=j.asarray(learned[ix]['A'],j.float32)
        prior=learned[max(0,ix-2):ix]
    else:
        leaves=len(jax.tree.leaves(eqx.filter(state.model,eqx.is_inexact_array)))
        coef=j.zeros((leaves,6),j.float32).at[:,0].set(-10.);prior=[]
    s=len(coef);gs=j.asarray(sum((np.array(r['gram']) for r in prior),np.zeros((s,6,6))),j.float32)
    cs=j.asarray(sum((np.array(r['feedback_cross']) for r in prior),np.zeros((s,6))),j.float32)
    tmp=Path('/tmp/cheap-temporal.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[k][512:] for k in ('tokens','weights','segments'))
    orig=Attention.__call__;fused_attention.install()
    def native(d):
        Attention.__call__=orig
        try:return native_audit(state.model,d,micro)
        finally:fused_attention.install()
    def rem(d,key):
        _,avg=debias.audit(state,d,key,12288,64,61,j.arange(1))
        return difference(native(d),avg)
    seed=80231+step+K+adaptive;rng=np.random.default_rng(seed)
    def draw(b):
        ix=rng.integers(len(pool[0]),size=b);return tuple(j.asarray(v[ix]) for v in pool)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,K=K,micro=micro,cal=adaptive,checkpoint=src,seed=seed,device=jax.devices()[0].device_kind)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='low-overhead-causal-prediction',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,k,n,b) for m,k,n in ((61,4096,16),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in (1,2,3,4,6,8,12,16,24,32,48,64)]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    out=dict(**cfg,wandb=run.url,rows=rows,times=[],components=[])
    dst=ROOT+f'/cheap-temporal/s{step}-K{K}-m{micro}-a{adaptive}.json';dummy=j.arange(1)
    pred=getpred(state,history,coef)
    for i in range(33):
        A,B,D,E,F,U,V=ready(tuple(draw(b) for b in (micro,micro,K,micro,micro,64,64)))
        L,R=ready(tuple({b:tuple(x[:b] for x in d) for _,_,_,b in cases} for d in (U,V)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(seed+i),4);times=[0.]*3
        def execute(c):
            if c>=0:
                m,k,n,b=cases[c];return update(state,L[b],ka,dummy,m,k,n)
            g=None if c==-1 else large_control(state,D,kb,micro)
            if c==-3:g=add_control(g,rem(E,kc),.125)
            return advance(state,history,coef,gs,cs,g,.25,ka,adaptive)
        for c in rng.permutation(np.arange(-3,len(cases))):
            t=time.perf_counter();ready(execute(c));ms=1000*(time.perf_counter()-t)
            if i:
                if c<0:times[-c-1]=ms
                else:rows[c]['ms'].append(ms)
        if not i:continue
        v=components(large_control(state,A,ka,micro),large_control(state,B,kb,micro),pred)[0]
        err=components(native(A),native(B),pred)[1];extra=norm_moments(rem(E,kc),rem(F,kd))
        out['times'].append(times);out['components'].append(np.asarray(j.concatenate((j.stack((v,err)),extra))).tolist())
        for row,(m,k,n,b) in zip(rows,cases):row['pair'].append(float(pair(state,L[b],R[b],ka,dummy,m,k,n)))
        if i%4==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'native_error2':np.mean(out['components'],axis=0)[1]});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

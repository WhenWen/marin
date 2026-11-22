"""Fit a per-tensor momentum coefficient from a cheap sketch on queried steps; run stochastic gates on device."""
import functools
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
from .diagnostic_artifacts import publish
from .independent_audits import copy_state,donated_baseline,regular,residual
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready
from .subspace_forecast import components,native_audit
from .temporal import momentum

ORIGINAL=Attention.__call__


def prior(state,history):
    tree=momentum(state);leaves=jax.tree.leaves(tree);G=np.zeros(len(leaves));C=np.zeros_like(G)
    for z in history:
        G=.75*G+np.asarray(z['gram'])[:,0,0];C=.75*C+np.asarray(z['feedback_cross'])[:,0]
    return j.asarray(G,j.float32),j.asarray(C,j.float32)


def coeff(G,C):
    # A relative ridge pulls each scalar toward the checkpoint's usable momentum scale.
    return j.clip((C-.08*8*G-8.e-16)/(1.08*G+1.e-16),-50.,50.)


def predict(mom,aux,learn):
    leaves,td=jax.tree.flatten(mom);beta=coeff(*aux)
    return jax.tree.unflatten(td,[beta[i]*x if learn and x.size<2_000_000 else j.zeros_like(x) for i,x in enumerate(leaves)])


def learn_stats(mom,raw,key,aux):
    G,C=aux;g=[];c=[]
    salt=jax.random.bits(key,(),dtype=j.uint32)
    for i,(m,z) in enumerate(zip(jax.tree.leaves(mom),jax.tree.leaves(raw))):
        if m.size>=2_000_000:g.append(j.array(0.));c.append(j.array(0.));continue
        n=min(m.size,256)
        # Each coordinate is uniform after the common random cyclic offset.
        ix=((salt%m.size+j.arange(n,dtype=j.uint32)*(m.size//n))%m.size).astype(j.int32)
        xs=m.ravel()[ix];ys=z.ravel()[ix];factor=m.size/n
        g.append(factor*j.vdot(xs,xs));c.append(factor*j.vdot(xs,ys))
    return .75*G+j.stack(g),.75*C+j.stack(c)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,carry):
    data,key,forced,p,learn=inputs;state,aux=carry
    c1,c2,sk,gk,dk,statkey,nextkey=jax.random.split(key,7)
    sampled=jax.random.bernoulli(c1,p).astype(j.int32)+2*jax.random.bernoulli(c2,.002).astype(j.int32)
    event=j.where(forced<0,sampled,forced)
    m=momentum(state);pred=predict(m,aux,learn)
    def branch(tag):
        def f(_):
            ids=jax.random.permutation(sk,128) if tag else None
            grad=pred;nxt=aux
            if tag&1:
                data_h=tuple(a[ids[:32]] for a in data)
                raw=debias.draw(state,data_h,gk,12288,64,61,0,j.arange(1))
                grad=jax.tree.map(lambda h,y:h+(y-h)/p,pred,raw)
                if learn:nxt=learn_stats(m,raw,statkey,aux)
            if tag&2:
                ext=tuple(a[ids[32:64]] for a in data)
                Attention.__call__=ORIGINAL
                try:gold=native_audit(state.model,ext,32)
                finally:fused_attention.install()
                d=residual(state,ext,dk,12288,64,gold)
                grad=jax.tree.map(lambda h,y:h+y/.002,grad,d)
            return eqx.filter((grad,nxt),eqx.is_array)
        return f
    grad,nxt=jax.lax.switch(event,[branch(i) for i in range(4)],None)
    return state.take_step(grad,key=nextkey)[0],nxt,event


def main(step):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);fused_attention.install()
    history=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'));aux=ready(prior(state,history[:step-4001]))
    f=Path('/tmp/sketch-gate.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:pool=tuple(z[name][512:] for name in ('tokens','weights','segments'))
    seed=98251+step;rng=np.random.default_rng(seed);dummy=j.arange(1)
    def sample(size):
        ix=rng.integers(len(pool[0]),size=size);return tuple(j.asarray(a[ix]) for a in pool)
    def delta(d,k):
        Attention.__call__=ORIGINAL
        try:gold=native_audit(state.model,d,32)
        finally:fused_attention.install()
        return residual(state,d,k,12288,64,gold)
    cases=[(m,k,n,b) for m,k,n in ((61,64,4),(61,256,8),(61,4096,32),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64),(3,32768,0),(4,4096,0),(206,1024,0)) for b in (1,2,4,8,12,16,24,32,48)]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    cfg=dict(step=step,seed=seed,checkpoint=src,device=jax.devices()[0].device_kind,rep=1)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='device-gate-sketched-momentum-regression',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,times=[],components=[],live=[])
    dest=ROOT+f'/sketch-gate/{cfg["device"].split()[1]}-s{step}.json'
    m=momentum(state);pred=ready(predict(m,aux,True));zero=ready(predict(m,aux,False));sizes={c[-1] for c in cases};p=j.asarray(.2)
    tags=tuple(j.asarray(i,j.int32) for i in (0,1,2,3,-1))
    for i in range(49):
        a,b,c,d,data,second=ready(tuple(sample(s) for s in (32,32,32,32,128,128)))
        left,right=ready(tuple({sz:tuple(x[:sz] for x in ds) for sz in sizes} for ds in (data,second)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(i+seed),4);times=np.zeros((2,4));live=np.zeros((2,2))
        for ix in rng.permutation(np.arange(-10,len(cases))):
            if ix<0:
                learn,tag=divmod(-int(ix)-1,5);carry=copy_state((state,aux))
                tic=time.perf_counter();_,_,event=ready(advance((data,ka,tags[tag],p,bool(learn)),carry))
                ms=1000*(time.perf_counter()-tic)
                if tag<4:times[learn,tag]=ms
                else:live[learn]=[int(event),ms]
            else:
                work=copy_state(state);m0,k,n,sz=cases[ix];tic=time.perf_counter()
                ready(donated_baseline((left[sz],ka,dummy,m0,k,n),work));ms=1000*(time.perf_counter()-tic)
                if i:rows[ix]['ms'].append(ms)
        if not i:continue
        g1=regular(state,a,ka,12288,64);g2=regular(state,b,kb,12288,64)
        vz=components(g1,g2,zero);vl=components(g1,g2,pred)
        er=norm_moments(delta(c,kc),delta(d,kd))
        out['times'].append(times.tolist());out['live'].append(live.tolist())
        out['components'].append(np.asarray(j.concatenate((vz,vl[1:],er))).tolist())
        for row,(m0,k,n,sz) in zip(rows,cases):row['pair'].append(float(pair(state,left[sz],right[sz],ka,dummy,m0,k,n)))
        if i%6==0:publish(fs,dest,out)
    run.log({'cycles':48});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(*map(int,sys.argv[1:]))

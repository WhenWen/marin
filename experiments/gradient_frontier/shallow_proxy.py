"""Fit affine residual shortcuts in the checkpoint transformer and use them as document controls."""
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
from levanter.layers.attention_mask import AttentionMask
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .head_to_head import update
from .contest import pair
from .fused_attention import install
from .run import ROOT,ready


def inputs(m,data):
    tok,_,sg=data;b,t=tok.shape
    axes=(h.Axis('batch',b),h.Axis('position',t))
    return m.embeddings.embed(h.named(tok,axes)),AttentionMask.causal().with_segment_ids(h.named(sg,axes))


def arr(x):
    return x.rearrange(('batch','position','embed')).array


@eqx.filter_jit
def fit(model,data,stride):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    x,mask=inputs(m,data);layers=m.transformer.layers.unstacked();bank=[]
    for start in range(0,len(layers),stride):
        h1=layers[start](x,mask);end=h1
        for layer in layers[start+1:start+stride]:end=layer(end,mask)
        e=arr(h1).reshape(-1,arr(x).shape[-1]).astype(j.float32)
        residual=arr(end).reshape(e.shape).astype(j.float32)-e
        mu=e.mean(0);bias=residual.mean(0);f=e-mu;sd=j.sqrt(j.maximum(.001,(f*f).mean(0)))
        f=f/sd
        coef=jax.scipy.linalg.solve(f.T@f+.15*len(f)*j.eye(f.shape[1]),f.T@(residual-bias),assume_a='pos')/sd[:,None]
        bank.append((coef.astype(j.bfloat16),(bias-mu@coef).astype(j.bfloat16)))
        x=end
    return tuple(bank)


def cheap(model,data,bank,stride,k,copy):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model);x,mask=inputs(m,data)
    layers=m.transformer.layers.unstacked();stop=jax.lax.stop_gradient
    for jx,start in enumerate(range(0,len(layers),stride)):
        layer=layers[start]
        if copy:
            # All later blocks receive a surrogate gradient without evaluating their transformer.
            for other in layers[start+1:start+stride]:
                layer=jax.tree.map(lambda a,b:a+.5*(b-stop(b)),layer,other)
        x=layer(x,mask);a=arr(x);coef,bias=bank[jx]
        x=h.named(a+a@coef+bias,x.rearrange(('batch','position','embed')).axes)
    x=arr(m.transformer.norm(x));w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(data[0],-1,axis=1);pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    logits=(x@w[:k].T).astype(j.float32);head=jax.nn.logsumexp(logits,-1)
    tail=jax.nn.logsumexp(logits[:,:,k//2:],-1)+j.log(1.5)
    z=j.logaddexp(j.logaddexp(head,tail),j.where(y>=k,pos,-j.inf))
    return ((z-pos)*data[1]).sum()/data[1].sum()


def target(model,d,key):
    return objective(model,d,j.arange(1),key,(61*100000+12288,64))


@eqx.filter_jit
def mixed(state,data,bank,stride,k,copy,b,gamma,key):
    small=jax.tree.map(lambda a:a[:b],data)
    def loss(m):return target(m,small,key)+gamma*(cheap(m,data,bank,stride,k,copy)-cheap(m,small,bank,stride,k,copy))
    grad=eqx.filter_grad(loss)(state.model)
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def measure(state,a,b,key,bank,stride,k,copy):
    ka,kb=jax.random.split(key)
    ga,gb=[eqx.filter_grad(target)(state.model,d,ke) for d,ke in ((a,ka),(b,kb))]
    ha,hb=[eqx.filter_grad(cheap)(state.model,d,bank,stride,k,copy) for d in (a,b)]
    def both(x,y,u,v):
        dg=x-y;dh=u-v
        return j.stack((j.vdot(dg,dg),j.vdot(dh,dh),j.vdot(dg,dh)))*.5
    return j.stack([both(*vs) for vs in zip(*map(jax.tree.leaves,(ga,gb,ha,hb)))])


def main(step,stride,k,copy,b,M):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    initial=state if step==4001 else recover(4001,fs)[0]
    file=Path('/tmp/shallow-pool.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(file))
    with np.load(file) as z:
        train=tuple(j.asarray(z[a][:4]) for a in ('tokens','weights','segments'))
        hold=tuple(z[a][512:] for a in ('tokens','weights','segments'))
    bank=ready(fit(initial.model,train,stride));prep=[]
    for _ in range(6):
        t=time.perf_counter();ready(fit(initial.model,train,stride));prep.append(1000*(time.perf_counter()-t))
    seed=9292+step;gen=np.random.default_rng(seed)
    def get(n):
        ix=gen.integers(len(hold[0]),size=n);return tuple(j.asarray(v[ix]) for v in hold)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,stride=stride,k=k,copy=copy,b=b,M=M,checkpoint=src,seed=seed)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='shallow-gradient-prediction',id=os.urandom(5).hex(),config=cfg)
    cases=[(m,B) for m in (61,62,7) for B in (4,8,16,24,32,48)]
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in cases]
    out=dict(**cfg,prep=prep,wandb=run.url,components=[],ms=[],rows=rows,device=jax.devices()[0].device_kind)
    dst=ROOT+f'/shallow-control/s{step}-d{stride}-k{k}-c{copy}-b{b}-M{M}.json'
    for i in range(49):
        left,right=ready((get(max(M,48)),get(max(M,48))));key=jax.random.key(seed+i)
        a,c=ready(tuple({B:tuple(x[:B] for x in d) for B in {4,M,*(B for _,B in cases)}} for d in (left,right)))
        for ix in gen.permutation(np.arange(-1,len(cases))):
            t=time.perf_counter()
            if ix==-1:ready(mixed(state,a[M],bank,stride,k,copy,b,1.,key))
            else:
                m,B=cases[ix];size=12288 if m==61 else 16384;ready(update(state,a[B],key,j.arange(1),m,size,64))
            elapsed=(time.perf_counter()-t)*1000
            if i:
                if ix==-1:out['ms'].append(elapsed)
                else:rows[ix]['ms'].append(elapsed)
        if not i:continue
        out['components'].append(np.asarray(measure(state,a[4],c[4],key,bank,stride,k,copy)).tolist())
        for row,(m,B) in zip(rows,cases):
            row['pair'].append(float(pair(state,a[B],c[B],key,j.arange(1),m,12288 if m==61 else 16384,64)))
        if i%8==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'time_ms':np.median(out['ms'])});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

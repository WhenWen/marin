"""Use original-context debiased gradients with cropped-document prediction controls."""
import json
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .fused_attention import install
from .lowmem import objective as strong,stats
from .tail import trunk
from .run import ROOT,ready


def windows(model,data,starts,context,targets):
    tok,weight,segments=data
    index=starts[:,:,None]+j.arange(targets-context,targets)
    flat=index.reshape(len(tok),-1).clip(0,tok.shape[1]-1)
    words=j.take_along_axis(tok,flat,axis=1).reshape(-1,context)
    seg=j.take_along_axis(segments,flat,axis=1).reshape(-1,context)
    seg=j.where(index.reshape(-1,context)<0,-1,seg)
    h=trunk(model,(words,j.ones_like(words),seg))
    return h[:,-targets:,:].reshape(len(tok),-1,h.shape[-1])


def loss(model,data,key,b,k,mixing,groups,context,targets):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    tokens,weights,_=data;M,t=tokens.shape
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    full=jax.tree.map(lambda a:a[:b],data)
    h=trunk(m,full)
    y=j.roll(tokens,-1,axis=1)
    uk,headkey=jax.random.split(key)
    scale=t//groups
    starts=j.arange(groups)[None,:]*scale+targets*jax.random.randint(uk,(M,groups),0,scale//targets)
    index=(starts[:,:,None]+j.arange(targets)).reshape(M,-1)
    yy=j.take_along_axis(y,index,axis=1)
    ww=j.take_along_axis(weights,index,axis=1)
    # Each full document has 1023 active targets. All positions, including the masked last one,
    # have uniform inclusion probability. Keep the original denominator after cropping.
    short=windows(m,data,starts,context,targets)
    logits=short@w[:k].T
    pos=j.einsum('btd,btd->bt',short,w[yy]).astype(j.float32)
    z,_,_=stats(logits,pos,yy,.5)
    per_doc=((z-pos)*ww).sum(1)*(t/index.shape[1])/weights.sum(1)
    reference=strong(h,w,y[:b],weights[:b],headkey,(61*100000+12288,64))
    return reference+mixing*(per_doc.mean()-per_doc[:b].mean())


@eqx.filter_jit
def evaluate(state,a,z,key,b,k,gamma,groups,context,targets):
    u,v=jax.random.split(key)
    g=eqx.filter_grad(loss)(state.model,a,u,b,k,gamma,groups,context,targets);h=eqx.filter_grad(loss)(state.model,z,v,b,k,gamma,groups,context,targets)
    return sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(g),jax.tree.leaves(h)))*.5


@eqx.filter_jit
def train_step(state,a,key,b,k,gamma,groups,context,targets):
    g=eqx.filter_grad(loss)(state.model,a,key,b,k,gamma,groups,context,targets)
    return state.take_step(g,key=key)[0]


def main(step,b,M,k,groups,context,targets):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    f=Path('/tmp/fused-doc-ppi.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:hold=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    assert np.all(hold[1].sum(axis=1)==1023) and context>=targets and 1024%(groups*targets)==0
    seed=79178+step+b+k+groups+context;rng=np.random.default_rng(seed)
    sizes=sorted({max(2,b//2),b,2*b,3*b,4*b,6*b,8*b})
    def draw():
        i=rng.integers(len(hold[0]),size=max(M,max(sizes)))
        return {B:tuple(j.asarray(v[i[:B]]) for v in hold) for B in set(sizes+[M])}
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,b=b,M=M,k=k,groups=groups,context=context,targets=targets,seed=seed,checkpoint=src)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='cropped-context-document-PPI',id=os.urandom(5).hex(),config=cfg)
    base=[(m,B) for m in (61,62,7) for B in sizes];gammas=(.25,.5,.75,1.)
    rows=[dict(method=m,b=B,ms=[],pair=[]) for m,B in base]
    cand=[dict(gamma=g,ms=[],pair=[]) for g in gammas]
    out=dict(**cfg,device=jax.devices()[0].device_kind,wandb=run.url,rows=rows,candidates=cand)
    dst=ROOT+f'/cropped-ppi/s{step}-b{b}-M{M}-k{k}-g{groups}-L{context}-q{targets}.json';dummy=j.arange(1)
    for i in range(37):
        a,z=ready((draw(),draw()));key=jax.random.key(seed+i)
        for ix in rng.permutation(np.arange(-len(cand),len(rows))):
            start=time.perf_counter()
            if ix<0:ready(train_step(state,a[M],key,b,k,gammas[-ix-1],groups,context,targets))
            else:
                m,B=base[ix];ready(update(state,a[B],key,dummy,m,12288 if m==61 else 16384,64))
            t=(time.perf_counter()-start)*1000
            if i:(cand[-ix-1] if ix<0 else rows[ix])['ms'].append(t)
        if not i:continue
        for r in cand:r['pair'].append(float(evaluate(state,a[M],z[M],key,b,k,r['gamma'],groups,context,targets)))
        for r,(m,B) in zip(rows,base):r['pair'].append(float(pair(state,a[B],z[B],key,dummy,m,12288 if m==61 else 16384,64)))
        if i%6==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':36});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

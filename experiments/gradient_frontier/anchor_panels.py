"""Learn representative contexts and backpropagate once per context for a signed difference of raw-text panels."""
import functools
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
from . import fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .head_to_head import gradient
from .independent_audits import copy_state,donated_baseline
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit
from .tail import trunk


def codes(tokens,segments):
    prev=j.roll(tokens,1,axis=-1)
    valid=(segments==j.roll(segments,1,axis=-1))&(j.arange(tokens.shape[-1])>0)
    return ((j.where(valid,prev,0)*1103515245)^(tokens*12345))%65536


def assign(tokens,seg,bank):
    bigram,unigram,fallback=bank
    bi=bigram[codes(tokens,seg)];uni=unigram[tokens]
    return j.where(bi>=0,bi,j.where(uni>=0,uni,fallback[tokens]))


@eqx.filter_jit
def fit(model,data,R,L):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    tok,_,seg=data
    def dictionary(values,size,start,count):
        hits=j.bincount(values.ravel(),length=size);ids=jax.lax.top_k(hits,count)[1]
        return j.full(size,-1,j.int32).at[ids].set(start+j.arange(count))
    bi=dictionary(codes(tok,seg),65536,0,R//2);uni=dictionary(tok,len(w),R//2,R//4)
    # Less frequent inputs share cells by the angular signature of their learned input embedding.
    emb=m.embeddings.token_embeddings.weight.rearrange(('vocab','embed')).array
    bits=(R//4).bit_length()-1
    proj=jax.random.rademacher(jax.random.key(717), (emb.shape[1],bits),dtype=j.float32).astype(j.bfloat16)
    fallback=3*R//4+j.sum(((emb@proj)>0).astype(j.int32)*(2**j.arange(bits)),axis=1)
    bank=(bi,uni,fallback);group=assign(tok,seg,bank)
    permutation=jax.random.permutation(jax.random.key(871),tok.size)
    first=jax.ops.segment_min(j.arange(tok.size),group.ravel()[permutation],R)
    index=permutation[j.minimum(first,tok.size-1)]
    pos=j.maximum(0,index[:,None]%tok.shape[1]-j.arange(L-1,-1,-1));doc=index[:,None]//tok.shape[1]
    tokens=tok[doc,pos];segments=seg[doc,pos]+j.arange(R)[:,None]*tok.shape[1]
    anchors=(tokens.reshape(-1,128),j.ones_like(tokens,j.float32).reshape(-1,128),segments.reshape(-1,128))
    x=trunk(m,anchors).reshape(R,L,-1)[:,-1]
    p=jax.nn.softmax((x@w.T).astype(j.float32),axis=-1);mu=p.astype(j.bfloat16)@w
    top,ids=jax.lax.top_k(p,16)
    return bank,anchors,mu,top.astype(j.bfloat16),ids


def contrast(model,data,bank,L):
    tables,anchors,mu,top,ids=bank;R,d=mu.shape
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,anchors).reshape(R,L,d)[:,-1]
    tokens,weights,seg=data;mid=len(tokens)//2
    weights=j.concatenate((-.5*weights[:mid]/weights[:mid].sum(),.5*weights[mid:]/weights[mid:].sum())).ravel()
    groups=assign(tokens,seg,tables).ravel();y=j.roll(tokens,-1,axis=-1).ravel()
    mass=jax.ops.segment_sum(weights,groups,R)
    cot=mass[:,None]*mu-jax.ops.segment_sum(weights[:,None]*w[y],groups,R)
    stop=jax.lax.stop_gradient
    hidden=j.sum(x.astype(j.float32)*stop(cot).astype(j.float32))
    read=j.sum((mass[:,None]*top)*j.einsum('rd,rkd->rk',stop(x),w[ids]).astype(j.float32))
    read-=j.sum(weights*j.sum(stop(x)[groups]*w[y],axis=1,dtype=j.float32))
    return hidden+read


@eqx.filter_jit
def control(model,panel,key,bank,L):
    M,b,t=panel[0].shape;depths=M.bit_length()-1
    coins=jax.random.bernoulli(key,.5,(depths,)).astype(j.int32)
    offset=j.array(0,j.int32);total=None
    for level in range(depths):
        span=M//(2**level)
        block=jax.tree.map(lambda arr:jax.lax.dynamic_slice_in_dim(arr,offset,span).reshape(span*b,t),panel)
        delta=eqx.filter_grad(contrast)(model,block,bank,L)
        sign=1.-2.*coins[level]
        total=jax.tree.map(lambda v:sign*v,delta) if total is None else jax.tree.map(lambda a,v:a+sign*v,total,delta)
        offset+=coins[level]*(span//2)
    leaf=jax.tree.map(lambda arr:jax.lax.dynamic_index_in_dim(arr,offset,keepdims=False),panel)
    return total,leaf


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(args,state):
    raw,d,key=args
    return state.take_step(jax.tree.map(j.add,raw,d),key=key)[0]


@eqx.filter_jit
def stats(h,g,d,e):
    raw=jax.tree.map(j.subtract,h,g);cov=jax.tree.map(j.subtract,d,e)
    inner=lambda a,b:sum(j.vdot(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))/2
    return j.stack((inner(raw,raw),inner(raw,cov),inner(cov,cov)))


def main(step,b,M,R,L):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);orig=Attention.__call__;fused_attention.install()
    f=Path('/tmp/anchor-panel.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:
        pool=tuple(z[key][512:] for key in ('tokens','weights','segments'))
        cal=tuple(j.asarray(z[key][:128]) for key in ('tokens','weights','segments'))
    initial=state if step==4001 else recover(4001,fs)[0]
    bank=ready(fit(initial.model,cal,R,L));prep=[]
    for _ in range(8):
        tic=time.perf_counter();bank=ready(fit(initial.model,cal,R,L));prep.append(1000*(time.perf_counter()-tic))
    seed=137551+step+b+R+L;gen=np.random.default_rng(seed);dummy=j.arange(1)
    def draw(shape):
        ix=gen.integers(len(pool[0]),size=shape);return tuple(j.asarray(a[ix]) for a in pool)
    def gold(leaf):
        Attention.__call__=orig
        try:return native_audit(state.model,leaf,b)
        finally:fused_attention.install()
    sizes=sorted({max(1,b//4),max(1,b//2),b,b*2,b*3,b*4})
    cases=[(m,k,n,s) for m,k,n in ((3,32768,0),(4,4096,0),(201,4096,0),(202,8192,0),(206,1024,0),(61,4096,32),(61,8192,64),(61,12288,64),(62,16384,64),(7,16384,64)) for s in sizes]
    rows=[dict(method=m,k=k,n=n,b=s,ms=[],pair=[]) for m,k,n,s in cases]
    cfg=dict(step=step,b=b,M=M,R=R,L=L,seed=seed,checkpoint=src,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='learned-anchor-document-panels',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,prep=prep,rows=rows,times=[],components=[])
    dst=ROOT+f'/anchor-panels/s{step}-b{b}-M{M}-R{R}-L{L}.json'
    for it in range(37):
        a,z,u,v=ready((draw((M,b)),draw((M,b)),draw(max(sizes)),draw(max(sizes))))
        left,right=ready(tuple({s:tuple(arr[:s] for arr in ds) for s in sizes} for ds in (u,v)))
        c1,c2,k1,k2=jax.random.split(jax.random.key(seed+it),4);ts=[0.,0.]
        for ix in gen.permutation(np.arange(-2,len(cases))):
            work=copy_state(state);tic=time.perf_counter()
            if ix<0:
                d,leaf=control(state.model,a,c1,bank,L)
                raw=gold(leaf) if ix==-1 else gradient(state,leaf,k1,dummy,61,12288,64)
                ready(advance((raw,d,k1),work))
            else:
                m,k,n,s=cases[ix];ready(donated_baseline((left[s],k1,dummy,m,k,n),work))
            ms=1000*(time.perf_counter()-tic)
            if it:
                if ix<0:ts[-ix-1]=ms
                else:rows[ix]['ms'].append(ms)
        if not it:continue
        da,la=control(state.model,a,c1,bank,L);dz,lz=control(state.model,z,c2,bank,L)
        native=stats(gold(la),gold(lz),da,dz)
        cheap=stats(gradient(state,la,k1,dummy,61,12288,64),gradient(state,lz,k2,dummy,61,12288,64),da,dz)
        out['components'].append(np.asarray(j.stack((native,cheap))).tolist());out['times'].append(ts)
        for row,(m,k,n,s) in zip(rows,cases):row['pair'].append(float(pair(state,left[s],right[s],k1,dummy,m,k,n)))
        if it%6==0:publish(fs,dst,out)
    run.log({'cycles':36});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(*map(int,sys.argv[1:]))

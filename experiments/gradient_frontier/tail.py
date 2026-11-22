"""Fit missing softmax probability mass and its hidden-state score on disjoint calibration tokens."""
import equinox as eqx
import haliax as h
import jax
import jax.numpy as j
from levanter.layers.attention_mask import AttentionMask
from .multifidelity import shortlist

def trunk(m,data):
    tokens,_,seg=data; b,t=tokens.shape
    axes=(h.Axis('batch',b),h.Axis('position',t))
    x=m.embeddings.embed(h.named(tokens,axes))
    mask=AttentionMask.causal().with_segment_ids(h.named(seg,axes))
    for layer in m.transformer.layers.unstacked():x=layer(x,mask)
    return m.transformer.norm(x).rearrange(('batch','position','embed')).array

def features(x,z0,positive,top,inside):
    a=x.astype(j.float32)
    return j.concatenate((a,a*a,z0[:,:,None],positive[:,:,None],(top-z0)[:,:,None],
                          j.exp(j.minimum(0.,positive-z0))[:,:,None],inside[:,:,None]),axis=-1)

@eqx.filter_jit
def fit(model,data,cal_tokens,k,ncal):
    ids=shortlist(cal_tokens,k,128256)
    member=j.zeros((128256,),bool).at[ids].set(True)
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    def moments(d):
        x=trunk(m,d);y=j.roll(d[0],-1,axis=1)
        scores=(x@w.T).astype(j.float32)
        pos=j.take_along_axis(scores,y[:,:,None],-1)[:,:,0]
        head=scores[:,:,ids];top=j.max(head,-1);inside=member[y]
        z0=j.logaddexp(jax.nn.logsumexp(head,-1),j.where(inside,-1e9,pos))
        ff=features(x,z0,pos,top,inside)
        raw=j.where(member,-1e9,scores)
        b,t=y.shape
        raw=raw.at[j.arange(b)[:,None],j.arange(t)[None,:],y].set(-1e9)
        target_z=jax.nn.logsumexp(raw,-1)
        target_w=(jax.nn.softmax(raw,-1).astype(j.bfloat16)@w).astype(j.float32)
        target=j.concatenate((target_z[:,:,None],target_w),-1)
        f=ff.reshape((-1,ff.shape[-1]));u=target.reshape((-1,target.shape[-1]))
        return f.T@f,f.T@u,f.sum(0),u.sum(0)
    blocks=jax.tree.map(lambda a:a[:ncal].reshape((-1,2,a.shape[1])),data)
    s=moments(jax.tree.map(lambda a:a[0],blocks))
    def add(s,i):return jax.tree.map(j.add,s,moments(jax.tree.map(lambda a:a[i],blocks))),None
    (xx,xy,sx,sy),_=jax.lax.scan(add,s,j.arange(1,ncal//2))
    N=ncal*data[0].shape[1];mx=sx/N;my=sy/N
    cov=xx/N-j.outer(mx,mx);sd=j.sqrt(j.maximum(j.diag(cov),1e-9))
    coef=j.linalg.solve(cov/sd[:,None]/sd[None,:]+.1*j.eye(len(sd)),(xy/N-j.outer(mx,my))/sd[:,None])/sd[:,None]
    return ids,coef,my-mx@coef

def estimate(x,head,z0,pos,member,bank):
    _,coef,inter=bank
    pred=jax.lax.stop_gradient(features(x,z0,pos,j.max(head,-1),member)@coef+inter)
    z=pred[:,:,0]+j.sum((x.astype(j.float32)-jax.lax.stop_gradient(x.astype(j.float32)))*pred[:,:,1:],axis=-1)
    total=j.logaddexp(z0,z)
    return total,j.exp(z-total)

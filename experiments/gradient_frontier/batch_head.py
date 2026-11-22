"""Correct an adaptive vocabulary formed from the actual minibatch, conditioning on all its tokens."""
import jax
import jax.numpy as j

def objective(x,w,y,wt,key,config):
    code,n=config;kind=code//100000;k=code%100000;vocab=w.shape[0]
    extra={30:1024,31:2048,32:4096,33:8192,34:2048,35:4096,36:4096,37:4096}[kind]
    sorted_ids=j.sort(y.reshape(-1))
    unique=j.concatenate((j.ones(1,bool),sorted_ids[:-1]!=sorted_ids[1:]))&(sorted_ids>=k)
    ix=j.nonzero(unique,size=extra,fill_value=len(sorted_ids))[0]
    more=j.concatenate((sorted_ids,j.asarray([vocab],j.int32)))[ix]
    table=j.concatenate((w[:k],w[j.minimum(more,vocab-1)]),axis=0)
    scores=(x@table.T).astype(j.float32)
    common=scores[:,:,:k];ext=j.where(more<vocab,scores[:,:,k:],-1e9)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    member=(y<k)|j.zeros((vocab+1,),bool).at[more].set(True)[y]
    zp=j.where(member,-1e9,positive)
    l0=jax.nn.logsumexp(common,-1);l1=jax.nn.logsumexp(ext,-1)
    if kind in (34,35):
        # Smoothly inflate the observed tail except for an observed true target.
        inside=(y>=k)&member
        dropped=j.where(inside,j.minimum(1.,j.exp(positive-l1)),0.)
        z=j.logaddexp(j.logaddexp(l0,l1),j.logaddexp(zp,l1+j.log(j.maximum(1.-dropped,1e-5))))
    else:z=j.logaddexp(j.logaddexp(l0,l1),zp)
    top=j.maximum(j.maximum(j.max(common,-1),j.max(ext,-1)),positive)
    diff=1.-j.minimum(1.,j.exp(top-z))
    score=.035+j.sqrt(j.minimum(1.,j.exp(l1-z)))+.15*diff
    if kind==36:score=j.ones_like(score)
    b,t=y.shape
    if kind==37:
        p=jax.lax.stop_gradient(score/score.sum(-1,keepdims=True))
        uu=(j.arange(n)[None,:]+jax.random.uniform(key,(b,n)))/n
        pos=jax.vmap(lambda a,u:j.searchsorted(a,u,side='right',method='scan_unrolled'))(j.cumsum(p,-1),uu)
        pos=j.minimum(t-1,pos);iw=1./(n*j.take_along_axis(p,pos,1))
    else:
        width=t//n;s=jax.lax.stop_gradient(score.reshape((b,n,width)));p=s/s.sum(-1,keepdims=True)
        u=jax.random.uniform(key,(b,n,1));local=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
        pos=j.arange(n)[None,:]*width+local;iw=1./j.take_along_axis(p,local[:,:,None],-1)[:,:,0]
    chosen=j.take_along_axis(x,pos[:,:,None],1)
    true_z=jax.nn.logsumexp((chosen@w.T).astype(j.float32),-1)
    delta=(true_z-j.take_along_axis(z,pos,1))*j.take_along_axis(wt,pos,1)*iw
    return (((z-positive)*wt).sum()+delta.sum())/wt.sum()

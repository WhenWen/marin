"""Shared low-precision head probabilities for the proxy gradient and token sampling scores."""
from functools import partial
import jax
import jax.numpy as j

def parts(head,pos,y,alpha):
    k=head.shape[-1]
    top=jax.lax.stop_gradient(j.maximum(j.max(head,axis=-1).astype(j.float32),pos))
    # The max shift guarantees exponentials no larger than one. The cheap probability buffer is BF16.
    exp=j.exp(head-top.astype(head.dtype)[:,:,None]).astype(j.bfloat16)
    half=k//2
    left=j.sum(exp[:,:,:half],-1,dtype=j.float32);right=j.sum(exp[:,:,half:],-1,dtype=j.float32)
    epos=j.exp(pos-top)
    ext=j.where(y>=k,epos,0.)
    total=left+right+ext+alpha*right
    z=top+j.log(total)
    tail=right/total
    q=(exp.astype(j.float32)/total[:,:,None]).astype(head.dtype)
    return (z,tail,1./total),(q,ext/total)

@partial(jax.custom_vjp,nondiff_argnums=(3,))
def stats(head,pos,y,alpha):
    return parts(head,pos,y,alpha)[0]
def fwd(head,pos,y,alpha):return parts(head,pos,y,alpha)
def bwd(alpha,saved,cts):
    q,r=saved;dz=cts[0]
    # Both auxiliary outputs appear only inside stopped sampling probabilities.
    a=dz.astype(q.dtype)[:,:,None]
    if alpha:
        half=q.shape[-1]//2
        q=j.concatenate((q[:,:,:half],q[:,:,half:]*(1.+alpha)),axis=-1)
    return q*a,r*dz,None
stats.defvjp(fwd,bwd)

def objective(x,w,y,weight,key,config,offsets=None):
    code,n=config;kind=code//100000;k=code%100000;b,t,d=x.shape
    sparse_head=kind in range(70,79)
    keep={70:0,71:256,72:1024,73:2048,74:4096,75:0,76:1024,77:2048,78:256}.get(kind,k)
    if not sparse_head:head=x@w[:k].T
    elif keep==0:head=x@jax.lax.stop_gradient(w[:k]).T
    else:head=j.concatenate((x@w[:keep].T,x@jax.lax.stop_gradient(w[keep:k]).T),axis=-1)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    alpha={60:0.,61:.5,62:1.,63:2.,64:0.,65:0.}.get(kind,0.)
    z,outer,top=stats(head,positive,y,alpha)
    cheap=z-positive
    if sparse_head and kind!=75:
        # Keep the observed label's negative readout contribution, even for a frozen proxy row.
        target_only=j.einsum('btd,btd->bt',jax.lax.stop_gradient(x),w[y]).astype(j.float32)
        qy=jax.lax.stop_gradient(j.where((y>=keep)&(y<k),j.exp(positive-z),0.))
        cheap=cheap+qy*(target_only-jax.lax.stop_gradient(target_only))
        if kind in (76,77,78):
            arg=j.argmax(head,axis=-1)
            best=j.take_along_axis(head,arg[:,:,None],-1)[:,:,0].astype(j.float32)
            qb=jax.lax.stop_gradient(j.where((arg>=keep)&(arg!=y),j.exp(best-z),0.))
            other=j.einsum('btd,btd->bt',jax.lax.stop_gradient(x),w[arg]).astype(j.float32)
            cheap=cheap+qb*(other-jax.lax.stop_gradient(other))
    if kind==64:score=.04+j.sqrt(outer)
    elif kind==65:score=.025+outer+.15*(1.-j.minimum(top,1.))
    else:score=.035+j.sqrt(outer)+.15*(1.-j.minimum(top,1.))
    width=t//n
    p=jax.lax.stop_gradient(score.reshape((b,n,width)));p=p/p.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    if offsets is None:
        inv=1./j.take_along_axis(p,at[:,:,None],-1)[:,:,0]
    else:
        at=offsets
        inv=width
    pos=j.arange(n)[None,:]*width+at
    xx=j.take_along_axis(x,pos[:,:,None],axis=1)
    logits=(xx@w.T).astype(j.float32);target=j.take_along_axis(y,pos,1)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    residual=(exact-j.take_along_axis(cheap,pos,1))*j.take_along_axis(weight,pos,1)*inv
    return ((weight*cheap).sum()+residual.sum())/weight.sum()

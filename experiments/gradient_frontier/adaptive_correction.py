"""Exact token correction of a target-aware log normalizer."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations

def values(m,tokens,ids,n,variant,key):
    h=activations(m,tokens)
    y=tokens[:,1:];w=m.lm_head.weight.array;scale=1./m.config.width_multiplier
    rows,t,_=h.shape;width=t//n
    head=(h@w[ids].T*scale).astype(j.float32)
    z_head=jax.nn.logsumexp(head,-1)
    positive=(j.einsum('btd,btd->bt',h,w[y])*scale).astype(j.float32)
    member=j.zeros((w.shape[0],),bool).at[ids].set(True)
    if variant in (1,3):
        z=j.logaddexp(z_head,j.where(member[y],-1e9,positive))
    elif variant==2:
        doc=j.sort(y,axis=-1)
        unique=j.concatenate((j.ones_like(doc[:,:1],dtype=bool),doc[:,1:]!=doc[:,:-1]),-1)
        keep=unique & ~member[doc]
        local=(j.einsum('btd,bsd->bts',h,w[doc])*scale).astype(j.float32)
        maxval=jax.lax.stop_gradient(j.maximum(z_head,j.max(j.where(keep[:,None,:],local,-j.inf),-1)))
        z=maxval+j.log(j.exp(z_head-maxval)+j.sum(j.exp(j.where(keep[:,None,:],local-maxval[:,:,None],-j.inf)),axis=-1))
    else:z=z_head
    if variant==3:
        # The residual is usually larger where the head assigns diffuse probability.
        ptop=j.exp(j.maximum(j.max(head,-1),positive)-z)
        s=jax.lax.stop_gradient((0.1+1.-j.minimum(1.,ptop)).reshape((rows,n,width)))
        p=s/s.sum(-1,keepdims=True);ix=jax.random.categorical(key,j.log(p),axis=-1)
        weight=(1./width)/j.take_along_axis(p,ix[:,:,None],-1)[:,:,0]
    else:
        ix=jax.random.randint(key,(rows,n),0,width);weight=1.
    pos=ix+j.arange(n)[None,:]*width
    chosen=j.take_along_axis(h,pos[:,:,None],axis=1)
    exact_z=jax.nn.logsumexp((chosen@w.T*scale).astype(j.float32),-1)
    return (z-positive).mean()+((exact_z-j.take_along_axis(z,pos,axis=1))*weight).mean()

def loss(model,tokens,key,ids,n,variant):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    return values(m,tokens,ids,n,variant,key)
@eqx.filter_jit
def sample(model,tokens,key,ids,n,variant):
    return ravel_pytree(eqx.filter_grad(loss)(model,tokens,key,ids,n,variant))[0]

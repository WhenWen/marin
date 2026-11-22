"""Document-adaptive softmax control with a shared head and optional importance-weighted tail."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations, full_losses

@eqx.filter_jit
def shortlist(cal,k,vocab):
    # Fit once from disjoint calibration examples. The realized dictionary is fixed for ten steps.
    head, tail = k % 100000, k // 100000 * 256
    counts=j.bincount(cal[:,1:].reshape(-1),length=vocab).astype(j.float32)
    ids=jax.lax.top_k(counts,head)[1]
    mask=j.zeros((vocab,),bool).at[ids].set(True)
    if tail:
        q=j.where(mask,0.,(counts+0.25)**0.75)
        q=q/q.sum()
        extra=jax.random.choice(jax.random.key(733),vocab,(tail,),p=q,replace=True)
        weights=j.concatenate((j.zeros((head,)), -j.log(tail*q[extra])))
        ids=j.concatenate((ids,extra))
        mask=mask.at[extra].set(True)
    else:
        weights=j.zeros((head,))
    return ids,mask,weights

def proxy(h,w,y,short,scale):
    ids, present,logweights=short
    logits=(j.matmul(h,w[ids].T)*scale).astype(j.float32)+logweights
    head_z=jax.nn.logsumexp(logits,-1)
    # Explicitly normalize over document-specific observed targets including confident rare words.
    doc=j.sort(y,axis=-1)
    unique=j.concatenate((j.ones_like(doc[:,:1],dtype=bool),doc[:,1:]!=doc[:,:-1]),-1)
    new=unique & ~present[doc]
    local=(j.einsum('btd,bsd->bts',h,w[doc])*scale).astype(j.float32)
    max_head=jax.lax.stop_gradient(j.maximum(head_z,j.max(j.where(new[:,None,:],local,-j.inf),axis=-1)))
    z=max_head+j.log(j.exp(head_z-max_head)+j.sum(j.exp(j.where(new[:,None,:],local-max_head[:,:,None],-j.inf)),axis=-1))
    pos=(j.einsum('btd,btd->bt',h,w[y])*scale).astype(j.float32)
    return (z-pos).mean(1)

def objective(model,tokens,short,b,kind,power):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    h=activations(m,tokens)
    w=m.lm_head.weight.array
    y=tokens[:,1:]
    scale=1/m.config.width_multiplier
    if kind=='full':return full_losses(h,w,y,scale).mean()
    cheap=proxy(h,w,y,short,scale)
    if kind=='cheap':return cheap.mean()
    return full_losses(h[:b],w,y[:b],scale).mean()+power*(cheap.mean()-cheap[:b].mean())

def grad_vec(model,tokens,short,b,kind,power=1.):
    return ravel_pytree(eqx.filter_grad(objective)(model,tokens,short,b,kind,power))[0]
estimator=eqx.filter_jit(grad_vec)
per_example=eqx.filter_jit(lambda model,tokens,short,kind:jax.vmap(lambda x:grad_vec(model,x[None],short,1,kind))(tokens))

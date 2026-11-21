"""Unbiased two-level stochastic gradients with a cheap shortlist loss."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .kernels import inputs,AttentionMask

def activations(m,tokens):
    ids=inputs(tokens[:,:-1])
    x=m.embeddings.embed(ids)
    for layer in m.transformer.layers.unstacked():
        x=layer(x,AttentionMask.causal())
    return m.transformer.norm(x).rearrange(('batch','position','embed')).array

def full_losses(hidden,table,targets,scale):
    logits=(j.matmul(hidden,table.T)*scale).astype(j.float32)
    return (jax.nn.logsumexp(logits,axis=-1)-j.take_along_axis(logits,targets[:,:,None],axis=-1)[:,:,0]).mean(1)

def cheap_losses(hidden,table,targets,short,scale):
    logits=(j.matmul(hidden,table[short].T)*scale).astype(j.float32)
    positive=(j.einsum('btd,btd->bt',hidden,table[targets])*scale).astype(j.float32)
    return (jax.nn.logsumexp(logits,axis=-1)-positive).mean(1)

def objective(model,tokens,short,b,kind,power):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    h=activations(m,tokens)
    w=m.lm_head.weight.array
    y=tokens[:,1:]
    scale=1/m.config.width_multiplier
    if kind=='full':return full_losses(h,w,y,scale).mean()
    cheap=cheap_losses(h,w,y,short,scale)
    if kind=='cheap':return cheap.mean()
    full=full_losses(h[:b],w,y[:b],scale)
    return full.mean()+power*(cheap.mean()-cheap[:b].mean())

def grad_vec(model,tokens,short,b,kind,power=1.):
    tree=eqx.filter_grad(objective)(model,tokens,short,b,kind,power)
    return ravel_pytree(tree)[0]
estimator=eqx.filter_jit(grad_vec)
per_example=eqx.filter_jit(lambda model,tokens,short,kind:jax.vmap(lambda x:grad_vec(model,x[None],short,1,kind))(tokens))

@eqx.filter_jit
def shortlist(cal,k,vocab):
    count=j.bincount(cal[:,1:].reshape(-1),length=vocab)
    return jax.lax.top_k(count,k)[1]

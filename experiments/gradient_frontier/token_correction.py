"""Two-phase estimator with exact log-normalizer corrections at sampled token positions."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations,shortlist

def loss(model,tokens,key,ids,n,importance):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    h=activations(m,tokens)
    w=m.lm_head.weight.array
    scale=1/m.config.width_multiplier
    y=tokens[:,1:]
    rows,t,_=h.shape
    width=t//n
    logits=(j.matmul(h,w[ids].T)*scale).astype(j.float32)
    cheap_z=jax.nn.logsumexp(logits,-1)
    positive=(j.einsum('btd,btd->bt',h,w[y])*scale).astype(j.float32)
    if importance:
        # Predict missing mass from head confidence and an observed target absent from the head.
        top=j.max(logits,-1)
        score=0.06+jax.nn.sigmoid(6.-(top-cheap_z+6.)*1.3)+jax.nn.sigmoid(positive-cheap_z)
        score=jax.lax.stop_gradient(score.reshape((rows,n,width)))
        prob=score/score.sum(-1,keepdims=True)
        index=jax.random.categorical(key,j.log(prob),axis=-1)
        inverse=(1./width)/j.take_along_axis(prob,index[:,:,None],axis=-1)[:,:,0]
    else:
        index=jax.random.randint(key,(rows,n),0,width)
        inverse=1.
    ix=j.arange(n)[None,:]*width+index
    chosen=j.take_along_axis(h,ix[:,:,None],axis=1)
    exact_z=jax.nn.logsumexp((j.matmul(chosen,w.T)*scale).astype(j.float32),axis=-1)
    cheap_sel=j.take_along_axis(cheap_z,ix,axis=1)
    return (cheap_z-positive).mean()+((exact_z-cheap_sel)*inverse).mean()

@eqx.filter_jit
def sample(model,tokens,key,ids,n,importance):
    return ravel_pytree(eqx.filter_grad(loss)(model,tokens,key,ids,n,importance))[0]

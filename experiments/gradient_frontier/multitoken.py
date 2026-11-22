"""Three unbiased token-fidelity levels with shared context and nested stratified positions."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations

def mean_loss(m,tokens,key,ids,n,code):
    low,mid,nmid=code
    h=activations(m,tokens);y=tokens[:,1:];w=m.lm_head.weight.array
    scale=1./m.config.width_multiplier
    rows,t,_=h.shape
    # Each coarse block draws one member from uniformly drawn representatives of its subblocks.
    keys=jax.random.split(key)
    s0=j.arange(nmid)[None,:]*(t//nmid)+jax.random.randint(keys[0],(rows,nmid),0,t//nmid)
    i=j.arange(n)[None,:]*(nmid//n)+jax.random.randint(keys[1],(rows,n),0,nmid//n)
    def take(v,pos):
        return j.take_along_axis(v,pos[:,:,None] if v.ndim==3 else pos,axis=1)
    s1=take(s0,i)
    pos=(j.einsum('btd,btd->bt',h,w[y])*scale).astype(j.float32)
    rank=j.full(w.shape[0],mid,dtype=j.int32).at[ids].set(j.arange(mid))
    def approx(x,k,positive,yr):
        head=jax.nn.logsumexp((x@w[ids[:k]].T*scale).astype(j.float32),-1)
        return j.logaddexp(head,j.where(yr<k,-1e9,positive))
    yr=rank[y]
    a=approx(h,low,pos,yr)
    b=approx(take(h,s0),mid,take(pos,s0),take(yr,s0))
    c=jax.nn.logsumexp((take(h,s1)@w.T*scale).astype(j.float32),-1)
    return (a-pos).mean()+(b-take(a,s0)).mean()+(c-take(b,i)).mean()

def loss(model,tokens,key,ids,n,code):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    return mean_loss(m,tokens,key,ids,n,code)

@eqx.filter_jit
def sample(model,tokens,key,ids,n,code):
    return ravel_pytree(eqx.filter_grad(loss)(model,tokens,key,ids,n,code))[0]

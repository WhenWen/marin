"""A target-aware nested softmax backward using contiguous vocabulary rows and one dense gradient buffer."""
from functools import partial
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations,full_losses

@eqx.filter_jit
def setup(model,cal):
    w=model.lm_head.weight.array
    counts=j.bincount(cal[:,1:].reshape(-1),length=w.shape[0])
    order=j.argsort(-counts,stable=True)
    inv=j.zeros_like(order).at[order].set(j.arange(len(order)))
    permuted=eqx.tree_at(lambda v:v.lm_head.weight.array,model,w[order])
    # The two Adam moment tables use this same row permutation in a real training run.
    return permuted,inv

def parts(h,w,y,pos,k):
    b,t,d=h.shape;rows=j.arange(b)[:,None]
    small=(h@w[:k].T).astype(j.float32)
    true_y=(h*w[y]).astype(j.float32).sum(-1)
    extra=j.where(y>=k,true_y,-1e9)
    z=j.logaddexp(jax.nn.logsumexp(small,-1),extra)
    sel=h[rows,pos]
    big=(sel@w.T).astype(j.float32)
    bigz=jax.nn.logsumexp(big,-1)
    value=(z-true_y).mean()+(bigz-z[rows,pos]).mean()
    return value,(h,w,y,pos,j.exp(small-z[:,:,None]),j.exp(extra-z),j.exp(big-bigz[:,:,None]))

@partial(jax.custom_vjp,nondiff_argnums=(4,))
def ce(h,w,y,pos,k):
    return parts(h,w,y,pos,k)[0]
def forward(h,w,y,pos,k):return parts(h,w,y,pos,k)
def backward(k,res,v):
    h,w,y,pos,q,r,p=res;b,t,d=h.shape;n=pos.shape[1];rows=j.arange(b)[:,None]
    c=t/n
    a=j.ones((b,t),j.float32).at[rows,pos].add(-c)
    norm=v/(b*t)
    left=(q*(a*norm)[:,:,None]).astype(h.dtype).reshape((-1,k))
    right=(p*(c*norm)).astype(h.dtype).reshape((-1,w.shape[0]))
    target=((a*r-1)*norm).astype(h.dtype)[:,:,None]
    dh=(left@w[:k]).reshape(h.shape)+target*w[y]
    dh=dh.at[rows,pos].add((right@w).reshape((b,n,d)))
    dw=right.T@h[rows,pos].reshape((-1,d))
    dw=dw.at[:k].add(left.T@h.reshape((-1,d)))
    dw=dw.at[y.reshape(-1)].add((target*h).reshape((-1,d)))
    return dh,dw,None,None
ce.defvjp(forward,backward)

def loss(model,tok,inv,key,k,n):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=activations(m,tok); y=inv[tok[:,1:]];w=m.get_lm_head().rearrange(('vocab','embed')).array
    if k==0:return full_losses(x,w,y,1.).mean()
    t=x.shape[1];pos=j.arange(n)[None,:]*(t//n)+jax.random.randint(key,(x.shape[0],n),0,t//n)
    return ce(x,w,y,pos,k)
@eqx.filter_jit
def sample(m,tok,inv,key,k,n):return ravel_pytree(eqx.filter_grad(loss)(m,tok,inv,key,k,n))[0]

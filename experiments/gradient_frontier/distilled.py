"""Fit a forward-feature predictor of the missing hidden-state gradient using disjoint calibration tokens."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations,full_losses,shortlist

def proxy_parts(h,w,y,ids,scale):
    vec=w[ids]
    logits=(j.matmul(h,vec.T)*scale).astype(j.float32)
    positive=(j.einsum('btd,btd->bt',h,w[y])*scale).astype(j.float32)
    # Include the true target when it is absent from the static candidate dictionary.
    member=j.zeros((w.shape[0],),bool).at[ids].set(True)[y]
    tiny=-1e9
    other=j.where(member,tiny,positive)
    z=j.logaddexp(jax.nn.logsumexp(logits,-1),other)
    prob=j.exp(logits-z[:,:,None])
    wtarget=w[y].astype(j.float32)
    e=(j.matmul(prob.astype(j.bfloat16),vec).astype(j.float32)+j.exp(other-z)[:,:,None]*wtarget)*scale
    weight=j.exp(positive-z)[:,:,None]
    # The label vector and its estimated probability help predict confident rare targets.
    coords=h.astype(j.float32)
    norms=coords/j.maximum(j.sqrt((coords**2).mean(-1,keepdims=True)),1e-4)
    intop=j.exp(j.max(logits,axis=-1)-z)[:,:,None]
    f=j.concatenate([norms,e,wtarget*scale,wtarget*scale*weight,
        e*weight,e*intop,norms*intop,norms*weight,z[:,:,None],weight,intop,member[:,:,None]],axis=-1)
    return z-positive,e,jax.lax.stop_gradient(f)

@eqx.filter_jit
def fit(model,cal,dict_data,code):
    k=code%100000; n=code//100000
    ids=shortlist(dict_data,k,model.Vocab.size)
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    w=m.lm_head.weight.array;scale=1./m.config.width_multiplier
    def block(x):
        h=activations(m,x)
        _,e,f=proxy_parts(h,w,x[:,1:],ids,scale)
        logits=(h@w.T*scale).astype(j.float32)
        true=(jax.nn.softmax(logits).astype(j.bfloat16)@w).astype(j.float32)*scale
        f=f.reshape((-1,f.shape[-1]));v=(true-e).reshape((-1,e.shape[-1]))
        return f.T@f,f.T@v,f.sum(0),v.sum(0)
    # Each block bounds calibration activation memory. All reference work is timed.
    x=cal[:n].reshape((-1,2,cal.shape[-1]))
    first=block(x[0])
    def add(accum,v):
        new=block(v)
        return jax.tree.map(j.add,accum,new),None
    (xx,xy,sx,sy),_=jax.lax.scan(add,first,x[1:])
    N=n*(cal.shape[-1]-1)
    mx=sx/N;my=sy/N
    cov=xx/N-j.outer(mx,mx);covxy=xy/N-j.outer(mx,my)
    std=j.sqrt(j.maximum(j.diag(cov),1e-10))
    corr=cov/std[:,None]/std[None,:]
    # Scaling before ridge treats all feature groups comparably.
    coeff=j.linalg.solve(corr+0.15*j.eye(len(corr)),covxy/std[:,None])/std[:,None]
    return ids,coeff,my-mx@coeff

def objective(model,tokens,params,b,kind,power):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    h=activations(m,tokens);w=m.lm_head.weight.array;y=tokens[:,1:]
    scale=1./m.config.width_multiplier
    if kind=='full':return full_losses(h,w,y,scale).mean()
    ids,coeff,inter=params
    base,_,f=proxy_parts(h,w,y,ids,scale)
    residual=jax.lax.stop_gradient(f@coeff+inter)
    cheap=(base+(h.astype(j.float32)*residual).sum(-1)).mean(1)
    if kind=='cheap':return cheap.mean()
    return full_losses(h[:b],w,y[:b],scale).mean()+power*(cheap.mean()-cheap[:b].mean())

def grad_vec(model,tokens,params,b,kind,power=1.):
    return ravel_pytree(eqx.filter_grad(objective)(model,tokens,params,b,kind,power))[0]
estimator=eqx.filter_jit(grad_vec)
per_example=eqx.filter_jit(lambda model,tokens,params,kind:jax.vmap(lambda x:grad_vec(model,x[None],params,1,kind))(tokens))

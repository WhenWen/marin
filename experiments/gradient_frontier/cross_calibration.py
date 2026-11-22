"""Cross-fit a multivariate regression of complete token gradients using only scalar loss weights."""
import jax
import jax.numpy as j
from .group_regression import fit as group_fit,levels as group_levels


def levels(kind):
    return group_levels(113) if kind in (144,145,146,147) else (2,)


def fit(model,data,k,kind):
    if kind in (144,145,146,147):return group_fit(model,data,4096,113)
    # No supervised preprocessing. The response labels enter through dual ridge weights during the current step.
    return j.arange(1)


def weights(feature,chosen,iw,w,penalty):
    # Arrays are [fold, strata, width, feature]. Each fold observes one point per stratum.
    nf,ng,width,d=feature.shape
    sampled=j.take_along_axis(feature,chosen[:,:,None,None],axis=2)[:,:,0,:]
    wt=j.take_along_axis(w,chosen[:,:,None],axis=2)[:,:,0]
    observed=iw*wt
    gap=(feature*w[:,:,:,None]).sum((1,2))-(sampled*observed[:,:,None]).sum(1)
    def coefficients(X,opposite):
        cov=X.T@X+penalty*ng*j.eye(d)
        return X@jax.scipy.linalg.solve(cov,opposite,assume_a='pos')
    # The full-gradient regression on fold j is independent of sampling in fold 1-j.
    return observed+jax.vmap(coefficients)(sampled,gap[::-1])


def objective(x,w,tok,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);b,t,d=x.shape;width=t//n
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    if kind in (144,145,146,147):
        ids,_,group,_,beta=bank
        from .group_regression import phi
        pred=(phi(x).astype(j.bfloat16)@beta).astype(j.float32)
        a=jax.nn.logsumexp((x@w[ids].T).astype(j.float32),-1);tail=jax.nn.logsumexp(pred,-1)
        z=j.logaddexp(j.logaddexp(a,tail),j.where(group[y]>=0,pos,-j.inf))
        cheap=z-pos
        score=.06+j.sqrt(j.exp(tail-z))+.10*(1.-j.minimum(1.,j.exp(pos-z)))
    else:
        cheap=-pos
        score=j.ones_like(pos)
    # Sketch parameter-gradient structure using activations, response embeddings and tensor-product features.
    wy=w[y];wc=w[tok]
    choose=k//4;proj=jax.random.normal(jax.random.key(903),(2,d,choose)).astype(j.bfloat16)/j.sqrt(float(d))
    hx=(x@proj[0]).astype(j.float32);ey=(wy@proj[1]).astype(j.float32);cx=(wc@proj[1]).astype(j.float32)
    feature=j.concatenate((hx,ey,hx*ey,cx*ey),-1)
    feature=(feature-feature.mean((0,1)))/j.maximum(1e-4,feature.std((0,1)))
    feature=j.concatenate((j.ones_like(feature[:,:,:1]),j.clip(feature,-8,8)),-1)
    # Interleave strata across the two folds, including all sequences in both folds.
    def fold(a):return j.moveaxis(a.reshape(b,n//2,2,width,*a.shape[2:]),2,0).reshape(2,-1,width,*a.shape[2:])
    f=jax.lax.stop_gradient(fold(feature));ww=fold(wt);p=jax.lax.stop_gradient(fold(score));p=p/p.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(*p.shape[:-1],1));at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    iw=1/j.take_along_axis(p,at[:,:,None],axis=2)[:,:,0]
    ridge={140:.15,141:.6,142:1.5,143:.05,144:.15,145:.6,146:1.5,147:.05}[kind]
    co=jax.lax.stop_gradient(weights(f,at,iw,ww,ridge)).reshape(-1)
    # Every chosen row is evaluated against the complete checkpoint vocabulary.
    index=fold(j.arange(b*t).reshape(b,t))
    sel=j.take_along_axis(index,at[:,:,None],axis=2).reshape(-1)
    hidden=x.reshape(-1,d)[sel];target=y.reshape(-1)[sel]
    logits=(hidden@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,None],-1)[:,0]
    return ((wt*cheap).sum()+j.vdot(co,exact-cheap.reshape(-1)[sel]))/wt.sum()

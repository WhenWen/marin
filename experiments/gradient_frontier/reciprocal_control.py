"""Learn a vocabulary importance proposal and cross-fit a polynomial reciprocal of the partition."""
import equinox as eqx
import jax
import jax.numpy as j
from .group_regression import phi
from .tail import trunk


def levels(kind):
    return (4,)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda v:v.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    sample=x[::32];p=jax.nn.softmax((sample@w.T).astype(j.float32),axis=-1)
    rms=j.sqrt(j.mean(p*p,axis=0))
    head=jax.lax.top_k(rms,k)[1]
    remaining=j.ones(len(w),bool).at[head].set(False)
    qq=j.where(remaining,rms,0.)
    qq=.95*qq/qq.sum()+.05*remaining/remaining.sum()
    cdf=j.cumsum(qq);cdf=cdf/cdf[-1]
    def label(h):
        scores=(h@w.T).astype(j.float32)
        return jax.nn.logsumexp(j.where(remaining,scores,-j.inf),axis=-1)
    z=jax.lax.map(label,x.reshape(-1,64,w.shape[1])).reshape(-1)
    ff=phi(x);sd=j.maximum(.01,j.sqrt((ff*ff).mean(0)));f=ff/sd
    beta=jax.scipy.linalg.solve(f.T@f+.08*len(f)*j.eye(f.shape[1]),f.T@(z-z.mean()),assume_a='pos')/sd
    beta=beta.at[0].add(z.mean())
    return head,remaining,qq,cdf,beta


def control(x,w,y,bank,key,kind):
    head,tail,qq,cdf,beta=bank
    size=len(head)//2 if kind%4<2 else len(head)
    # Latin-hypercube proposals are independent between panels, with expected multiplicity size*q_v.
    draws=(j.arange(size)[None,:]+jax.random.uniform(key,(3,size)))/size
    ids=j.minimum(j.searchsorted(cdf,draws,side='right'),len(w)-1)
    head_s=(x@w[head].T).astype(j.float32)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    logtail=phi(x)@beta
    z0=jax.lax.stop_gradient(j.logaddexp(jax.nn.logsumexp(head_s,-1),j.maximum(logtail,j.where(tail[y],pos,-j.inf))))
    # Include an observed tail label exactly in every panel.
    target=j.where(tail[y],j.exp(j.minimum(0.,pos-z0)),0.)
    head_p=j.exp(head_s-z0[:,:,None])
    fixed=head_p.sum(-1)+target
    fixed_linear=(jax.lax.stop_gradient(head_p)*head_s).sum(-1)+jax.lax.stop_gradient(target)*pos
    s=(x@w[ids.reshape(-1)].T).astype(j.float32).reshape(*y.shape,3,size)
    log_importance=s-z0[:,:,None,None]-j.log(j.maximum(qq[ids]*size,1e-30))
    mass=j.where((ids!=y[:,:,None,None])&tail[ids],j.exp(j.clip(log_importance,-50.,8.)),0.)
    linear=fixed_linear[:,:,None]+(jax.lax.stop_gradient(mass)*s).sum(-1)
    totals=fixed[:,:,None]+mass.sum(-1)
    # Independent panels supply denominator factors for numerator panel i. Stabilization is free
    # to alter the predictor since the later position correction conditions on all three panels.
    e=jax.lax.stop_gradient(j.clip(1.-totals,-1.5,.99))
    right=j.roll(e,1,axis=-1);left=j.roll(e,2,axis=-1)
    factor=1.+.5*(left+right)
    if kind%2:factor=factor+right*left
    proxy=(jax.lax.stop_gradient(j.clip(factor,.05,3.))*linear).mean(-1)-pos
    score=.10+j.clip(j.std(totals,axis=-1),0.,4.)+.5*j.sqrt(j.clip(j.exp(logtail-z0),0.,1.))
    return proxy,jax.lax.stop_gradient(score)


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind=code//100000;b,t,d=x.shape
    sample,locations=jax.random.split(key)
    cheap,score=control(x,w,y,bank,sample,kind)
    width=t//n
    s=score.reshape(b,n,width) if kind<264 else j.ones((b,n,width))
    prob=s/s.sum(-1,keepdims=True);u=jax.random.uniform(locations,(b,n,1))
    ofs=j.minimum((u>=j.cumsum(prob,-1)).sum(-1),width-1)
    samplepos=j.arange(n)[None,:]*width+ofs
    inv=1./j.take_along_axis(prob,ofs[:,:,None],-1)[:,:,0]
    a=j.take_along_axis(x,samplepos[:,:,None],axis=1);target=j.take_along_axis(y,samplepos,axis=1)
    full=(a@w.T).astype(j.float32)
    gold=jax.nn.logsumexp(full,-1)-j.take_along_axis(full,target[:,:,None],-1)[:,:,0]
    residual=(gold-j.take_along_axis(cheap,samplepos,1))*j.take_along_axis(wt,samplepos,1)*inv
    return ((cheap*wt).sum()+residual.sum())/wt.sum()

"""Locally tilt the exact softmax distribution at learned hidden-state landmarks."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4,)


def settings(kind):
    landmarks=32 if kind in (190,191,194,195) else 96
    return landmarks,kind%2==1,kind>=194


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1]).astype(j.float32)
    n,dense,adaptive=settings(kind)
    c=x[jax.random.permutation(jax.random.key(1229),len(x))[:n]]
    def assign(points,means):
        return j.argmax((points.astype(j.bfloat16)@means.astype(j.bfloat16).T).astype(j.float32)-.5*(means*means).sum(1),axis=1)
    for _ in range(5):
        labels=assign(x,c);cnt=j.bincount(labels,length=n)
        c=j.where(cnt[:,None]>0,jax.ops.segment_sum(x,labels,n)/j.maximum(1,cnt[:,None]),c)
    logits=(x[::32].astype(j.bfloat16)@w.T).astype(j.float32)
    important=jax.lax.top_k((jax.nn.softmax(logits,-1)**2).mean(0),k)[1]
    tail=j.ones(len(w),bool).at[important].set(False)
    # Align the quadratic coordinates with the covariance of common softmax-weighted vocabulary rows.
    probs=jax.nn.softmax(j.where(tail[None,:],(c.astype(j.bfloat16)@w.T).astype(j.float32),-j.inf),-1)
    mass=probs.mean(0);center=mass@w.astype(j.float32)
    ww=w.astype(j.float32)-center;cov=(ww*mass[:,None]).T@ww
    _,rot=j.linalg.eigh(cov)
    bank=refresh(w,c,tail,rot)
    return important,tail,c,rot,bank


def refresh(w,c,tail,rot):
    scores=j.where(tail[None,:],(c.astype(j.bfloat16)@w.T).astype(j.float32),-j.inf)
    z=jax.nn.logsumexp(scores,-1)
    p=j.exp(scores-z[:,None]).astype(j.bfloat16)
    projected=(w@rot.astype(j.bfloat16)).astype(j.float32)
    mean=(p@w).astype(j.float32);mu=mean@rot
    second=(p@(projected*projected).astype(j.bfloat16)).astype(j.float32)
    variance=j.maximum(0.,second-mu*mu)
    cc=c@rot
    linear=mean-(cc*variance)@rot.T
    bias=z-j.sum(mean*c,axis=1)+.5*j.sum(cc*cc*variance,axis=1)
    return linear.astype(j.bfloat16),variance.astype(j.bfloat16),bias,p


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000)
    num,dense,adaptive=settings(kind)
    ids,tail,c,rot,mom=bank
    stop=jax.lax.stop_gradient
    if adaptive:mom=jax.tree.map(stop,refresh(stop(w),c,tail,rot))
    lin,var,bias,prob=mom
    # Smoothly blend the closest conditional distributions, freezing the routing in the control's VJP.
    score=(x@c.astype(x.dtype).T).astype(j.float32)-.5*(c*c).sum(1)
    route=stop(jax.nn.softmax(score/10.,-1))
    proj=(x@rot.astype(x.dtype)).astype(j.float32)
    local=(x@lin.T).astype(j.float32)+.5*((proj*proj).astype(j.bfloat16)@var.T).astype(j.float32)+bias
    log_tail=(route*local).sum(-1)
    head=jax.nn.logsumexp((x@w[ids].T).astype(j.float32),-1)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    if dense:
        pooled=prob@w
        correction=route.astype(j.bfloat16)@(pooled-stop(pooled))
        log_tail=log_tail+j.einsum('btd,btd->bt',stop(x),correction).astype(j.float32)
    z=j.logaddexp(j.logaddexp(head,log_tail),j.where(tail[y],pos,-j.inf))
    cheap=z-pos
    b,t,d=x.shape;width=t//n
    mass=stop(.04+j.sqrt(j.exp(log_tail-z))+.12*(1.-j.minimum(1.,j.exp(pos-z))))
    p=mass.reshape(b,n,width);p=p/p.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    iw=1/j.take_along_axis(p,at[:,:,None],-1)[:,:,0]
    ix=j.arange(n)[None,:]*width+at;h=j.take_along_axis(x,ix[:,:,None],1);target=j.take_along_axis(y,ix,1)
    logits=(h@w.T).astype(j.float32);true=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    delta=(true-j.take_along_axis(cheap,ix,1))*j.take_along_axis(wt,ix,1)*iw
    return ((cheap*wt).sum()+delta.sum())/wt.sum()

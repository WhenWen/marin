"""Frozen FP8 readout dictionary and a calibrated tail adjoint, with full-vocabulary residuals on sampled positions."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4 if kind%2 else 1,)


def pack(x):
    scale=j.maximum(j.max(j.abs(x),axis=-1,keepdims=True).astype(j.float32)/400.,1e-10)
    return (x.astype(j.float32)/scale).astype(j.float8_e4m3fn),scale


def product(a,b):
    return jax.lax.dot_general(a,b,(((1,),(0,)),((),())),preferred_element_type=j.float32)


def features(h,head):
    z=jax.nn.logsumexp(head,axis=-1,keepdims=True)
    top=j.max(head,axis=-1,keepdims=True)
    tail=jax.nn.logsumexp(head[...,head.shape[-1]//2:],axis=-1,keepdims=True)
    h=h.astype(j.float32)
    return j.concatenate((j.ones_like(z),h,h*h,z,top,tail),axis=-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[1])
    # All calibration labels are from the starting checkpoint's separate four-document partition.
    if kind in (292,293,296,297):
        labels=jax.nn.softmax((x[::16]@w.T).astype(j.float32),axis=-1)
        ids=jax.lax.top_k(j.mean(labels*labels,axis=0),k)[1]
    else:ids=j.arange(k)
    member=j.zeros(len(w),bool).at[ids].set(True)
    packed,s=pack(w[ids]);A,sa=pack(x)
    head=product(A,packed.T)*sa*s.T
    def labels(xx):
        vals=(xx@w.T).astype(j.float32)
        tail=j.where(member,-j.inf,vals)
        tz=jax.nn.logsumexp(tail,axis=1)
        mu=j.exp(tail-tz[:,None]).astype(j.bfloat16)@w
        return j.concatenate((tz[:,None],mu.astype(j.float32)),axis=1)
    yy=jax.lax.map(labels,x.reshape(-1,64,x.shape[1])).reshape(len(x),-1)
    X=features(x,head);scale=j.sqrt(j.maximum(j.mean(X*X,axis=0),.0001))
    F=X/scale;mean=yy.mean(0)
    coef=j.linalg.solve(F.T@F+.05*len(X)*j.eye(F.shape[1]),F.T@(yy-mean))/scale[:,None]
    coef=coef.at[0].add(mean)
    return ids,packed,s,coef.astype(j.bfloat16)


def objective(x,w,y,weight,key,config,bank,forced=None):
    method,k=divmod(config[0],100000);n=config[1]
    B,T,D=x.shape
    stop=jax.lax.stop_gradient
    ids,packed,s,coef=bank
    h=stop(x.reshape(-1,D));a,sc=pack(h)
    head=product(a,packed.T)*sc*s.T
    tail=(features(h,head).astype(j.bfloat16)@coef).astype(j.float32)
    loghead=jax.nn.logsumexp(head,axis=-1)
    z=j.logaddexp(loghead,tail[:,0])
    mass=j.exp(tail[:,0]-z)
    prob=j.exp(head-z[:,None])
    # The dictionary supplies only stopped adjoint predictions. All differentiation below is linear in the residual weights.
    qp,sp=pack(prob*s.T)
    mu=product(qp,packed)*sp+mass[:,None]*tail[:,1:]
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    cheap=j.sum((x-stop(x)).astype(j.float32)*stop(mu.reshape(B,T,D)),axis=-1)-positive
    keep=512 if method in (294,295,296,297) else 64
    keep=min(k,keep)
    q=stop(prob[:,:keep].astype(j.bfloat16))
    forward=(h@w[ids[:keep]].T).astype(j.float32)
    read=j.sum(q.astype(j.float32)*(forward-stop(forward)),axis=-1)
    # Carry the most probable head token outside the retained dense readout slice.
    arg=j.argmax(prob,axis=-1);best=stop(j.max(prob,axis=-1)*(arg>=keep))
    top=j.sum(h*w[ids[arg]],axis=-1,dtype=j.float32)
    cheap=cheap+(read+best*(top-stop(top))).reshape(B,T)
    width=T//n
    score=stop(.06+j.sqrt(mass.reshape(B,n,width)))
    p=score/score.sum(-1,keepdims=True)
    if forced is None:
        u=jax.random.uniform(key,(B,n,1));at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    else:at=forced
    inv=1./j.take_along_axis(p,at[:,:,None],axis=2)[:,:,0]
    index=j.arange(n)[None,:]*width+at
    sel=lambda v:j.take_along_axis(v,index,axis=1)
    hx=j.take_along_axis(x,index[:,:,None],axis=1)
    scores=(hx@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(scores,axis=-1)-j.take_along_axis(scores,sel(y)[:,:,None],-1)[:,:,0]
    return ((weight*cheap).sum()+(sel(weight)*(exact-sel(cheap))*inv).sum())/weight.sum()

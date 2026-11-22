"""Predict log partition for every vocabulary group from held-out full-vocabulary teacher evaluations."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4 if kind in (112,113,116,117) else 2,)


def settings(kind):
    return (128 if kind in (110,112,114,116) else 256, kind>=114)


def phi(x):
    h=x.astype(j.float32)
    return j.concatenate((j.ones_like(h[...,:1]),h,h*h),axis=-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    width,_=settings(kind)
    # Give high-impact rows singleton groups. Other groups have equal capacity and use supervised context projections.
    calibration=x[::max(1,len(x)//128)][:128]
    scores=(calibration@w.T).astype(j.float32)
    p=jax.nn.softmax(scores,axis=-1)
    rank=j.argsort(-j.mean(p*p,axis=0))
    important=rank[:k]
    others=rank[k:]
    # Sort by a binary signature of the row's scores on representative calibration contexts.
    signatures=(scores[:16,others]>j.median(scores[:16,others],axis=-1,keepdims=True)).astype(j.int32)
    code=(signatures*(2**j.arange(16))[:,None]).sum(0)
    order=others[j.argsort(code)]
    pad=(-len(order))%width
    sorted_ids=j.pad(order,(0,pad),constant_values=w.shape[0])
    table=j.pad(w,((0,1),(0,0)))[sorted_ids]
    g=len(sorted_ids)//width
    mask=sorted_ids.reshape(g,width)<w.shape[0]
    def labels(h):
        values=(h@table.T).reshape(len(h),g,width).astype(j.float32)
        values=j.where(mask,values,-j.inf)
        z=jax.nn.logsumexp(values,-1)
        return z,j.exp(values-z[:,:,None]).sum(0)
    z,prior=jax.lax.map(labels,x.reshape(-1,128,x.shape[-1]))
    prior=prior.sum(0)/len(x)
    target=z.reshape(len(x),g)
    f=phi(x);scale=j.maximum(.01,j.sqrt(j.mean(f*f,0)));ff=f/scale
    reg=.03
    center=target.mean(0)
    beta=j.linalg.solve(ff.T@ff+reg*len(x)*j.eye(ff.shape[-1]),ff.T@(target-center))/scale[:,None]
    beta=beta.at[0].add(center)
    reverse=j.full((w.shape[0]+1,),-1,j.int32).at[sorted_ids].set(j.repeat(j.arange(g),width))
    return important,sorted_ids,reverse,prior.astype(j.bfloat16),beta.astype(j.bfloat16)


def objective(x,w,y,wt,key,config,bank,offsets=None):
    code,n=config;kind,k=divmod(code,100000);width,dense=settings(kind)
    ids,sorted_ids,group,prior,beta=bank
    b,t,d=x.shape;g=prior.shape[0];stop=jax.lax.stop_gradient
    common=(x@w[ids].T).astype(j.float32)
    pred=(phi(x).astype(j.bfloat16)@beta).astype(j.float32)
    if dense:
        grouped=j.pad(w,((0,1),(0,0)))[sorted_ids].reshape(g,width,d)
        agg=j.einsum('gw,gwd->gd',prior,grouped)
        pred=pred+(stop(x)@(agg-stop(agg)).T).astype(j.float32)
    head_z=jax.nn.logsumexp(common,-1);tail_z=jax.nn.logsumexp(pred,-1)
    z=j.logaddexp(head_z,tail_z)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    # Target's exact gradient is retained. A target outside the singleton groups is an extra learned control feature.
    z=j.logaddexp(z,j.where(group[y]>=0,pos,-j.inf))
    cheap=z-pos
    if n==0:return (cheap*wt).sum()/wt.sum()
    score=.06+j.sqrt(j.exp(tail_z-z))+.10*(1-j.minimum(1.,j.exp(j.maximum(j.max(common,-1),pos)-z)))
    stride=t//n;pr=stop(score.reshape(b,n,stride));pr=pr/pr.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(pr,-1)).sum(-1),stride-1)
    if offsets is None:inv=1/j.take_along_axis(pr,at[:,:,None],-1)[:,:,0]
    else:at=offsets;inv=stride
    sel=j.arange(n)[None,:]*stride+at
    h=j.take_along_axis(x,sel[:,:,None],1);target=j.take_along_axis(y,sel,1)
    true_scores=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(true_scores,-1)-j.take_along_axis(true_scores,target[:,:,None],-1)[:,:,0]
    correction=(exact-j.take_along_axis(cheap,sel,1))*j.take_along_axis(wt,sel,1)*inv
    return ((wt*cheap).sum()+correction.sum())/wt.sum()

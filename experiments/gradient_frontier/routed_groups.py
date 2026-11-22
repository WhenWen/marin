"""Refine influential vocabulary clusters from a learned partition at the current forward activations."""
import jax
import jax.numpy as j
from . import group_regression as groups


def levels(kind):
    return groups.levels(113)


def fit(model,data,k,kind):
    return groups.fit(model,data,k,113)


def refine(x,w,ids,proposal,block,readout):
    b,t,d=x.shape;flat=x.reshape(-1,d);route=proposal.reshape(-1)
    perm=j.argsort(route,stable=True);layout=perm.reshape(-1,block)
    chosen=route[layout[:,block//2]]
    local=ids[chosen];table=w[j.minimum(local,len(w)-1)]
    if not readout:table=jax.lax.stop_gradient(table)
    scores=j.einsum('gtd,gvd->gtv',flat[layout],table).astype(j.float32)
    val=jax.nn.logsumexp(j.where(local[:,None,:]<len(w),scores,-j.inf),-1)
    def undo(a):
        return j.zeros(b*t,a.dtype).at[perm].set(a.reshape(-1)).reshape(b,t)
    return undo(j.broadcast_to(chosen[:,None],layout.shape)),undo(val)


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);head,flat_ids,group,prior,beta=bank
    count=2 if kind%4<2 else 4
    block=8 if kind<244 else 32
    candidates=flat_ids.reshape(prior.shape)
    logits=(groups.phi(x).astype(j.bfloat16)@beta).astype(j.float32)
    ranking=jax.lax.top_k(logits,count)[1]
    refined=[];values=[]
    for p in range(count):
        route,val=refine(x,w,candidates,ranking[:,:,p],block,kind%2==1)
        new=j.ones_like(y,dtype=bool)
        for old in refined:new=new&(route!=old)
        refined.append(route);values.append(j.where(new,val,-j.inf))
    a=jax.nn.logsumexp((x@w[head].T).astype(j.float32),-1)
    zpred=jax.nn.logsumexp(logits,-1)
    # All local predictions are measured relative to the same max, making removal stable.
    frac=j.zeros_like(a)
    covered=group[y]<0
    for idx,(route,v) in enumerate(zip(refined,values)):
        old=j.take_along_axis(logits,route[:,:,None],-1)[:,:,0]
        frac+=j.where(j.isfinite(v),j.exp(old-zpred),0.)
        covered=covered|(group[y]==route)
        a=j.logaddexp(a,v)
    rem=zpred+j.log(j.maximum(1.e-7,1.-frac))
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z=j.logaddexp(j.logaddexp(a,rem),j.where(covered,-j.inf,pos));cheap=z-pos
    b,t,d=x.shape;width=t//n
    p=jax.lax.stop_gradient(.04+j.sqrt(j.exp(rem-z))+.12*(1.-j.minimum(1.,j.exp(pos-z)))).reshape(b,n,width)
    p=p/p.sum(-1,keepdims=True);u=jax.random.uniform(key,(b,n,1))
    at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    ix=at+j.arange(n)[None,:]*width;iw=1/j.take_along_axis(p,at[:,:,None],-1)[:,:,0]
    h=j.take_along_axis(x,ix[:,:,None],1);target=j.take_along_axis(y,ix,1)
    exact=(h@w.T).astype(j.float32)
    loss=jax.nn.logsumexp(exact,-1)-j.take_along_axis(exact,target[:,:,None],-1)[:,:,0]
    diff=(loss-j.take_along_axis(cheap,ix,1))*j.take_along_axis(wt,ix,1)*iw
    return ((wt*cheap).sum()+diff.sum())/wt.sum()

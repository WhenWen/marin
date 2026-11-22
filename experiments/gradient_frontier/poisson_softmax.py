"""An exactly continued coupled Metropolis estimator with a learned independence proposal."""
import jax
import jax.numpy as j
from .proposal_chain import fit as learn_proposal


def levels(kind):
    return (4,)


def fit(model,data,k,kind):
    return learn_proposal(model,data,32 if kind%2==0 else 64,250)


def potential(logits,proposal,left,right):
    """Poisson solution difference for the independent MH kernel, linear in the differentiated logits."""
    stop=jax.lax.stop_gradient
    pi=stop(jax.nn.softmax(logits,axis=-1))
    q=stop(proposal)
    order=j.argsort(pi/q,axis=-1)
    pi=j.take_along_axis(pi,order,-1);q=j.take_along_axis(q,order,-1)
    f=j.take_along_axis(logits,order,-1)
    before=lambda v:j.pad(j.cumsum(v,axis=-1)[:,:-1],((0,0),(1,0)))
    prefix=before(pi)
    acceptance=j.clip(prefix*q/j.maximum(pi,1e-35)+j.cumsum(q[:,::-1],axis=-1)[:,::-1],1e-30,1.)
    inv=stop(1/acceptance)
    avgh=before(pi*f)/j.maximum(prefix,1e-35)
    u=f*inv-j.cumsum(j.diff(inv,axis=1,prepend=inv[:,:1])*avgh,axis=-1)
    l=j.argmax(order==left[:,None],axis=1);r=j.argmax(order==right[:,None],axis=1)
    row=j.arange(len(f))
    return u[row,l]-u[row,r]


def objective(x,w,y,wt,key,config,bank,forced=None):
    method,k=divmod(config[0],100000);n=config[1]
    B,T,D=x.shape;V=len(w);width=T//n
    rounds=1 if method in (280,281,284,285) else 4
    stop=jax.lax.stop_gradient
    centers,table,counts=bank
    region=j.argmax(jax.nn.standardize(x.astype(j.float32),axis=-1).astype(centers.dtype)@centers.T,axis=-1)
    head=(x@w[:k].T).astype(j.float32)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z=j.logaddexp(jax.nn.logsumexp(head,axis=-1),j.where(y>=k,pos,-j.inf))
    cdf=stop(j.cumsum(j.exp(head-z[:,:,None]),axis=-1))
    logV=j.log1p(float(V))
    key1,key2=jax.random.split(key)
    random=jax.random.uniform(key1,(B,T,rounds+2,3))
    uniforms=random[...,1].reshape(B*T,-1)
    loc=jax.vmap(lambda cum,s:j.searchsorted(cum,s,side='left',method='scan'))(cdf.reshape(-1,k),uniforms).reshape(B,T,-1)
    short=j.where(loc<k,loc,y[:,:,None])
    reg=region[:,:,None]
    lexical=table[reg,j.minimum((256*random[...,1]).astype(j.int32),255)]
    broad=j.minimum(j.expm1(logV*random[...,1]).astype(j.int32),V-1)
    proposed=j.where(random[...,0]<.6,short,j.where(random[...,0]<.85,lexical,broad))
    def score(id):return j.einsum('btd,btd->bt',x,w[id]).astype(j.float32)
    def ratio(id,s):
        local=j.where((id<k)|(id==y),j.exp(s-z),0.)
        q=.60*local+.25*counts[region,id].astype(j.float32)/256.+.15*j.log1p(1./(1.+id))/logV
        return stop(s-j.log(q))
    start=short[:,:,0]
    l=r=start;sl=sr=score(start);wl=wr=ratio(start,sl)
    control=z-pos
    # One lagged transition, then identical proposals and uniforms for both chains.
    for t in range(rounds+1):
        vv=proposed[:,:,t+1];sv=score(vv);wv=ratio(vv,sv)
        logu=j.log(random[:,:,t+1,2])
        take_l=logu<j.minimum(0.,wv-wl)
        l,sl,wl=(j.where(take_l,a,b) for a,b in ((vv,l),(sv,sl),(wv,wl)))
        if t:
            take_r=logu<j.minimum(0.,wv-wr)
            r,sr,wr=(j.where(take_r,a,b) for a,b in ((vv,r),(sv,sr),(wv,wr)))
        if t<rounds:control=control+sl-sr
    live=l!=r
    logitscale=stop(j.maximum(wl,wr))
    if method>=284:score_p=.001+j.where(live,j.exp(j.clip(.5*(logitscale-z),-3,4)),0.)
    else:score_p=.001+live.astype(j.float32)
    prop=stop(score_p.reshape(B,n,width));prop=prop/prop.sum(axis=-1,keepdims=True)
    if forced is None:
        u=jax.random.uniform(key2,(B,n,1));at=j.minimum((u>=j.cumsum(prop,-1)).sum(-1),width-1)
    else:at=forced
    iw=1./j.take_along_axis(prop,at[:,:,None],2)[:,:,0]
    idx=j.arange(B)[:,None]*T+j.arange(n)[None,:]*width+at
    gather=lambda v:v.reshape(B*T,*v.shape[2:])[idx.reshape(-1)]
    xx,yy,ll,rr,regions,zz,ww=map(gather,(x,y,l,r,region,z,wt))
    iw=iw.reshape(-1)
    def exact(xx,yy,ll,rr,reg,zz):
        # Compute the continuation in bounded-memory groups of positions. All proposal and Poisson weights are stopped.
        lo=(xx@w.T).astype(j.float32)
        ids=j.arange(V)
        q0=j.where((ids<k)|(ids[None,:]==yy[:,None]),j.exp(lo-stop(zz[:,None])),0.)
        q=.6*q0+.25*counts[reg].astype(j.float32)/256.+.15*j.log1p(1./(1.+ids))/logV
        return potential(lo,q,ll,rr)
    size=min(8,len(xx))
    assert len(xx)%size==0
    def one(carry,items):
        xx,yy,ll,rr,reg,zz,weights=items
        def cost():return (exact(xx,yy,ll,rr,reg,zz)*weights).sum()
        out=jax.lax.cond(j.any(ll!=rr),cost,lambda:j.array(0.,j.float32))
        return carry+out,None
    items=(xx,yy,ll,rr,regions,zz,ww*iw)
    cont,_=jax.lax.scan(jax.checkpoint(one),j.array(0.,j.float32),tuple(v.reshape(-1,size,*v.shape[1:]) for v in items))
    return ((control*wt).sum()+cont)/wt.sum()

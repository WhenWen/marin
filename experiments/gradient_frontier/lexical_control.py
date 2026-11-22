"""Learn a token-conditional vocabulary from the checkpoint's own singleton-context predictions."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4,)


def settings(kind):
    return (8192 if kind>=234 else 2048,64 if kind%4>=2 else 32)


def features(x,token,headz,localz,bank):
    c,base,mean=bank;idx=j.minimum(token,len(c)-1)
    valid=(token<len(c))&(token!=len(c)-1)
    x32=x.astype(j.float32)
    local=base[idx]+j.sum((x32-c[idx].astype(j.float32))*mean[idx].astype(j.float32),axis=-1)
    context=j.stack((j.where(valid,local,0.),headz,localz,valid.astype(j.float32)),axis=-1)
    return j.concatenate((j.ones_like(x32[...,:1]),x32,x32*x32,context),axis=-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda v:v.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    N,L=settings(kind);tok=j.arange(N).reshape(-1,128)
    # Distinct segment IDs force each position to attend only to itself in the original forward path.
    c=trunk(m,(tok,j.ones_like(tok,j.float32),j.broadcast_to(j.arange(128),tok.shape))).reshape(N,-1)
    def dictionary(h):
        raw=(h@w[k:].T).astype(j.float32)
        _,near=jax.lax.top_k(raw,L)
        rem=raw.at[j.arange(len(h))[:,None],near].set(-j.inf)
        z=jax.nn.logsumexp(rem,-1);p=j.exp(rem-z[:,None]).astype(j.bfloat16)
        return near+k,z,p@w[k:]
    ids,logz,mean=jax.lax.map(dictionary,c.reshape(-1,64,c.shape[-1]))
    ids=ids.reshape(N,L);logz=logz.reshape(N);mean=mean.reshape(N,-1)
    # Include a fallback for infrequent source IDs. Its selected rows are distinct from the common head.
    ids=j.concatenate((ids,j.arange(k,k+L)[None,:]))
    c=j.pad(c,((0,1),(0,0)));mean=j.pad(mean,((0,1),(0,0)));logz=j.pad(logz,(0,1))
    x=trunk(m,data).reshape(-1,w.shape[1])[::2];token=data[0].reshape(-1)[::2]
    def teach(inp):
        h,tk=inp;near=ids[j.minimum(tk,N)]
        logits=(h@w.T).astype(j.float32);hx=jax.nn.logsumexp(logits[:,:k],-1)
        lex=jax.nn.logsumexp(j.take_along_axis(logits,near,1),-1)
        scores=logits[:,k:].at[j.arange(len(h))[:,None],near-k].set(-j.inf)
        z=jax.nn.logsumexp(scores,-1);pr=j.exp(scores-z[:,None]).astype(j.bfloat16)
        labels=j.concatenate((z[:,None],(pr@w[k:]).astype(j.float32)),1)
        f=features(h,tk,hx,lex,(c,logz,mean))
        return f,labels
    f,target=jax.lax.map(teach,(x.reshape(-1,64,x.shape[-1]),token.reshape(-1,64)))
    f=f.reshape(len(x),-1);target=target.reshape(len(x),-1)
    sd=j.maximum(.02,j.sqrt(j.mean(f*f,0)));u=f/sd;center=target.mean(0)
    coef=jax.scipy.linalg.solve(u.T@u+.06*len(u)*j.eye(u.shape[-1]),u.T@(target-center),assume_a='pos')/sd[:,None]
    return ids,c,logz,mean,coef.at[0].add(center).astype(j.bfloat16)


def objective(x,w,tok,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);ids,c,zz,mu,coef=bank
    stop=jax.lax.stop_gradient;b,t,d=x.shape
    selected=ids[j.minimum(tok,len(ids)-1)]
    mat=w[selected] if kind%2 else stop(w[selected])
    lex=j.einsum('btd,btld->btl',x,mat).astype(j.float32)
    h=jax.nn.logsumexp((x@w[:k].T).astype(j.float32),-1);l=jax.nn.logsumexp(lex,-1)
    f=stop(features(x,tok,h,l,(c,zz,mu))).astype(j.bfloat16)
    pred=stop((f@coef).astype(j.float32))
    tail=pred[:,:,0]+j.sum((x.astype(j.float32)-stop(x.astype(j.float32)))*pred[:,:,1:],axis=-1)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    missing=(y>=k)&~j.any(selected==y[:,:,None],axis=-1)
    z=j.logaddexp(j.logaddexp(j.logaddexp(h,l),tail),j.where(missing,pos,-j.inf));cheap=z-pos
    width=t//n;p=stop(.04+j.sqrt(j.exp(tail-z))+.12*(1.-j.minimum(1.,j.exp(pos-z)))).reshape(b,n,width)
    p=p/p.sum(-1,keepdims=True);u=jax.random.uniform(key,(b,n,1))
    at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1);sel=j.arange(n)[None,:]*width+at
    iw=1/j.take_along_axis(p,at[:,:,None],-1)[:,:,0];xs=j.take_along_axis(x,sel[:,:,None],1);target=j.take_along_axis(y,sel,1)
    logits=(xs@w.T).astype(j.float32);exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    delta=(exact-j.take_along_axis(cheap,sel,1))*j.take_along_axis(wt,sel,1)*iw
    return ((cheap*wt).sum()+delta.sum())/wt.sum()

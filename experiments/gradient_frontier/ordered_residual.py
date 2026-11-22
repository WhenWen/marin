"""Stratify exact residual evaluations by learned covector similarity in the original forward features."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4 if kind % 2 else 1,)


def proxy(x, w, y, k):
    logits=(x@w[:k].T).astype(j.float32)
    pos=j.sum(x*w[y],axis=-1,dtype=j.float32)
    loga=jax.nn.logsumexp(logits,axis=-1)
    logb=jax.nn.logsumexp(logits[...,k//2:],axis=-1)-j.log(2.)
    z=j.logaddexp(j.logaddexp(loga,logb),j.where(y>=k,pos,-j.inf))
    return z-pos,j.exp(logb-z),j.exp(pos-z)


def features(x, target, loss, tail, confidence):
    h=x.astype(j.float32);v=target.astype(j.float32)
    return j.concatenate((h,h*h,v,h*v,j.stack((loss,tail,confidence),-1)),axis=-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[1]);y=j.roll(data[0],-1,axis=1).reshape(-1)
    def labels(args):
        h,t=args
        cheap,tail,conf=proxy(h,w,t,k)
        f=features(h,w[t],cheap,tail,conf)
        def residual(h):
            scores=(h@w.T).astype(j.float32)
            ce=jax.nn.logsumexp(scores,-1)-j.take_along_axis(scores,t[:,None],-1)[:,0]
            return (ce-proxy(h,w,t,k)[0]).sum()
        return f,jax.grad(residual)(h).astype(j.float32)
    f,r=jax.lax.map(labels,(x.reshape(-1,64,x.shape[-1]),y.reshape(-1,64)))
    f=f.reshape(len(x),-1);r=r.reshape(len(x),-1)
    eig,vec=j.linalg.eigh(j.cov(r.T))
    z=j.concatenate((r@vec[:,-3:],j.log1p(j.linalg.norm(r,axis=-1,keepdims=True))),axis=-1)
    fm=f.mean(axis=0);fs=j.maximum(.001,f.std(axis=0));f=(f-fm)/fs
    zm=z.mean(axis=0);zs=j.maximum(.001,z.std(axis=0));z=(z-zm)/zs
    coef=j.linalg.solve(f.T@f+.10*len(f)*j.eye(f.shape[1]),f.T@z)
    beta=coef/fs[:,None]
    return beta.astype(j.bfloat16),-fm@beta,zm[-1],zs[-1]


def groups(pred,mode):
    if mode<2:
        return j.argsort(pred[:,2],stable=True)
    if mode<4:
        return j.argsort(pred[:,3],stable=True)
    # Interleave the three quantized coordinates to keep similar covectors near each other.
    code=j.clip((jax.nn.sigmoid(pred[:,:3])*64).astype(j.uint32),0,63)
    xyz=j.arange(3,dtype=j.uint32)
    def interleave(bit, total):
        digits=((code>>bit)&1) << (3*j.asarray(bit,j.uint32)+xyz)
        return total+j.sum(digits,axis=1,dtype=j.uint32)
    return j.argsort(jax.lax.fori_loop(0,6,interleave,j.zeros(len(pred),j.uint32)),stable=True)


def objective(x,w,y,wt,key,config,bank,forced=None):
    method,k=divmod(config[0],100000);n=config[1];b,t,d=x.shape
    beta,bias,mean,scale=bank
    cheap,mass,conf=proxy(x,w,y,k)
    f=features(x,w[y],cheap,mass,conf)
    pred=jax.lax.stop_gradient(((f.astype(beta.dtype)@beta).astype(j.float32)+bias).reshape(-1,4))
    strata=groups(pred,method-300).reshape(b*n,t//n)
    score=.01+j.expm1(j.clip(mean+scale*pred[:,3],.005,6))[strata]
    p=score/score.sum(-1,keepdims=True) if method>=306 else j.full_like(score,1./score.shape[1])
    offsets=j.minimum((jax.random.uniform(key,(len(p),1))>=j.cumsum(p,axis=1)).sum(axis=1),p.shape[1]-1) if forced is None else forced
    rows=j.arange(len(p));indices=strata[rows,offsets]
    h=x.reshape(-1,d)[indices];targets=y.ravel()[indices];logits=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,axis=-1)-logits[rows,targets]
    residual=(exact-cheap.ravel()[indices])*wt.ravel()[indices]/p[rows,offsets]
    return ((wt*cheap).sum()+residual.sum())/wt.sum(),p

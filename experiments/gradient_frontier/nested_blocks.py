"""Share one teacher forward, then inverse-weight separate token samples for the trunk and readout adjoints."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (1 if kind%2==0 else 4,)


def cheap(h,w,y,k):
    head=(h@w[:k].T).astype(j.float32)
    pos=j.sum(h*w[y],axis=1,dtype=j.float32)
    z0=jax.nn.logsumexp(head,-1)
    tail=jax.nn.logsumexp(head[:,k//2:],-1)-j.log(2.)
    z=j.logaddexp(j.logaddexp(z0,tail),j.where(y>=k,pos,-j.inf))
    q=j.exp(head-z[:,None]);qy=j.where(y>=k,j.exp(pos-z),0.)
    return q,qy,j.stack((z-pos,j.exp(tail-z),j.exp(pos-z)),axis=-1)


def features(h,w,y,s):
    x=h.astype(j.float32);v=w[y].astype(j.float32)
    return j.concatenate((x,x*x,v,x*v,s),axis=1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[1]);y=j.roll(data[0],-1,axis=1).ravel()
    def labels(args):
        h,target=args;q,qy,s=cheap(h,w,target,k)
        p=jax.nn.softmax((h@w.T).astype(j.float32),-1)
        res=p.at[:,:k].add(-q).at[j.arange(len(h)),target].add(-qy)
        rh=res.astype(w.dtype)@w
        norms=j.stack((j.linalg.norm(rh.astype(j.float32),axis=-1),
                       j.linalg.norm(res,axis=-1)*j.linalg.norm(h.astype(j.float32),axis=-1)),axis=-1)
        return features(h,w,target,s),j.log(j.maximum(.01,norms))
    f,z=jax.lax.map(labels,(x.reshape(-1,64,x.shape[1]),y.reshape(-1,64)))
    f=f.reshape(len(x),-1);z=z.reshape(len(x),2)
    mean=f.mean(0);scale=j.maximum(.01,f.std(0));f=(f-mean)/scale
    center=z.mean(0);coef=j.linalg.solve(f.T@f+.07*len(x)*j.eye(f.shape[-1]),f.T@(z-center))/scale[:,None]
    return coef.astype(j.bfloat16),center-mean@coef


def draw(scores,key,forced):
    p=scores/scores.sum(-1,keepdims=True)
    at=(j.minimum((jax.random.uniform(key,(len(p),1))>=j.cumsum(p,axis=1)).sum(1),p.shape[1]-1)
        if forced is None else forced)
    rows=j.arange(len(p))
    return rows*p.shape[1]+at,1./p[rows,at],p


def objective(x,w,y,wt,key,config,bank,forced=None):
    kind,k=divmod(config[0],100000);n=config[1];b,t,d=x.shape
    sparse_hidden=kind in (314,315,316,317)
    sub=4 if kind in (310,311,314,315) else 8
    a,bkey=jax.random.split(key)
    h=x.reshape(-1,d);target=y.ravel();weight=wt.ravel()/wt.sum()
    q,qy,s=cheap(h,w,target,k)
    coef,bias=bank;f=features(h,w,target,s)
    lognorm=(f.astype(coef.dtype)@coef).astype(j.float32)+bias
    norm=j.exp(j.clip(lognorm,-5.,5.))
    score=.03+norm[:,int(sparse_hidden)]/j.maximum(.01,norm[:,int(sparse_hidden)].mean())
    hi,ih,p=draw(score.reshape(-1,t//n),a,None if forced is None else forced[0])
    sh=h[hi];gold=jax.nn.softmax((sh@w.T).astype(j.float32),axis=-1)
    # Exact readout residual norm is available after the shared softmax. A learned hidden norm avoids a full backward GEMM.
    if sparse_hidden:
        score=norm[hi,0]
    else:
        gr=gold.at[:,:k].add(-q[hi]).at[j.arange(len(hi)),target[hi]].add(-qy[hi])
        score=j.linalg.norm(gr,axis=-1)*j.linalg.norm(sh.astype(j.float32),axis=-1)
    low,il,r=draw((.02+score*ih).reshape(-1,sub),bkey,None if forced is None else forced[1])
    all_hi=j.arange(len(hi));counts=ih*weight[hi]
    idx_t,wt_t,idx_r,wt_r=(low,counts[low]*il,all_hi,counts) if sparse_hidden else (all_hi,counts,low,counts[low]*il)
    # Replace cheap positive-softmax terms by their exact counterparts for each independent parameter block.
    ct=weight.at[hi[idx_t]].add(-wt_t);cr=weight.at[hi[idx_r]].add(-wt_r)
    dx=(q*ct[:,None]).astype(w.dtype)@w[:k]
    dx=dx+(ct*qy-weight)[:,None]*w[target]
    dx=dx.at[hi[idx_t]].add((gold[idx_t]*wt_t[:,None]).astype(w.dtype)@w)
    dw=(gold[idx_r]*wt_r[:,None]).astype(h.dtype).T@sh[idx_r]
    dw=dw.at[:k].add((q*cr[:,None]).astype(h.dtype).T@h)
    dw=dw.at[target].add(((cr*qy-weight)[:,None]*h).astype(dw.dtype))
    stop=jax.lax.stop_gradient
    scalar=j.sum((h-stop(h)).astype(j.float32)*stop(dx).astype(j.float32))+j.sum((w-stop(w)).astype(j.float32)*stop(dw).astype(j.float32))
    return scalar,(p,r)

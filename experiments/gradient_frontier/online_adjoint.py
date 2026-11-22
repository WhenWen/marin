"""Cross-fit hidden-state residual adjoints on the current batch before the backbone VJP."""
import jax
import jax.numpy as j
from . import group_regression as groups


def levels(kind):
    return groups.levels(113)


def fit(model,data,k,kind):
    return groups.fit(model,data,k,113)


def proxy(x,w,y,bank):
    ids,_,group,_,beta=bank
    head=(x@w[ids].T).astype(j.float32)
    group_logits=(groups.phi(x).astype(j.bfloat16)@beta).astype(j.float32)
    hlog=jax.nn.logsumexp(head,-1);glog=jax.nn.logsumexp(group_logits,-1)
    pos=j.einsum('...d,...d->...',x,w[y]).astype(j.float32)
    z=j.logaddexp(j.logaddexp(hlog,glog),j.where(group[y]>=0,pos,-j.inf))
    return z-pos,j.exp(glog-z),j.exp(pos-z)


def featurize(x,yw,mass,py,kind):
    width=32 if kind%2==0 else 64;d=x.shape[-1]
    random=jax.random.normal(jax.random.key(182),(d,width)).astype(j.bfloat16)/j.sqrt(float(d))
    hx=(x@random).astype(j.float32);target=(yw@random).astype(j.float32)
    # Target confidence and tail mass gate the nonlinear influence of the observed outcome.
    out=j.concatenate((hx,target,hx*target,hx*mass[:,:,None],target*py[:,:,None],mass[:,:,None],py[:,:,None]),-1)
    out=(out-out.mean((0,1)))/j.maximum(.001,out.std((0,1)))
    return j.concatenate((j.ones_like(out[:,:,:1]),j.clip(out,-7,7)),-1)


@jax.custom_vjp
def teacher_loss(x,w,y):
    return _teacher(x,w,y)[0]


def _teacher(x,w,y):
    logits=(x@w.T).astype(j.float32);z=jax.nn.logsumexp(logits,-1)
    p=j.exp(logits-z[:,None]).astype(x.dtype)
    mean=p@w
    exact=z-j.take_along_axis(logits,y[:,None],-1)[:,0]
    return (exact,jax.lax.stop_gradient(mean-w[y])),(p,x,w,y,mean)


def _backward(saved,cts):
    p,x,w,y,mean=saved;scale=cts[0]
    adj=p.at[j.arange(len(y)),y].add(-1)
    return ((mean-w[y]).astype(j.float32)*scale[:,None]).astype(x.dtype),((adj.astype(j.float32)*scale[:,None]).astype(x.dtype).T@x),None
teacher_loss.defvjp(_teacher,_backward)


def objective(x,w,y,wt,key,config,bank,forced=None,return_prob=False,uniform=False):
    code,n=config;kind,k=divmod(code,100000)
    b,t,d=x.shape;width=t//n;stop=jax.lax.stop_gradient
    cheap,mass,py=proxy(x,w,y,bank)
    score=.06+j.sqrt(mass)+.10*(1-j.minimum(py,1.))
    def fold(a):return j.moveaxis(a.reshape(b,n//2,2,width,*a.shape[2:]),2,0).reshape(2,-1,width,*a.shape[2:])
    p=stop(fold(score));p=p/p.sum(-1,keepdims=True)
    if uniform:p=j.full_like(p,1./width)
    u=jax.random.uniform(key,(*p.shape[:-1],1))
    at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1) if forced is None else forced
    iw=1/j.take_along_axis(p,at[:,:,None],axis=2)[:,:,0]
    grid=fold(j.arange(b*t).reshape(b,t))
    selected=j.take_along_axis(grid,at[:,:,None],2)[:,:,0]
    line=selected.reshape(-1);h=x.reshape(-1,d)[line];target=y.reshape(-1)[line]
    full,adj=teacher_loss(h,w,target)
    # Differentiate the same deterministic proxy at the selected positions only.
    cheap_adj=jax.grad(lambda a:proxy(a,w,target,bank)[0].sum())(h)
    residual=stop((adj-cheap_adj).astype(j.float32)).reshape(2,-1,d)
    f=stop(fold(featurize(x,w[y],mass,py,kind)))
    sf=j.take_along_axis(f,at[:,:,None,None],2)[:,:,0,:]
    reg=.15 if kind in (170,171,174,175) else .65
    def solve(F,R):
        return jax.scipy.linalg.solve(F.T@F+reg*len(F)*j.eye(F.shape[-1]),F.T@R,assume_a='pos')
    coef=jax.vmap(solve)(sf,residual)
    other=j.einsum('fgta,fad->fgtd',f.astype(j.bfloat16),coef[::-1].astype(j.bfloat16))
    if kind>=174:other=other*.5
    # Keeping only the hidden covector here avoids altering the well-estimated readout block twice.
    xfold=fold(x)
    adapt=j.einsum('fgtd,fgtd->fgt',xfold-stop(xfold),stop(other)).astype(j.float32)
    all_cheap=fold(cheap)+adapt;weight=fold(wt)
    chosen_loss=j.take_along_axis(all_cheap,at[:,:,None],2)[:,:,0]
    chosen_weight=j.take_along_axis(weight,at[:,:,None],2)[:,:,0]
    delta=(full.reshape(2,-1)-chosen_loss)*chosen_weight*iw
    total=((weight*all_cheap).sum()+delta.sum())/wt.sum()
    return (total,p) if return_prob else total

"""Moment-matched virtual vocabulary rows calibrated once at the starting checkpoint."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk

def group_count(kind):return {50:128,51:512,52:2048,53:512,54:2048,55:1024}[kind]
def partition(x,bank):
    center,diag,count,coef=bank
    mu=(x@center.T).astype(j.float32)
    sigma=(j.square(x.astype(j.float32)).astype(j.bfloat16)@diag.T).astype(j.float32)
    return jax.nn.logsumexp(mu+coef[0]*sigma+count,axis=-1)+coef[1]

@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    rows=w[k:].astype(j.float32);groups=group_count(kind)
    bits=(groups//4).bit_length()-1
    hyper=jax.random.normal(jax.random.PRNGKey(7362),(rows.shape[1],bits))
    sign=(rows@hyper)>0
    code=(sign.astype(j.int32)*(2**j.arange(bits))).sum(-1)
    norms=j.sqrt((rows*rows).sum(-1))
    limits=j.quantile(norms[::16],j.asarray([.25,.50,.75]))
    code=code+groups//4*(norms[:,None]>limits).sum(-1)
    count=j.bincount(code,length=groups)
    mean=jax.ops.segment_sum(rows,code,groups)/j.maximum(count[:,None],1)
    var=j.maximum(0.,jax.ops.segment_sum(rows*rows,code,groups)/j.maximum(count[:,None],1)-mean*mean)
    logcount=j.where(count>0,j.log(j.maximum(count,1)),-1e9)
    x=trunk(m,data)
    y=j.roll(data[0],-1,axis=1)
    true=(x@w[k:].T).astype(j.float32)
    bi,ti=y.shape
    tailtarget=j.clip(y-k,0,true.shape[-1]-1)
    old=j.take_along_axis(true,tailtarget[:,:,None],-1)[:,:,0]
    true=true.at[j.arange(bi)[:,None],j.arange(ti)[None,:],tailtarget].set(j.where(y>=k,-1e9,old))
    tz=jax.nn.logsumexp(true,-1)
    mean=mean.astype(j.bfloat16);var=var.astype(j.bfloat16)
    coefficients=j.asarray([0.,.1,.2,.3,.4,.5,.65])
    mu=(x@mean.T).astype(j.float32);sigma=((x*x)@var.T).astype(j.float32)
    def trial(alpha):
        approx=jax.nn.logsumexp(mu+alpha*sigma+logcount,-1);diff=tz-approx
        return j.var(diff),j.mean(diff)
    errors,bias=jax.vmap(trial)(coefficients);best=j.argmin(errors)
    return mean,var,logcount,j.stack((coefficients[best],bias[best]))

def objective(x,w,y,weight,key,config,bank):
    code,n=config;kind=code//100000;k=code%100000;b,t,d=x.shape
    head=(x@w[:k].T).astype(j.float32);positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    common=j.logaddexp(jax.nn.logsumexp(head,-1),j.where(y>=k,positive,-1e9))
    tail=partition(x,bank)
    z=j.logaddexp(common,tail)
    if kind<53:score=.025+j.sqrt(j.exp(tail-z))
    else:
        top=j.maximum(positive,j.max(head,-1));score=.03+j.exp(tail-z)+.1*(1-j.minimum(1.,j.exp(top-z)))
    width=t//n;p=jax.lax.stop_gradient(score.reshape((b,n,width)));p=p/p.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));ii=j.minimum((u>=j.cumsum(p,axis=-1)).sum(-1),width-1)
    pos=j.arange(n)[None,:]*width+ii;inv=1./j.take_along_axis(p,ii[:,:,None],-1)[:,:,0]
    chosen=j.take_along_axis(x,pos[:,:,None],1)
    full=jax.nn.logsumexp((chosen@w.T).astype(j.float32),-1)
    delta=(full-j.take_along_axis(z,pos,1))*j.take_along_axis(weight,pos,1)*inv
    return (((z-positive)*weight).sum()+delta.sum())/weight.sum()

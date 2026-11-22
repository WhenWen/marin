"""A deterministic product-quantized full-vocabulary control with randomized exact position corrections."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (2,)


def codebook(inputs,k,key):
    centers=inputs[jax.random.permutation(key,len(inputs))[:k]]
    def assign(rows,c):
        return j.argmax((rows.astype(j.bfloat16)@c.astype(j.bfloat16).T).astype(j.float32)-.5*j.sum(c*c,-1),axis=-1)
    sample=inputs[::8]
    for _ in range(4):
        idx=assign(sample,centers)
        counts=j.bincount(idx,length=k)
        sums=jax.ops.segment_sum(sample,idx,k)
        centers=j.where(counts[:,None]>0,sums/j.maximum(1,counts[:,None]),centers)
    return jax.lax.map(lambda part:assign(part,centers),inputs.reshape(-1,256,inputs.shape[-1])).reshape(-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1]);v,d=w.shape
    h=x[::16].astype(j.float32)
    eig,rot=j.linalg.eigh(h.T@h/len(h)+.1*j.eye(d))
    # Alternating principal coordinates distribute high-signal directions across both codebooks.
    order=j.concatenate((j.arange(0,d,2),j.arange(1,d,2)))
    rot=rot[:,order];scales=j.sqrt(eig[order])
    transformed=(w.astype(j.float32)@rot)*scales
    key1,key2=jax.random.split(jax.random.key(81));half=d//2
    left=codebook(transformed[:,:half],k,key1);right=codebook(transformed[:,half:],k,key2)
    singleton={130:0,131:0,132:256,133:1024,134:2048,135:4096,136:1024,137:2048}[kind]
    true_logits=(h.astype(j.bfloat16)@w.T).astype(j.float32)
    importance=(jax.nn.softmax(true_logits,-1)**2).sum(0)
    head=jax.lax.top_k(importance,singleton)[1] if singleton else j.empty(0,j.int32)
    remaining=j.ones(v,j.float32).at[head].set(0.)
    cpair=j.bincount(left*k+right,weights=remaining,length=k*k).reshape(k,k)
    lcount=j.maximum(1.,cpair.sum(1));rcount=j.maximum(1.,cpair.sum(0))
    def pooled(ww):
        projected=ww.astype(j.float32)@rot
        la=jax.ops.segment_sum(remaining[:,None]*projected[:,:half],left,k)/lcount[:,None]
        rb=jax.ops.segment_sum(remaining[:,None]*projected[:,half:],right,k)/rcount[:,None]
        return la@rot[:,:half].T,rb@rot[:,half:].T
    wa,wb=pooled(w)
    a=h@wa.T;b=h@wb.T
    za=a.max(-1);zb=b.max(-1);aa=j.exp(a-za[:,None]);bb=j.exp(b-zb[:,None])
    raw=za+zb+j.log(j.maximum(1e-20,((aa@cpair)*bb).sum(-1)))
    target=jax.nn.logsumexp(j.where(remaining>0,true_logits,-j.inf),-1)
    err=target-raw;f=j.concatenate((j.ones((len(h),1)),h,h*h),-1)
    sd=j.maximum(.01,j.sqrt((f*f).mean(0)));ff=f/sd
    coef=j.linalg.solve(ff.T@ff+.15*len(h)*j.eye(ff.shape[-1]),ff.T@(err-err.mean()))/sd
    coef=coef.at[0].add(err.mean())
    return head,remaining.astype(j.bfloat16),left,right,cpair.astype(j.bfloat16),rot.astype(j.bfloat16),wa.astype(j.bfloat16),wb.astype(j.bfloat16),coef


def objective(x,w,y,weight,key,config,bank):
    code,n=config;kind,k=divmod(code,100000)
    head,remaining,left,right,counts,rot,wa,wb,coef=bank
    stop=jax.lax.stop_gradient;b,t,d=x.shape
    if kind in (131,136,137):
        projection=(w-stop(w))@rot
        half=d//2
        aa=jax.ops.segment_sum(remaining[:,None]*projection[:,:half],left,k)/j.maximum(1,counts.astype(j.float32).sum(1)[:,None])
        bb=jax.ops.segment_sum(remaining[:,None]*projection[:,half:],right,k)/j.maximum(1,counts.astype(j.float32).sum(0)[:,None])
        wa=wa+aa.astype(wa.dtype)@rot[:,:half].T
        wb=wb+bb.astype(wb.dtype)@rot[:,half:].T
    a=(x@wa.T).astype(j.float32);bpart=(x@wb.T).astype(j.float32)
    ma=stop(a.max(-1));mb=stop(bpart.max(-1))
    lhs=j.exp(a-ma[:,:,None]).astype(j.bfloat16);rhs=j.exp(bpart-mb[:,:,None])
    tail=ma+mb+j.log(j.maximum(1e-12,((lhs@counts).astype(j.float32)*rhs).sum(-1)))
    xx=x.astype(j.float32);tail=tail+coef[0]+xx@coef[1:d+1]+(xx*xx)@coef[d+1:]
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    # Replace the quantized mass of the observed label approximately with its exact positive-logit channel.
    z=j.logaddexp(tail,j.where(remaining[y]>0,pos,-j.inf))
    if head.size:z=j.logaddexp(z,jax.nn.logsumexp((x@w[head].T).astype(j.float32),-1))
    cheap=z-pos
    if n==0:return (cheap*weight).sum()/weight.sum()
    score=.05+j.sqrt(j.exp(tail-z))+.05*(1-j.minimum(1.,j.exp(pos-z)))
    width=t//n;s=stop(score.reshape(b,n,width));pr=s/s.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));i=j.minimum((u>=j.cumsum(pr,-1)).sum(-1),width-1)
    sel=j.arange(n)[None,:]*width+i;inv=1/j.take_along_axis(pr,i[:,:,None],-1)[:,:,0]
    hidden=j.take_along_axis(x,sel[:,:,None],1);target=j.take_along_axis(y,sel,1)
    logits=(hidden@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    return ((cheap*weight).sum()+((exact-j.take_along_axis(cheap,sel,1))*j.take_along_axis(weight,sel,1)*inv).sum())/weight.sum()

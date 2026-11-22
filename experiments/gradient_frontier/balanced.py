"""Learn a residual-gradient embedding and balance its leading coordinate exactly within each stratum."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk
from .learned_adjoint import ridge


def levels(kind):
    return (4 if kind%2 else 2,)


def features(x,wy,token_w):
    h=x.astype(j.float32);d=h.shape[-1]
    # The observed label is a legitimate covariate before the random quadrature draw.
    interaction=h*wy.astype(j.float32)
    return j.concatenate((j.ones_like(h[...,:1]),h,h*h,interaction,token_w.astype(j.float32)),axis=-1)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    tok=data[0].reshape(-1);y=j.roll(data[0],-1,axis=1).reshape(-1)
    def teacher(h):
        s=(h@w.T).astype(j.float32)
        z=jax.nn.logsumexp(s,-1)
        probability=j.exp(s-z[:,None])
        return z,(probability.astype(j.bfloat16)@w).astype(j.float32)
    z,mu=jax.lax.map(teacher,x.reshape(-1,128,x.shape[-1]));z=z.reshape(-1);mu=mu.reshape(x.shape)
    f=features(x,w[y],w[tok]);coef0=ridge(f,z[:,None],.04)[:,0]
    py=j.exp(j.minimum(0.,j.einsum('id,id->i',x,w[y]).astype(j.float32)-z))
    target=mu-py[:,None]*w[y].astype(j.float32)
    coef=ridge(f,target,.04)
    # Retain the leading actual residual-adjoint directions, then learn them from lightweight token covariates.
    if kind in (122,123,126,127):
        left=target
    else:left=target-.55*(f@coef)
    centered=left-left.mean(0)
    _,basis=j.linalg.eigh(centered.T@centered)
    scores=centered@basis[:,-8:][:,::-1]
    score_coef=ridge(f,scores,.10)
    return coef0,coef.astype(j.bfloat16),score_coef.astype(j.bfloat16)


def balanced_pair(values,key):
    # Select two points on opposite sides of the mean. Their random weights integrate the scalar exactly.
    shape=values.shape
    d=values-values.mean(-1,keepdims=True)
    neg=(d<=0);pos=~neg
    lo=neg.mean(-1);a=j.maximum(-d,0.);b=j.maximum(d,0.)
    u=jax.random.uniform(key,(*shape[:-1],3))
    choose_lo_uniform=u[:,:,0]<lo
    def draw(weights,num):
        total=weights.sum(-1,keepdims=True)
        norm=j.where(total>0,weights/j.maximum(total,1e-12),1./shape[-1])
        return j.minimum((u[:,:,num,None]>=j.cumsum(norm,-1)).sum(-1),shape[-1]-1)
    i=draw(j.where(choose_lo_uniform[:,:,None],neg,a),1)
    jj=draw(j.where(choose_lo_uniform[:,:,None],b,pos),2)
    ai=j.take_along_axis(a,i[:,:,None],-1)[:,:,0];bj=j.take_along_axis(b,jj[:,:,None],-1)[:,:,0]
    mix=j.where(ai+bj>0,bj/j.maximum(ai+bj,1e-12),.5)
    return j.stack((i,jj),-1),j.stack((mix,1-mix),-1)


def objective(x,w,tok,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);zcoef,adjcoef,scoef=bank
    b,t,d=x.shape;N=b*t;stop=jax.lax.stop_gradient
    f=stop(features(x,w[y],w[tok]))
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    py=stop(j.exp(j.minimum(0.,positive-f@zcoef)))
    cheap=(py-1.)*positive
    adj=stop(f.astype(j.bfloat16)@adjcoef)
    cheap=cheap+j.einsum('btd,btd->bt',x,adj).astype(j.float32)
    cheap=cheap-stop(cheap)
    score=stop((f.astype(j.bfloat16)@scoef).astype(j.float32).reshape(N,8))
    score=(score-score.mean(0))/j.maximum(score.std(0),1e-4)
    if kind in (120,121,122,123):
        # Morton ordering of the other learned coordinates groups similar contexts across the whole minibatch.
        quant=j.clip(j.floor(4*(score[:,1:6]+2)),0,15).astype(j.int32)
        ordercode=j.zeros(N,j.int32)
        for bit in range(4):
            ordercode=ordercode+j.sum(((quant>>bit)&1)<<(5*bit+j.arange(5)),axis=-1)
        order=j.argsort(ordercode)
    else:order=j.argsort(score[:,1])
    width=2*(t//n);ng=N//width
    groups=order.reshape(1,ng,width)
    at,weights=balanced_pair(score[order,0].reshape(1,ng,width),key)
    idx=j.take_along_axis(groups,at,-1).reshape(-1)
    weights=weights.reshape(-1)*width
    h=x.reshape(-1,d)[idx];target=y.reshape(-1)[idx]
    logits=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,None],-1)[:,0]
    delta=(exact-cheap.reshape(-1)[idx])*wt.reshape(-1)[idx]*weights
    return ((wt*cheap).sum()+delta.sum())/wt.sum()

"""Distill the two gradient factors at the LM readout; all exact evaluations use the full vocabulary."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def features(x,anchors):
    h=jax.nn.standardize(x.astype(j.float32),axis=-1)
    sim=(h.astype(j.bfloat16)@anchors.T).astype(j.float32)/h.shape[-1]
    local=jax.nn.softmax(10*sim,axis=-1)*j.sqrt(float(anchors.shape[0]))
    return j.concatenate((j.ones_like(h[...,:1]),h,local),axis=-1)


def ridge(phi,targets,reg):
    scale=j.sqrt(j.maximum(j.mean(phi*phi,axis=0),1e-4))
    design=phi/scale
    cov=design.T@design+reg*len(phi)*j.eye(phi.shape[-1])
    mapping=j.linalg.solve(cov,design.T)/scale[:,None]
    return ((mapping.astype(j.bfloat16)@targets.astype(j.bfloat16)).astype(j.float32)
            if targets.shape[-1]>2048 else mapping@targets)


def levels(kind):
    return (8 if kind in (92,93,96,97) else 2,
            kind in (91,93,95,97),kind in (94,95,96,97))


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    token=data[0].reshape(-1)
    target=j.roll(data[0],-1,axis=1).reshape(-1)
    rows=j.arange(len(x))
    standardized=jax.nn.standardize(x.astype(j.float32),axis=-1)
    chosen=jax.random.permutation(jax.random.PRNGKey(17837),len(x))[:k]
    anchors=standardized[chosen].astype(j.bfloat16)
    for _ in range(2):
        labels=j.argmax(standardized.astype(j.bfloat16)@anchors.T,axis=-1)
        sums=jax.ops.segment_sum(standardized,labels,k)
        anchors=jax.nn.standardize(j.where(j.sum(sums*sums,-1,keepdims=True)>0,sums,anchors),axis=-1).astype(j.bfloat16)
    phi=features(x,anchors)
    logits=(x@w.T).astype(j.float32)
    z=jax.nn.logsumexp(logits,-1)
    prob=j.exp(logits-z[:,None])
    yprob=prob[rows,target];cprob=j.where(token!=target,prob[rows,token],0.)
    residual=prob.at[rows,target].set(0.).at[rows,token].set(0.)
    rem=j.maximum(1-yprob-cprob,1e-4)
    # A conditional mean of the remaining distribution is a signed linear gradient predictor.
    # Its low-rank factors produce a full readout-matrix gradient without allocating token-by-vocab scores.
    design=phi*rem[:,None]
    coef=ridge(design,residual,.08)
    hidden=(residual.astype(j.bfloat16)@w).astype(j.float32)
    hidden_coef=ridge(design,hidden,.04)
    zcoef=ridge(phi,z[:,None],.015)
    rough=design@hidden_coef
    hidden_error=j.sqrt(j.sum((rough-hidden)**2,axis=-1)+1e-5)
    readout_error=j.sqrt(j.sum(residual**2,axis=-1)+1e-6)
    difficulty=j.log(.01+hidden_error+readout_error)
    score_coef=ridge(phi,difficulty[:,None],.05)
    return anchors,coef.astype(j.bfloat16),hidden_coef.astype(j.bfloat16),zcoef,score_coef


def objective(x,w,tokens,y,weight,key,config,bank):
    code,n=config
    kind,k=divmod(code,100000)
    _,learned_score,hidden_only=levels(kind)
    anchors,coef,hcoef,zcoef,scoef=bank
    b,t,d=x.shape
    stop=jax.lax.stop_gradient
    phi=stop(features(x,anchors))
    pred=(phi@zcoef)[:,:,0]
    target_w=w[y];copy_w=w[tokens]
    target_logit=j.einsum('btd,btd->bt',x,target_w).astype(j.float32)
    copy_logit=j.einsum('btd,btd->bt',x,copy_w).astype(j.float32)
    unique=(tokens!=y)
    lower=j.logaddexp(target_logit,j.where(unique,copy_logit,-j.inf))
    z=stop(j.maximum(pred,lower))
    py=stop(j.exp(target_logit-z))
    pc=stop(j.where(unique,j.exp(copy_logit-z),0.))
    mass=stop(j.maximum(0.,1.-py-pc))
    design=(phi*mass[:,:,None]).astype(j.bfloat16)
    remaining_hidden=stop(design@hcoef)
    cheap=(py-1.)*target_logit+pc*copy_logit
    cheap=cheap+j.einsum('btd,btd->bt',x,remaining_hidden).astype(j.float32)
    if not hidden_only:
        compressed_w=coef@w
        predicted=design@compressed_w
        cheap=cheap+j.einsum('btd,btd->bt',stop(x),predicted).astype(j.float32)
    # This scalar only specifies a control vector through its VJP.
    cheap=cheap-stop(cheap)
    if n==0:return (cheap*weight).sum()/weight.sum()
    score=(.12+j.exp(j.clip((phi@scoef)[:,:,0],-4.,1.))*j.sqrt(.15+mass)
           if learned_score else .15+j.sqrt(mass))
    width=t//n
    s=stop(score.reshape(b,n,width));prob=s/s.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(prob,-1)).sum(-1),width-1)
    selected=j.arange(n)[None,:]*width+at
    inverse=1/j.take_along_axis(prob,at[:,:,None],-1)[:,:,0]
    hidden=j.take_along_axis(x,selected[:,:,None],1)
    true_logits=(hidden@w.T).astype(j.float32)
    target=j.take_along_axis(y,selected,1)
    full=jax.nn.logsumexp(true_logits,-1)-j.take_along_axis(true_logits,target[:,:,None],-1)[:,:,0]
    corrections=(full-j.take_along_axis(cheap,selected,1))*j.take_along_axis(weight,selected,1)*inverse
    return ((weight*cheap).sum()+corrections.sum())/weight.sum()

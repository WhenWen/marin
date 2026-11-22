"""Nonparametric prediction of token gradient factors from previous full-vocabulary teacher outputs."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (8 if kind%4>=2 else 4,)


def standardized(a):
    return jax.nn.standardize(a.astype(j.float32),axis=-1).astype(j.bfloat16)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    norm=standardized(x)
    centers=norm[jax.random.permutation(jax.random.PRNGKey(711),len(x))[:k]]
    for _ in range(4):
        label=j.argmax(norm@centers.T,axis=-1)
        sums=jax.ops.segment_sum(x.astype(j.float32),label,k)
        centers=standardized(j.where(j.sum(sums**2,-1,keepdims=True)>0,sums,centers))
    label=j.argmax(norm@centers.T,axis=-1)
    # Each teacher query covers all classes. The learned tables contain conditional gradient targets.
    def block(xx,ll):
        logits=(xx@w.T).astype(j.float32)
        z=jax.nn.logsumexp(logits,-1)
        p=j.exp(logits-z[:,None])
        return (jax.ops.segment_sum(p,ll,k),jax.ops.segment_sum(z,ll,k),jax.ops.segment_sum(p.astype(j.bfloat16)@w,ll,k).astype(j.float32))
    args=(x.reshape(-1,256,x.shape[-1]),label.reshape(-1,256))
    def accumulate(total,arr):
        return jax.tree.map(j.add,total,block(*arr)),None
    zeros=(j.zeros((k,w.shape[0]),j.float32),j.zeros(k),j.zeros((k,w.shape[-1]),j.float32))
    (prob,z,adj),_=jax.lax.scan(accumulate,zeros,args)
    count=j.maximum(1,j.bincount(label,length=k))
    prob=prob/count[:,None];near=jax.lax.top_k(prob,16)[1]
    # Fit log partition from the hidden vector and its squared coordinates.
    f=j.concatenate((j.ones((len(x),1)),x.astype(j.float32),j.square(x.astype(j.float32))),axis=-1)
    def exact_z(xx):return jax.nn.logsumexp((xx@w.T).astype(j.float32),axis=-1)
    target=jax.lax.map(exact_z,args[0]).reshape(-1)
    scale=j.maximum(j.sqrt(j.mean(f*f,axis=0)),1e-4);ff=f/scale
    coef=j.linalg.solve(ff.T@ff+.03*len(x)*j.eye(f.shape[1]),ff.T@target)/scale
    return centers,near,prob.astype(j.bfloat16),(adj/count[:,None]).astype(j.bfloat16),coef


def objective(x,w,tokens,y,weight,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);centers,labels,prior,adj,zcoef=bank
    stop=jax.lax.stop_gradient;b,t,d=x.shape
    assignment=j.argmax(standardized(x)@centers.T,axis=-1)
    # Predict a small set of influential readout coordinates from the teacher's labeled neighborhoods.
    picked=labels[assignment]
    ids=j.concatenate((y[:,:,None],tokens[:,:,None],picked),axis=-1)
    size=ids.shape[-1]
    first=~j.any((ids[:,:,:,None]==ids[:,:,None,:])&j.tril(j.ones((size,size),bool),-1),axis=-1)
    ws=w[ids];local=j.einsum('btd,btsd->bts',x,ws).astype(j.float32)
    af=x.astype(j.float32);pred=zcoef[0]+af@zcoef[1:d+1]+(af*af)@zcoef[d+1:]
    # Independent binary probability predictions use the learned full-vocabulary log partition.
    # Clamp their total and fill the remaining gradient from the teacher's empirical conditional mean.
    p=j.where(first,j.exp(j.minimum(0.,local-pred[:,:,None])),0.)
    p=stop(p/j.maximum(1.,p.sum(-1,keepdims=True)));q=prior[assignment[:,:,None],ids].astype(j.float32)*first
    rem=stop((1-p.sum(-1))/j.maximum(.05,1-q.sum(-1)))
    if kind>=104:
        gs=(p-rem[:,:,None]*q)
        prediction=(jax.nn.one_hot(assignment,k,dtype=j.bfloat16)@(prior@w))
        # Only differentiate the readout matrix on this branch.
        cheap=rem*j.einsum('btd,btd->bt',stop(x),prediction)
    else:
        gs=p
        cheap=j.zeros_like(pred)
    hidden=adj[assignment].astype(j.float32)-j.einsum('bts,btsd->btd',q,ws.astype(j.float32))
    cheap=cheap+rem*j.einsum('btd,btd->bt',x,stop(hidden))
    # When the prior readout is present, only its readout derivative needs subtracting at listed coordinates.
    cheap=cheap+(gs*local).sum(-1)-local[:,:,0]
    if kind>=104:
        cheap=cheap+rem*j.einsum('bts,btd,btsd->bt',q,x,stop(ws))
    cheap=cheap-stop(cheap)
    score=(.08+j.sqrt(j.maximum(0.,1-p.sum(-1)))+.1*(1-p.max(-1))) if kind%2 else j.ones_like(pred)
    width=t//n;score=stop(score.reshape(b,n,width));pr=score/score.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(pr,-1)).sum(-1),width-1)
    sel=j.arange(n)[None,:]*width+at;inv=1/j.take_along_axis(pr,at[:,:,None],-1)[:,:,0]
    h=j.take_along_axis(x,sel[:,:,None],1);yy=j.take_along_axis(y,sel,1)
    logits=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,yy[:,:,None],-1)[:,:,0]
    delta=(exact-j.take_along_axis(cheap,sel,1))*j.take_along_axis(weight,sel,1)*inv
    return ((weight*cheap).sum()+delta.sum())/weight.sum()

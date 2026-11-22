"""Learn local proposal tables and Rao–Blackwellize short independence-Metropolis chains for a gradient control."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (4,)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda v:v.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])
    h=jax.nn.standardize(x.astype(j.float32),axis=-1)
    centers=h[jax.random.permutation(jax.random.key(1129),len(h))[:k]]
    for _ in range(4):
        lab=j.argmax(h.astype(j.bfloat16)@centers.astype(j.bfloat16).T,axis=1)
        count=j.bincount(lab,length=k)
        centers=j.where(count[:,None]>0,jax.ops.segment_sum(h,lab,k)/j.maximum(count[:,None],1.),centers)
        centers=jax.nn.standardize(centers,axis=-1)
    # Average actual conditional softmax distributions to learn proposals in occupied regions.
    # Chunking bounds memory without restricting the support of the teacher.
    def teacher(carry,batch):
        hh,ll=batch
        pr=jax.nn.softmax((hh@w.T).astype(j.float32),axis=-1)
        return carry+jax.ops.segment_sum(pr,ll,k),None
    lab=j.argmax(h.astype(j.bfloat16)@centers.astype(j.bfloat16).T,axis=1)
    mass,_=jax.lax.scan(teacher,j.zeros((k,len(w)),j.float32),(x.reshape(-1,64,x.shape[1]),lab.reshape(-1,64)))
    vals,indices=jax.lax.top_k(mass,128)
    probabilities=(vals+1e-9)/(vals+1e-9).sum(axis=1,keepdims=True)
    ticks=(j.arange(256)+.5)/256
    slots=jax.vmap(lambda cdf:j.searchsorted(cdf,ticks,side='left'))(j.cumsum(probabilities,axis=1))
    table=j.take_along_axis(indices,j.minimum(slots,127),axis=1)
    counts=jax.vmap(lambda ids:j.bincount(ids,length=len(w)))(table)
    return centers.astype(j.bfloat16),table,counts.astype(j.uint16)


def control(x,w,y,bank,key,kind):
    centers,table,counts=bank
    rounds=4 if kind%2==0 else 8
    paths=4 if kind%4<2 else 8
    h=jax.nn.standardize(x.astype(j.float32),axis=-1).astype(j.bfloat16)
    region=j.argmax(h@centers.T,axis=-1)
    u=jax.random.uniform(key,(*y.shape,paths,rounds+1,3))
    group=region[:,:,None,None];target=y[:,:,None,None]
    mixture=u[...,0];draw=u[...,1];V=w.shape[0];normal=j.log1p(float(V))
    logid=j.minimum(j.expm1(normal*draw).astype(j.int32),V-1)
    learned=table[group,j.minimum((256*draw).astype(j.int32),255)]
    ids=j.where(mixture<.20,target,j.where(mixture<.65,learned,logid))
    prob=.20*(ids==target)+.45*counts[group,ids].astype(j.float32)/256+.35*j.log1p(1./(1.+ids))/normal
    scores=j.einsum('btd,btpsd->btps',x,w[ids]).astype(j.float32)
    logweight=jax.lax.stop_gradient(scores-j.log(prob))
    # The final proposed transition is integrated analytically; earlier transitions use actual draws.
    value=scores[...,0];logratio=logweight[...,0];acc_mean=0.
    for i in range(1,rounds+1):
        accept=j.exp(j.minimum(0.,logweight[...,i]-logratio))
        conditional=(1.-accept)*value+accept*scores[...,i]
        if i>=rounds//2:acc_mean=acc_mean+conditional
        take=u[...,i,2]<accept
        value=j.where(take,scores[...,i],value)
        logratio=j.where(take,logweight[...,i],logratio)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    control_loss=(acc_mean/(rounds-rounds//2+1)).mean(axis=-1)-positive
    return control_loss,jax.lax.stop_gradient(j.std(logweight,axis=(-1,-2)))


def objective(x,w,y,weight,key,config,bank):
    code,n=config;kind=code//100000;b,t,d=x.shape
    proposal_key,position_key=jax.random.split(key)
    cheap,spread=control(x,w,y,bank,proposal_key,kind)
    width=t//n
    p=j.clip(.5+spread,.5,6.).reshape(b,n,width) if kind<254 else j.ones((b,n,width))
    p=p/p.sum(axis=-1,keepdims=True)
    uniform=jax.random.uniform(position_key,(b,n,1))
    loc=j.minimum((uniform>=j.cumsum(p,-1)).sum(axis=-1),width-1)
    ix=j.arange(n)[None,:]*width+loc
    inv=1/j.take_along_axis(p,loc[:,:,None],axis=-1)[:,:,0]
    chosen=j.take_along_axis(x,ix[:,:,None],axis=1);label=j.take_along_axis(y,ix,axis=1)
    logits=(chosen@w.T).astype(j.float32)
    gold=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,label[:,:,None],axis=-1)[:,:,0]
    residual=(gold-j.take_along_axis(cheap,ix,1))*j.take_along_axis(weight,ix,1)*inv
    return ((cheap*weight).sum()+residual.sum())/weight.sum()

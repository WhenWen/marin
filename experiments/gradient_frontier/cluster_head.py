"""Route true forward hidden states to cached, context-dependent vocabulary shortlists."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk

def dimensions(kind):
    return {40:(64,512,32),41:(64,1024,32),42:(128,1024,32),
            43:(128,2048,32),44:(128,1024,64),45:(64,2048,64),46:(128,2048,16)}[kind]

@eqx.filter_jit
def fit(model,data,k,kind):
    groups,extra,_=dimensions(kind)
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    h=trunk(m,data).reshape((-1,128)).astype(j.float32)
    norms=jax.nn.standardize(h,axis=-1)
    indices=jax.random.permutation(jax.random.PRNGKey(13719),len(h))[:groups]
    anchors=norms[indices]
    def improve(_,mu):
        labels=j.argmax(norms@mu.T,axis=-1)
        sums=jax.ops.segment_sum(norms,labels,groups)
        mu=j.where((sums**2).sum(-1,keepdims=True)>0,sums,mu)
        return jax.nn.standardize(mu,axis=-1)
    anchors=jax.lax.fori_loop(0,3,improve,anchors)
    labels=j.argmax(norms@anchors.T,axis=-1)
    count=j.bincount(labels,length=groups)
    mean=jax.ops.segment_sum(h,labels,groups)/j.maximum(1,count[:,None])
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    scores=mean.astype(j.bfloat16)@w[k:].T
    ids=j.sort(jax.lax.top_k(scores,extra)[1]+k,axis=1)
    return anchors.astype(j.bfloat16),ids

def objective(x,w,y,wt,key,config,bank):
    code,n=config;k=code%100000;kind=code//100000
    _,extra,block=dimensions(kind);b,t,d=x.shape
    centers,dicts=bank
    flat=x.reshape((-1,d))
    sim=jax.nn.standardize(flat.astype(j.float32),axis=-1).astype(j.bfloat16)@centers.T
    route=j.argmax(sim,axis=-1);ix=j.argsort(route,stable=True)
    order=ix.reshape((-1,block))
    groups=route[order[:,block//2]]
    words=dicts[groups];xx=flat[order]
    tail=(xx@j.swapaxes(w[words],1,2)).astype(j.float32)
    ygroup=y.reshape(-1)[order]
    hit=jax.vmap(lambda row,labels:j.searchsorted(row,labels,method='scan_unrolled'))(words,ygroup)
    hit=j.take_along_axis(words,j.minimum(extra-1,hit),axis=1)==ygroup
    def unsort(a):return j.zeros(b*t,a.dtype).at[ix].set(a.reshape(-1)).reshape((b,t))
    extra_z=unsort(jax.nn.logsumexp(tail,-1));inside=unsort(hit)|(y<k)
    head=(x@w[:k].T).astype(j.float32);common=jax.nn.logsumexp(head,-1)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z=j.logaddexp(j.logaddexp(common,extra_z),j.where(inside,-1e9,positive))
    top=j.maximum(j.maximum(j.max(head,-1),unsort(j.max(tail,-1))),positive)
    score=.035+j.sqrt(j.minimum(1.,j.exp(extra_z-z)))+.15*(1.-j.minimum(1.,j.exp(top-z)))
    width=t//n
    s=jax.lax.stop_gradient(score.reshape((b,n,width)));p=s/s.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));col=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    pos=j.arange(n)[None,:]*width+col;iw=1./j.take_along_axis(p,col[:,:,None],-1)[:,:,0]
    chosen=j.take_along_axis(x,pos[:,:,None],1)
    true_z=jax.nn.logsumexp((chosen@w.T).astype(j.float32),-1)
    delta=(true_z-j.take_along_axis(z,pos,1))*j.take_along_axis(wt,pos,1)*iw
    return (((z-positive)*wt).sum()+delta.sum())/wt.sum()

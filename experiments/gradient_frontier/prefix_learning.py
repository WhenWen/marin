"""Learn minimum-variance weights for the nested panels revealed by exact causal prefixes."""
import equinox as eqx
import haliax as h
import jax
import jax.numpy as j
from levanter.layers.attention_mask import AttentionMask
from .tail import trunk
from .prefix_panels import layout
from .lowmem import objective as short_head


def uniform(K):
    mask=j.arange(K)[:,None]>=j.arange(K)[None,:]
    return mask*K/(K-j.arange(K))[None,:]


def calibrating_losses(model,data,K):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=trunk(m,data)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(data[0],-1,1)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    head=(x@w[:1024].T).astype(j.float32)
    proxy=j.logaddexp(jax.nn.logsumexp(head,axis=-1),j.where(y>=1024,pos,-j.inf))-pos
    return (proxy*data[1]).reshape(len(y),K,-1).mean(-1)


@eqx.filter_jit
def fit(model,data,K):
    dynamic,static=eqx.partition(model,eqx.is_inexact_array)
    def fn(x):return calibrating_losses(eqx.combine(x,static),data,K)
    leaves,td=jax.tree.flatten(dynamic)
    def one(probe):
        keys=jax.random.split(jax.random.fold_in(jax.random.key(5392),probe),len(leaves))
        # Symmetric perturbations work with the production attention kernel's reverse-mode rule.
        directions=[jax.random.rademacher(key,a.shape,dtype=a.dtype)*.0015 if a.size<2_000_000 else None
                    for a,key in zip(leaves,keys)]
        def evaluate(sign):
            return fn(jax.tree.unflatten(td,[a if d is None else a+sign*d for a,d in zip(leaves,directions)]))
        projection=(evaluate(1)-evaluate(-1))/.003
        centered=projection-projection.mean(0)
        return centered.T@centered
    cov=jax.lax.fori_loop(0,4,lambda i,s:s+one(i),j.zeros((K,K)))
    scale=j.maximum(1e-8,j.trace(cov)/K)
    cov=.65*cov+.35*j.diag(j.diag(cov))+.07*scale*j.eye(K)
    inverse=[]
    for cutoff in range(1,K+1):
        inv=j.linalg.solve(cov[:cutoff,:cutoff],j.eye(cutoff))
        inverse.append(j.pad(inv,((0,K-cutoff),(0,K-cutoff))))
    inverse=j.stack(inverse)
    lag=j.linalg.solve(inverse.mean(0),j.ones(K))
    learned=j.einsum('kij,j->ki',inverse,lag)
    return .7*learned+.3*uniform(K)


def loss(model,data,key,K,k,n,bank):
    tokens,wt,segments=data;B,T=tokens.shape
    left,right=jax.random.split(key);order=jax.random.permutation(left,B)
    ds,pos,parts,_,_=map(j.asarray,layout(B,T,K));ids=order[ds]
    shape=(h.Axis('batch',len(ds)),h.Axis('position',T));m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=m.embeddings.embed(h.named(tokens[ids,pos],shape))
    mask=AttentionMask.causal().with_segment_ids(h.named(segments[ids,pos]+parts,shape));coordinates=h.named(pos,shape)
    for layer in m.transformer.layers.unstacked():x=layer(x,mask,pos_ids=coordinates)
    x=m.transformer.norm(x).rearrange(('batch','position','embed')).array
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=tokens[ids,(pos+1)%T]
    weights=wt[ids,pos]*bank[ds//(B//K),pos//(T//K)]
    if k:
        val=short_head(x,w,y,weights,right,(6100000+k,n))
        return val*weights.sum()/wt.sum()
    scores=(x@w.T).astype(j.float32)
    ce=jax.nn.logsumexp(scores,axis=-1)-j.take_along_axis(scores,y[:,:,None],axis=-1)[:,:,0]
    return (ce*weights).sum()/wt.sum()


@eqx.filter_jit
def update(state,data,key,K,k,n,bank):
    g=eqx.filter_grad(loss)(state.model,data,key,K,k,n,bank)
    return state.take_step(g,key=key)[0]


@eqx.filter_jit
def measure(state,a,b,key,K,k,n,bank):
    ka,kb=jax.random.split(key)
    g=eqx.filter_grad(loss)(state.model,a,ka,K,k,n,bank);h=eqx.filter_grad(loss)(state.model,b,kb,K,k,n,bank)
    return .5*sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(g),jax.tree.leaves(h)))

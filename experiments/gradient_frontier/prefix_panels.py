"""Pack complementary exact causal prefixes and estimate a full sequence gradient from probability-sampled target panels."""
import functools
import equinox as eqx
import haliax as h
import jax
import jax.numpy as j
import numpy as np
from levanter.layers.attention_mask import AttentionMask
from .lowmem import objective as short_head


@functools.lru_cache(None)
def layout(B,T,K):
    assert B%(2*K)==0 and T%K==0
    width=T//K;group=B//K
    phase=lambda length:list(range((length-1)*group,length*group))
    packed=[]
    for length in range(1,(K+1)//2):
        packed.extend(zip(phase(length),phase(K-length)))
    if not K%2:
        half=phase(K//2);packed.extend(zip(half[::2],half[1::2]))
    packed.extend((a,) for a in phase(K))
    take=np.empty((len(packed),T),np.int32);pos=np.empty_like(take);part=np.empty_like(take)
    active=np.empty((B,width),np.int32);orig=np.empty_like(active)
    for row,docs in enumerate(packed):
        begin=0
        for li,doc in enumerate(docs):
            length=(doc//group+1)*width
            span=slice(begin,begin+length);alpha=np.arange(length)
            take[row,span]=doc;pos[row,span]=alpha;part[row,span]=li*T
            active[doc]=row*T+begin+length-width+np.arange(width)
            orig[doc]=length-width+np.arange(width)
            begin+=length
        assert begin==T
    assert len(np.unique(active))==B*width
    return take,pos,part,active,orig


def loss(model,data,key,K,k,n,phase=None):
    tokens,wt,segments=data;B,T=tokens.shape
    orderkey,selectkey=jax.random.split(key)
    order=jax.random.permutation(orderkey,B)
    if phase is not None:order=j.roll(order,phase*(B//K))
    ds,ps,parts,targets,orig=map(j.asarray,layout(B,T,K))
    ids=order[ds]
    shape=(h.Axis('batch',len(ds)),h.Axis('position',T))
    mm=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    x=mm.embeddings.embed(h.named(tokens[ids,ps],shape))
    mask=AttentionMask.causal().with_segment_ids(h.named(segments[ids,ps]+parts,shape))
    original_pos=h.named(ps,shape)
    for layer in mm.transformer.layers.unstacked():x=layer(x,mask,pos_ids=original_pos)
    x=mm.transformer.norm(x).rearrange(('batch','position','embed')).array
    x=x.reshape(-1,x.shape[-1])[targets]
    w=mm.get_lm_head().rearrange(('vocab','embed')).array
    y=tokens[order[:,None],(orig+1)%T];weights=wt[order[:,None],orig]
    if k:
        scalar=short_head(x,w,y,weights,selectkey,(6100000+k,n))
        return scalar*weights.sum()*K/wt.sum()
    logits=(x@w.T).astype(j.float32)
    ce=jax.nn.logsumexp(logits,axis=-1)-j.take_along_axis(logits,y[:,:,None],axis=-1)[:,:,0]
    return (weights*ce).sum()*K/wt.sum()


@eqx.filter_jit
def update(state,data,key,K,k,n):
    g=eqx.filter_grad(loss)(state.model,data,key,K,k,n)
    return state.take_step(g,key=key)[0]


@eqx.filter_jit
def measure(state,a,b,key,K,k,n):
    ka,kb=jax.random.split(key)
    g=eqx.filter_grad(loss)(state.model,a,ka,K,k,n);h2=eqx.filter_grad(loss)(state.model,b,kb,K,k,n)
    return .5*sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(g),jax.tree.leaves(h2)))

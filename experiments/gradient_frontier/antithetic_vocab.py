"""Exactly telescope the softmax normalizer over balanced vocabulary shards.

Each binary merge corrects the average of the two children's log normalizers.
The merge discrepancy is quadratic near zero child imbalance: log cosh((a-b)/2).
"""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (2,)


def settings(kind):
    leaves=64 if kind in (180,181,184,185) else 128
    learned=kind in (181,183,185,187)
    return leaves,learned


@eqx.filter_jit
def fit(model,data,k,kind):
    L,learned=settings(kind)
    w=model.get_lm_head().rearrange(('vocab','embed')).array
    order=j.arange(len(w))
    if learned:
        m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
        x=trunk(m,data).reshape(-1,w.shape[1])[::8]
        prob=jax.nn.softmax((x@w.astype(j.bfloat16).T).astype(j.float32),axis=1)
        order=j.argsort(-j.mean(prob*prob,axis=0))
    head=order[:k];tail=order[k:]
    # Distribute high-impact rows across the leaves. All tails have the same padded width.
    pad=(-len(tail))%L
    leaves=j.pad(tail,(0,pad),constant_values=len(w)).reshape(-1,L).T
    in_tail=j.zeros(len(w),bool).at[tail].set(True)
    return head,leaves,in_tail


def discrepancy(a,b):
    return j.logaddexp(a,b)-j.log(2.)-.5*(a+b)


def objective(x,w,y,weight,key,config,bank):
    code,n=config
    kind,k=divmod(code,100000)
    L,learned=settings(kind);depth=L.bit_length()-1
    ids,leaves,tail=bank;b,t,d=x.shape
    head=(x@(w[ids].T if learned else w[:k].T)).astype(j.float32)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    anchor=j.logaddexp(jax.nn.logsumexp(head,-1),j.where(tail[y],pos,-j.inf))
    flat=leaves.reshape(-1);leafsize=leaves.shape[1]
    def take(values,indices):
        if values.ndim==3:return j.take_along_axis(values,indices[:,:,None],axis=1)
        return j.take_along_axis(values,indices,axis=1)
    def values(h,target,a,node,merge,block):
        """One uniformly drawn vocabulary node per position block; sibling sums stay correlated."""
        docs,positions,_=h.shape;span=min(block,positions)
        groups=docs*positions//span
        # node is a vector of group-specific choices. Padding uses the one nonexistent token ID.
        width=leafsize*merge
        rows=flat[node[:,None]*width+j.arange(width)]
        xx=h.reshape(groups,span,d);yy=target.reshape(groups,span)
        aa=a.reshape(groups,span)
        logits=j.einsum('gtd,gkd->gtk',xx,w[j.minimum(rows,len(w)-1)]).astype(j.float32)
        logits=j.where((rows[:,None,:]!=yy[:,:,None])&(rows[:,None,:]<len(w)),logits,-j.inf)
        if merge==1:
            out=j.logaddexp(aa,jax.nn.logsumexp(logits,-1)+j.log(float(L)))
        else:
            half=width//2;scale=j.log(float(2*L/merge))
            left=j.logaddexp(aa,jax.nn.logsumexp(logits[:,:,:half],-1)+scale)
            right=j.logaddexp(aa,jax.nn.logsumexp(logits[:,:,half:],-1)+scale)
            out=discrepancy(left,right)
        return out.reshape(docs,positions)
    block=128 if kind<184 else 32
    keys=jax.random.split(key,2*depth+1)
    nodes=jax.random.randint(keys[0],(b*t//block,),0,L)
    base=values(x,y,anchor,nodes,1,block)-pos
    total=j.sum(weight*base)
    for level in range(1,depth+1):
        merge=2**level
        # Contiguous position strata preserve every original token's positive selection probability.
        count=min(t,n*2**((depth-level+1)//2))
        stride=t//count
        indices=j.arange(count)[None,:]*stride+jax.random.randint(keys[2*level-1],(b,count),0,stride)
        groupwidth=min(block,count)
        nodes=jax.random.randint(keys[2*level],(b*count//groupwidth,),0,L//merge)
        delta=values(take(x,indices),take(y,indices),take(anchor,indices),nodes,merge,block)
        total=total+stride*j.sum(take(weight,indices)*delta)
    return total/weight.sum()

"""Evaluate antithetic vocabulary corrections with one disjoint vocabulary partition per grouped GEMM."""
import equinox as eqx
import jax
import jax.numpy as j
from . import antithetic_vocab as tree


def levels(kind):
    return tree.levels(kind-30)


@eqx.filter_jit
def fit(model,data,k,kind):
    ids,leaves,tail=tree.fit(model,data,k,kind-30)
    order=j.concatenate((ids,leaves.reshape(-1)))
    inv=j.zeros(len(tail)+1,j.int32).at[order].set(j.arange(len(order)))
    return order,inv[:-1],tail


@jax.custom_vjp
def reorder(w,order,inv):
    return j.where((order<len(w))[:,None],w[j.minimum(order,len(w)-1)],0.)


def fwd(w,order,inv):
    return reorder(w,order,inv),inv


def bwd(inv,dy):
    return dy[inv],None,None


reorder.defvjp(fwd,bwd)


def values(x,y,anchor,ordered,ids,vocab,key,L,merge):
    nodes=L//merge
    h=x.reshape(-1,x.shape[-1]);a=anchor.reshape(-1);labels=y.reshape(-1)
    pad=(-len(h))%nodes
    # Each short chunk visits every vocabulary node exactly once. Independent cyclic rotations
    # make each token's marginal uniform, and sorting activations avoids duplicating any vocabulary rows.
    shape=(-1,nodes)
    rows=j.arange((len(h)+pad)//nodes)[:,None]
    offset=jax.random.randint(key,(len(rows),1),0,nodes)
    cols=j.arange(nodes)[None,:]
    index=(cols+offset)%nodes;undo=(cols-offset)%nodes
    H=j.pad(h,((0,pad),(0,0))).reshape(*shape,h.shape[-1])[rows,index].swapaxes(0,1)
    A=j.pad(a,(0,pad)).reshape(shape)[rows,index].T
    Y=j.pad(labels,(0,pad)).reshape(shape)[rows,index].T
    table=ordered.reshape(nodes,-1,h.shape[-1]);words=ids.reshape(nodes,-1)
    scores=j.einsum('nrd,nvd->nrv',H,table).astype(j.float32)
    scores=j.where((Y[:,:,None]!=words[:,None,:])&(words[:,None,:]<vocab),scores,-j.inf)
    if merge==1:
        v=j.logaddexp(A,jax.nn.logsumexp(scores,-1)+j.log(float(L)))
    else:
        half=scores.shape[-1]//2;scale=j.log(float(2*nodes))
        v=tree.discrepancy(j.logaddexp(A,jax.nn.logsumexp(scores[:,:,:half],-1)+scale),
                           j.logaddexp(A,jax.nn.logsumexp(scores[:,:,half:],-1)+scale))
    return v.T[rows,undo].reshape(-1)[:len(h)].reshape(y.shape)


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000)
    L,_=tree.settings(kind-30);depth=L.bit_length()-1
    order,inv,tail=bank;tab=reorder(w,order,inv)
    b,t,d=x.shape
    head=(x@tab[:k].T).astype(j.float32)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    anchor=j.logaddexp(jax.nn.logsumexp(head,-1),j.where(tail[y],pos,-j.inf))
    keys=jax.random.split(key,2*depth+1)
    total=j.sum(wt*(values(x,y,anchor,tab[k:],order[k:],len(w),keys[0],L,1)-pos))
    for level in range(1,depth+1):
        count=min(t,n*2**((depth-level+1)//2))
        stride=t//count
        sel=j.arange(count)[None,:]*stride+jax.random.randint(keys[2*level-1],(b,count),0,stride)
        def take(v):return j.take_along_axis(v,sel[:,:,None] if v.ndim==3 else sel,axis=1)
        delta=values(take(x),take(y),take(anchor),tab[k:],order[k:],len(w),keys[2*level],L,2**level)
        total+=stride*j.sum(take(wt)*delta)
    return total/wt.sum()

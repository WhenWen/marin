"""Sample attention-isolated packed segments with exact finite-family correction to native B32."""
import functools

import equinox as eqx
import jax
import jax.numpy as j
import numpy as np

from .fifty_crossover import native_call
from .lowmem import objective
from .subspace_forecast import native_audit
from .tail import trunk

B=32
PHASES=16


def plan(segments,cut,batch,rng,balanced=False):
    """All segment descriptors fit into fixed-width groups, including a final zero-padded group."""
    groups=[[],[]]
    for i,s in enumerate(segments):
        ends=np.r_[0,np.flatnonzero(s[:-1]!=s[1:])+1,len(s)]
        for a,b in zip(ends[:-1],ends[1:]):
            groups[int(b-a>cut)].append((i,a,b-a))
    assignments=[]
    for rows,length in zip(groups,(cut,1024)):
        if not rows:
            assignments.append(None)
            continue
        count=(len(rows)+batch-1)//batch
        padded=np.zeros((count*batch,3),np.int32)
        shuffled=rng.permutation(np.asarray(rows,np.int32))
        if balanced:
            # Every row appears exactly once. Spread long segments over groups before querying one group.
            by_length=shuffled[np.argsort(-shuffled[:,2],kind='stable')]
            striped=np.zeros((batch,count,3),np.int32)
            striped.reshape(-1,3)[:len(rows)]=by_length
            striped[1::2]=striped[1::2,::-1].copy()
            padded=striped.transpose(1,0,2).reshape(-1,3)
        else:
            padded[:len(rows)]=shuffled
        offsets=rng.integers(PHASES,size=(count,batch,length//PHASES),dtype=np.int32)
        assignments.append((j.asarray(padded.reshape(count,batch,3)),j.asarray(offsets),int(rng.integers(count))))
    return assignments


def piece(model,data,group,offset,phase,length,k):
    ids,begin,size=group.T
    col=j.arange(length)
    at=j.minimum(begin[:,None]+col,1023)
    sid=ids[:,None]
    tokens,weights,_=data
    selected=tokens[sid,at]
    labels=tokens[sid,(at+1)%1024]
    valid=col[None,:]<size[:,None]
    wt=j.where(valid,weights[sid,at],0.)
    # One isolated segment starts at position zero. Padding occurs only after all valid causal targets.
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    h=trunk(m,(selected,wt,j.zeros_like(selected)))
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    sample=(offset+phase)%PHASES
    # A real (non-padding) segment may contain only the masked final position. Stabilize the denominator there.
    total=wt.sum()
    safe=wt.at[0,0].add((total==0).astype(wt.dtype))
    loss=objective(h,w,labels,safe,jax.random.key(0),(6100000+k,length//PHASES),sample)
    return loss*(total/(B*1023.))


@eqx.filter_jit
def raw(model,data,group,offset,phase,length,k):
    return eqx.filter_grad(piece)(model,data,group,offset,phase,length,k)


@eqx.filter_jit
def scale(raw,factor):return jax.tree.map(lambda x:x*factor,raw)


@eqx.filter_jit
def plus(a,b):return jax.tree.map(j.add,a,b)


@eqx.filter_jit
def correct(native,s):
    return jax.tree.map(lambda a,b:a-b/PHASES,native,s)


def draw(model,data,layout,cut,k,phases):
    results=[]
    for length,entry,phase in zip((cut,1024),layout,phases):
        if entry is None:continue
        table,offset,which=entry
        results.append(scale(raw(model,data,table[which],offset[which],phase,length,k),len(table)))
    return results[0] if len(results)==1 else plus(*results)


def correction(model,data,layout,cut,k,phases):
    gold=native_call(native_audit,model,data,B)
    acc=None
    for length,entry in zip((cut,1024),layout):
        if entry is None:continue
        table,offsets,_=entry
        for group,offset in zip(table,offsets):
            for phase in phases:
                v=raw(model,data,group,offset,phase,length,k)
                acc=v if acc is None else plus(acc,v)
    return correct(gold,acc)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    grad,corr=inputs
    if corr is not None:grad=jax.tree.map(lambda a,b:a+b/.002,grad,corr)
    return state.take_step(grad,key=jax.random.split(state.training_key)[1])[0]

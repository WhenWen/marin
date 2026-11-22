"""Pack short and long sampled segments into one masked transformer forward and backward."""
import math

import equinox as eqx
import jax
import jax.numpy as j
import numpy as np

from . import segment_queries as seg
from .fifty_crossover import native_call
from .lowmem import objective
from .subspace_forecast import native_audit
from .tail import trunk

advance=seg.advance


def plan(s,cut,batch,rng,balanced,prefix):
    entries=seg.plan(s,cut,batch,rng,balanced,prefix)
    groups=[];base=[];counts=[]
    for entry,length in zip(entries,(cut,1024)):
        if entry is None:
            groups.append(j.zeros((1,batch,3),j.int32))
            base.append(j.zeros((1,batch,length//16),j.int32))
            counts.append(1)
        else:
            g,o,_=entry;groups.append(g);base.append(o);counts.append(len(g))
    period=math.lcm(*counts)
    tables=tuple(tuple((g[i],o[i]) for i in range(len(g))) for g,o in zip(groups,base))
    return tables,tuple(j.asarray(n,j.float32) for n in counts),period,int(rng.integers(period))


def pack(data,group,off,phase,length,skip):
    ids,begin,size=group.T
    col=j.arange(length)
    at=j.minimum(begin[:,None]+col,1023);sid=ids[:,None]
    x,w,_=data
    toks=x[sid,at];targets=x[sid,(at+1)%1024]
    wt=j.where((col>=skip)&(col<size[:,None]),w[sid,at],0.)
    # In a packed row each short segment has a distinct attention ID.
    identity=j.broadcast_to((j.arange(len(ids))%(1024//length))[:,None],toks.shape)
    return toks.reshape(-1,1024),targets.reshape(-1,1024),wt.reshape(-1,1024),identity.reshape(-1,1024),((off+phase)%16).reshape(-1,64)


def loss(model,data,chosen,counts,phase,cut,k,prefix):
    pieces=[]
    for length,(g,o),n in zip((cut,1024),chosen,counts):
        x,y,weight,segid,off=pack(data,g,o,phase,length,cut if prefix and length==1024 else 0)
        pieces.append((x,y,weight*n,segid,off))
    x,y,wt,ids,offset=(j.concatenate(values,0) for values in zip(*pieces))
    total=wt.sum();safe=wt.at[0,0].add((total==0).astype(wt.dtype))
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    hidden=trunk(m,(x,safe,ids));w=m.get_lm_head().rearrange(('vocab','embed')).array
    value=objective(hidden,w,y,safe,jax.random.key(0),(6100000+k,64),offset)
    return value*(total/(seg.B*1023.))


raw=eqx.filter_jit(eqx.filter_grad(loss))


def draw(model,data,layout,cut,k,phases,prefix):
    tables,counts,_,which=layout
    return raw(model,data,tuple(t[which%len(t)] for t in tables),counts,phases[0],cut,k,prefix)


@eqx.filter_jit
def repair(g,s,n):return jax.tree.map(lambda a,b:a-b/n,g,s)


def correction(model,data,layout,cut,k,phases,prefix):
    tables,counts,period,_=layout
    gold=native_call(native_audit,model,data,32);s=None
    for idx in range(period):
        selected=tuple(t[idx%len(t)] for t in tables)
        for phase in phases:
            v=raw(model,data,selected,counts,phase,cut,k,prefix)
            s=v if s is None else seg.plus(s,v)
    return repair(gold,s,16*period)

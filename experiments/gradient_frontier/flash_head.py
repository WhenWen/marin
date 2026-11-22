"""Compute the cheap log partition and its hidden-state gradient by cross attention."""
from functools import partial
import math

import jax
import jax.numpy as j
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt
from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha


def moments(x, w, split):
    length, dim = x.shape
    groups = w.shape[0] // split
    q = j.broadcast_to(x[None, :, None, :], (groups, length, 1, dim))
    kv = w.reshape(groups, split, 1, dim)
    value, lse2 = mha(q, kv, kv, None, sm_scale=1., causal=False,
                      block_sizes=BlockSizes(block_q=64, block_k=64),
                      return_residuals=True)
    return value[:, :, 0, :], lse2[:, 0, :] * math.log(2.)


def readout_vjp(x, w, lse, up, split_tokens):
    length, dim = x.shape
    chunk = min(split_tokens, length)
    tile, vocab_tile = 64, 32
    groups = length // chunk
    def kernel(xx, ww, zz, scales, output):
        weight = ww[...]
        def add(i, acc):
            span = pl.dslice(i * tile, tile)
            hidden = xx[span, :]
            logits = pt.dot(hidden, weight.T)
            coefficient = j.exp(logits - zz[span, None]) * scales[span, None]
            return acc + pt.dot(coefficient.astype(hidden.dtype).T, hidden)
        output[...] = jax.lax.fori_loop(0, chunk // tile, add, j.zeros((vocab_tile, dim), j.float32))
    partials = pl.pallas_call(
        kernel, grid=(w.shape[0] // vocab_tile, groups),
        in_specs=(pl.BlockSpec((chunk,dim),lambda i,g:(g,0)),
                  pl.BlockSpec((vocab_tile,dim),lambda i,g:(i,0)),
                  pl.BlockSpec((chunk,),lambda i,g:(g,)),
                  pl.BlockSpec((chunk,),lambda i,g:(g,))),
        out_specs=pl.BlockSpec((None,vocab_tile,dim),lambda i,g:(g,i,0)),
        out_shape=jax.ShapeDtypeStruct((groups,*w.shape),j.float32),
        compiler_params=pt.CompilerParams(num_warps=4), name='vocab_weight_cotangent'
    )(x,w,lse,up)
    return partials.sum(0).astype(w.dtype)


def compute(x,w,variant):
    alpha = 1. if variant in (81,83,85,87) else 0.5
    splits = 1024 if variant in (82,83,86,87) else 2048
    value, by_group = moments(x,w,splits)
    half = by_group.shape[0]//2
    factors = j.where(j.arange(by_group.shape[0])<half,0.,math.log1p(alpha))
    z = jax.nn.logsumexp(by_group+factors[:,None],axis=0)
    mix = j.exp(by_group+factors[:,None]-z)
    dx = j.einsum('gl,gld->ld',mix,value.astype(j.float32)).astype(x.dtype)
    outer = mix[half:].sum(0)/(1.+alpha)
    return (z,outer),(x,w,z,dx)


@partial(jax.custom_vjp,nondiff_argnums=(2,))
def head(x,w,variant):
    return compute(x,w,variant)[0]


def backward(variant,res,up):
    x,w,z,dx=res
    dz=up[0]
    alpha=1. if variant in (81,83,85,87) else .5
    if variant in (84,85,86,87):
        scores=(x@w.T).astype(j.float32)
        coeff=(j.exp(scores-z[:,None])*dz[:,None]).astype(w.dtype)
        dw=coeff.T@x
    else:
        dw=readout_vjp(x,w,z,dz,1024)
    half=w.shape[0]//2
    dw=j.concatenate((dw[:half],dw[half:]*(1.+alpha)),axis=0)
    return (dx*dz[:,None]).astype(x.dtype),dw
head.defvjp(compute,backward)


def objective(x,w,y,weight,key,config):
    code,n=config
    variant,k=divmod(code,100000)
    b,t,d=x.shape
    pred,outer=head(x.reshape(-1,d),w[:k],variant)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    pred=pred.reshape(b,t)
    z=j.logaddexp(pred,j.where(y>=k,pos,-j.inf))
    cheap=z-pos
    score=.035+j.sqrt(outer.reshape(b,t)*j.exp(pred-z))+.15*(1-j.minimum(1.,j.exp(pos-z)))
    width=t//n
    score=jax.lax.stop_gradient(score.reshape(b,n,width))
    prob=score/score.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1))
    at=j.minimum((u>=j.cumsum(prob,-1)).sum(-1),width-1)
    selected=j.arange(n)[None,:]*width+at
    inverse=1/j.take_along_axis(prob,at[:,:,None],-1)[:,:,0]
    chosen=j.take_along_axis(x,selected[:,:,None],1)
    scores=(chosen@w.T).astype(j.float32)
    target=j.take_along_axis(y,selected,1)
    exact=jax.nn.logsumexp(scores,-1)-j.take_along_axis(scores,target[:,:,None],-1)[:,:,0]
    residual=(exact-j.take_along_axis(cheap,selected,1))*j.take_along_axis(weight,selected,1)*inverse
    return ((weight*cheap).sum()+residual.sum())/weight.sum()

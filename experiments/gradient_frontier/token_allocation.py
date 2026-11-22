"""Compute exact per-token residual gradient norms for held-out, randomly selected position strata."""
import io
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover
from .tail import trunk
from .fused_attention import install
from .group_regression import fit
from .online_adjoint import proxy
from .lowmem import stats
from .run import ROOT


def residual(model,data,bank,index,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=trunk(m,data)[:,index]
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(data[0],-1,axis=1)[:,index]
    target=w[y];pos=j.einsum('btd,btd->bt',x,target).astype(j.float32)
    if kind==113:
        cheap,mass,py=proxy(x,w,y,bank)
        prob=.06+j.sqrt(mass)+.10*(1.-j.minimum(py,1.))
    else:
        z,outer,top=stats(x@w[:12288].T,pos,y,.5)
        cheap=z-pos;mass=outer;py=j.exp(pos-z)
        prob=.035+j.sqrt(outer)+.15*(1.-j.minimum(top,1.))
    logits=(x@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,y[:,:,None],-1)[:,:,0]
    features=j.concatenate((x,target,x*target,j.stack((mass,py,cheap,prob),-1)),axis=-1).astype(j.float32)
    return ((exact-cheap)*data[1][:,index])[0],features[0]


@eqx.filter_jit
def measure(model,data,bank,index,kind):
    fun=lambda m: residual(m,data,bank,index,kind)
    vals,back,features=eqx.filter_vjp(fun,model,has_aux=True)
    width=len(index)
    def squared(row):
        grad,=back(row)
        return sum(j.vdot(g,g) for g in jax.tree.leaves(grad))
    # All 16-, 32- and 64-token partition sums use the same realized strata and gradients.
    masks=[j.eye(width)]
    for size in (16,32,64):
        if size<=width:masks.append(j.repeat(j.eye(width//size),size,axis=1))
    return jax.lax.map(squared,j.concatenate(masks)),features


def main(step,kind,stripes,seed):
    configure_storage();fs=fsspec.filesystem('s3');install();state,_=recover(step,fs)
    tmp=Path('/tmp/position-alloc.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:
        hold=tuple(z[k][512:] for k in ('tokens','weights','segments'))
        train=tuple(j.asarray(z[k][:4]) for k in ('tokens','weights','segments'))
    original=state if step==4001 else recover(4001,fs)[0]
    bank=fit(original.model,train,4096,113) if kind==113 else j.arange(1)
    rng=np.random.default_rng(seed);rows=[];features=[];indices=[]
    for i in range(stripes):
        doc=rng.integers(len(hold[0]));offset=64*rng.integers(16)
        d=tuple(j.asarray(a[doc:doc+1]) for a in hold)
        v,f=measure(state.model,d,bank,j.arange(64)+offset,kind)
        rows.append(np.asarray(v));features.append(np.asarray(f,dtype=np.float16));indices.append((int(doc),int(offset)))
        if (i+1)%8==0:
            buf=io.BytesIO();np.savez_compressed(buf,rows=rows,features=features,indices=indices,step=step,kind=kind)
            fs.pipe(ROOT+f'/token-allocation/s{step}-v{kind}-n{stripes}-r{seed}.npz',buf.getvalue())


if __name__=='__main__':main(*map(int,sys.argv[1:]))

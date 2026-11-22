"""Per-sequence gradient norms and sketch for testing document-level prediction and allocation."""
import io
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .fused_attention import install
from .native_eval import recover,objective
from .run import ROOT
from .tail import trunk


def count_sketch(a,key):
    # The decoder and embedding are large and sparse. Report their exact norms, and sketch the trunk only.
    bins=256
    v=a.reshape(-1)
    v=v*jax.random.rademacher(key,v.shape,dtype=j.float32)
    return j.pad(v,(0,(-v.size)%bins)).reshape(-1,bins).sum(0)


@eqx.filter_jit
def observe(model,data,key,method):
    config=(0,0) if method==0 else (61*100000+12288,64)
    grad=eqx.filter_grad(objective)(model,data,j.arange(1),key,config)
    leaves=jax.tree.leaves(grad)
    norm=j.stack([j.vdot(g,g) for g in leaves])
    trunk_sketch=sum(count_sketch(g,jax.random.key(4373+i)) for i,g in enumerate(leaves) if g.size<2_000_000)
    return norm,trunk_sketch


@eqx.filter_jit
def descriptors(model,data):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    x=trunk(m,data).astype(j.float32)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(data[0],-1,axis=1)
    yw=w[y].astype(j.float32)
    return j.concatenate((x.mean(1),x.std(1),yw.mean(1),(x*yw).mean(1)),1)


def main(step,begin,end,pop):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    tmp=Path('/tmp/doc-sketch.npz');fs.get_file(ROOT+'/inputs/'+pop+'.npz',str(tmp))
    with np.load(tmp) as z:pool=tuple(z[k][begin:end] for k in ('tokens','weights','segments'))
    norms=[];views=[];descs=[]
    for i in range(end-begin):
        d=tuple(j.asarray(v[i:i+1]) for v in pool)
        measured=[observe(state.model,d,jax.random.key(8402+begin+i+rep*15091),method) for rep,method in enumerate((0,61,61))]
        norm,sketch=zip(*measured)
        norms.append(np.asarray(norm));views.append(np.asarray(sketch))
        if i%8==0:
            descs.extend(np.asarray(descriptors(state.model,tuple(j.asarray(v[i:i+8]) for v in pool))))
        if (i+1)%128==0 or i+1==end-begin:
            buff=io.BytesIO();np.savez_compressed(buff,norms=np.asarray(norms),sketch=np.asarray(views),forward=np.asarray(descs)[:len(norms)],begin=begin,step=step)
            fs.pipe(ROOT+f'/document-sketch/{pop}-s{step}-{begin}-{end}.npz',buff.getvalue())
    print(json.dumps(dict(step=step,begin=begin,end=end,source=src)),flush=True)


if __name__=='__main__': main(*map(int,sys.argv[1:4]),sys.argv[4])

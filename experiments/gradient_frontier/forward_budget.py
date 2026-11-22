"""Measure the budget for observing extra independent examples and the variance a perfect readout would have."""
import json
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .contest import pair
from .head_to_head import update
from .native_eval import recover
from .fused_attention import install
from .tail import trunk
from .run import ROOT,ready


@eqx.filter_jit
def view(state,data,known_gradient,key,what):
    m=jax.tree.map(lambda v:v.astype(j.bfloat16),state.model)
    h=trunk(m,data)
    if what==1:
        w=m.get_lm_head().rearrange(('vocab','embed')).array
        y=j.roll(data[0],-1,axis=1)
        target=w[y]
        result=j.concatenate((h.astype(j.float32).mean(1),(h*target).astype(j.float32).mean(1)),axis=-1)
    else:result=h.astype(j.float32).mean(1)
    # Return both independently observable results, permitting XLA's usual overlap.
    return state.take_step(known_gradient,key=key)[0],result


def main(step,band):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    f=Path('/tmp/forward-budget.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:pool=tuple(z[n][512:] for n in ('tokens','weights','segments'))
    sizes=(2,4,8,16,32,48,64,96) if band==0 else (32,48,64,96,128,160,192,256)
    rng=np.random.default_rng(438742+step+band)
    def sample():
        ix=rng.integers(len(pool[0]),size=max(sizes))
        return {b:tuple(j.asarray(x[ix[:b]]) for x in pool) for b in sizes}
    # Use a real input gradient for the original optimizer; constructing it is explicitly free in this idealized arm.
    from .head_to_head import gradient
    dummy=j.arange(1);a=ready(sample())
    known=ready(gradient(state,a[min(sizes)],jax.random.key(47),dummy,61,4096,32))
    configs=[(m,k,n) for m,k,n in ((61,4096,32),(61,8192,64),(61,12288,64),(62,16384,64),(7,16384,64))]
    cases=[(m,k,n,b) for m,k,n in configs for b in sizes]
    controls=[(what,b) for what in (0,1) for b in sizes]
    rows=[dict(method=m,k=k,n=n,b=b,ms=[],pair=[]) for m,k,n,b in cases]
    views=[dict(label='forward' if what==0 else 'forward-and-labels',b=b,ms=[]) for what,b in controls]
    cfg=dict(step=step,band=band,checkpoint=src,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='forward-acquisition-cost-and-oracle',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,rows=rows,views=views,reference_batch=min(sizes),reference=[])
    dst=ROOT+f'/forward-budget/s{step}-r{band}.json'
    for i in range(33):
        a,b=ready((sample(),sample()));key=jax.random.key(994+i)
        for jx in rng.permutation(np.arange(-len(views),len(cases))):
            tick=time.perf_counter()
            if jx<0:
                what,B=controls[-jx-1];ready(view(state,a[B],known,key,what))
            else:
                m,k,n,B=cases[jx];ready(update(state,a[B],key,dummy,m,k,n))
            elapsed=1000*(time.perf_counter()-tick)
            if i:(views[-jx-1] if jx<0 else rows[jx])['ms'].append(elapsed)
        if not i:continue
        B=min(sizes);out['reference'].append(float(pair(state,a[B],b[B],key,dummy,0,0,0)))
        for row,(m,k,n,B) in zip(rows,cases):row['pair'].append(float(pair(state,a[B],b[B],key,dummy,m,k,n)))
        if i%4==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':32});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

"""Sweep the cost of calibration, vocabulary coverage and exact target positions against tuned debiased batches."""
import json
import os
import sys
import time
from pathlib import Path
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
from .group_regression import fit
from .run import ROOT,ready


def main(step,B,k):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    start=state if step==4001 else recover(4001,fs)[0]
    path=Path('/tmp/grid-data.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(path))
    with np.load(path) as z:
        hold=tuple(z[name][512:] for name in ('tokens','weights','segments'))
        cal=tuple(j.asarray(z[name][:4]) for name in ('tokens','weights','segments'))
    bank=ready(fit(start.model,cal,k,113));prep=[]
    for _ in range(8):
        t=time.perf_counter();bank=ready(fit(start.model,cal,k,113));prep.append(1000*(time.perf_counter()-t))
    cases=[(m,k,n,B) for m in (113,172) for n in (8,16,32,64,128,256)]
    bs=sorted({max(2,B//4),B//2,3*B//4,B,5*B//4})
    configs=((61,4096,16),(61,4096,32),(61,8192,32),(61,8192,64),(61,12288,64),(61,16384,128),(62,16384,64),(7,16384,64))
    cases+= [(m,kb,n,b) for m,kb,n in configs for b in bs]
    rows=[dict(method=m,k=kb,n=n,b=b,ms=[],pair=[]) for m,kb,n,b in cases]
    seed=78328+step+k+B;rng=np.random.default_rng(seed)
    def draw():
        ix=rng.integers(len(hold[0]),size=max(bs))
        return {b:tuple(j.asarray(v[ix[:b]]) for v in hold) for b in bs}
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,B=B,k=k,seed=seed,checkpoint=src)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='tuned-debiased-crossover-grid',id=os.urandom(5).hex(),config=cfg)
    out=dict(**cfg,wandb=run.url,device=jax.devices()[0].device_kind,prep=prep,rows=rows)
    dst=ROOT+f'/strong-grid/s{step}-b{B}-k{k}.json';dummy=j.arange(1)
    for it in range(25):
        left,right=ready((draw(),draw()));key=jax.random.key(seed+it)
        for ix in rng.permutation(len(cases)):
            m,kb,n,b=cases[ix];bk=bank if m in (113,172) else dummy
            val=float(pair(state,left[b],right[b],key,bk,m,kb,n))
            if it:rows[ix]['pair'].append(val)
        for ix in rng.permutation(len(cases)):
            m,kb,n,b=cases[ix];bk=bank if m in (113,172) else dummy
            t=time.perf_counter();ready(update(state,left[b],key,bk,m,kb,n));ms=1000*(time.perf_counter()-t)
            if it:rows[ix]['ms'].append(ms)
        if it and it%6==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'cycles':24});run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

"""Measure the joint short/long segment estimator, integrated native precision correction and optimizer costs."""
import os
import subprocess
import sys
import time

import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from experiments.hybrid_transfer.storage import configure_storage
from rigging.secrets import require_secret
from . import prepared_blocks as prep,segment_queries as separate,fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import configurations
from .independent_audits import copy_state
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready


def main(step,cut,batch,k,balanced=0,prefix=0,joint=0):
    from . import joint_segment_queries
    seg=joint_segment_queries if joint else separate
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    state,src=recover(step,fs);fused_attention.install()
    board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg=dict(**meta,step=step,cut=cut,batch=batch,k=k,length_balanced=bool(balanced),split_prefix=bool(prefix),joint=bool(joint),device=jax.devices()[0].device_kind,board=board,source=src)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='attention-segment-sampling',id=os.urandom(6).hex(),mode='online',config=cfg)
    phases=tuple(j.asarray(i,j.int32) for i in range(16))
    cases=[(m,kv,n,b,0) for m,kv,n in ((61,4096,32),(61,12288,64),(62,16384,64))
           for b in (1,2,4,8,12,16,24,32)]
    out=dict(**cfg,wandb=run.url,parts=[],prices=[],rows=[dict(config=c,ms=[],pairs=[]) for c in cases])
    folder='joint-static-segment-prefix-pricing' if joint and prefix else 'joint-static-segment-pricing' if joint else 'segment-prefix-pricing' if prefix else 'balanced-segment-pricing' if balanced else 'segment-pricing'
    path=ROOT+f'/{folder}/s{step}-l{cut}-b{batch}-k{k}.json'
    for i in range(25):
        rng=np.random.default_rng(np.random.SeedSequence([820153,step,i]))
        host=[tuple(x[idx] for x in pool) for idx in rng.integers(len(pool[0]),size=(4,32))]
        panels=ready([tuple(j.asarray(x) for x in d) for d in host])
        tic=time.perf_counter()
        layouts=ready([seg.plan(x[2],cut,batch,rng,bool(balanced),bool(prefix)) for x in host])
        phase=[tuple(phases[int(rng.integers(16))] for _ in range(2)) for _ in host]
        overhead=1000*(time.perf_counter()-tic)/4
        g=[ready(seg.draw(state.model,d,l,cut,k,p,bool(prefix))) for d,l,p in zip(panels[:2],layouts,phase)]
        de=[ready(seg.correction(state.model,d,l,cut,k,phases,bool(prefix))) for d,l in zip(panels[2:],layouts[2:])]
        stats=np.asarray(norm_moments(*g)).tolist()+np.asarray(norm_moments(*de)).tolist()
        sizes=sorted({c[3] for c in cases})
        a,b=ready([{n:tuple(x[:n] for x in d) for n in sizes} for d in panels[:2]])
        salt=j.asarray(i+43179,j.uint32);prices=np.zeros(2)
        for ix in rng.permutation(np.arange(-2,len(cases))):
            ix=int(ix);work=copy_state(state);tic=time.perf_counter()
            if ix<0:
                raw=seg.draw(work.model,panels[0],layouts[0],cut,k,phase[0],bool(prefix))
                delta=seg.correction(work.model,panels[2],layouts[2],cut,k,phases,bool(prefix)) if ix==-2 else None
                ans=seg.advance((raw,delta),work)
            else:
                m,kv,n,bs,_=cases[ix];ans=prep.baseline((a[bs],salt,m,kv,n),work)
            ready(ans);ms=1000*(time.perf_counter()-tic)
            if ix<0:prices[-ix-1]=ms+overhead*(1+(ix==-2))
            elif i:out['rows'][ix]['ms'].append(ms)
        if i:
            out['parts'].append(stats);out['prices'].append(prices.tolist())
            for row,(m,kv,n,bs,_) in zip(out['rows'],cases):row['pairs'].append(float(pair(state,a[bs],b[bs],jax.random.key(i+72437),j.arange(1),m,kv,n)))
            if i%2==0:publish(fs,path,out)
    run.log({'cycles':24});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

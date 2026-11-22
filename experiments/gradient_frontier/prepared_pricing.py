"""Compare a prepared block estimator with baselines using the same fused key handling."""
import json
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
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fifty_blocks_strict as strict,prepared_blocks as impl,fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import configurations
from .independent_audits import copy_state,forecast
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready

def main(step):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs);state,src=recover(step,fs);fused_attention.install()
    dev=jax.devices()[0].device_kind
    board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    plan,phases,prep=impl.prepare(151705+step)
    cfg=dict(**meta,step=step,checkpoint=src,hardware=dev,board=board,algorithm=406,prep_ms=prep)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='prepared-fused-block-pricing',id=os.urandom(6).hex(),mode='online',config=cfg)
    cases=sorted({c for panel in (4,5) for c in configurations(panel) if c[0]!=402})
    out=dict(**cfg,wandb=run.url,times=[],components=[],rows=[dict(config=c,ms=[],pairs=[]) for c in cases])
    rng=np.random.default_rng(np.random.SeedSequence([459731,step]));dummy=j.arange(1);pred=ready(forecast(state,10))
    for i in range(65):
        a,b,c,d=ready(tuple(tuple(j.asarray(x[idx]) for x in pool) for idx in rng.integers(len(pool[0]),size=(4,128))))
        bs=sorted({v[3] for v in cases});left,right=ready(tuple({s:tuple(x[:s] for x in panel) for s in bs} for panel in (a,b)))
        times=np.zeros(6);slot=plan[i%50];salt=j.asarray(19596+i,j.uint32)
        for item in rng.permutation(np.arange(-6,len(cases))):
            ix=int(item);work=copy_state(state);tic=time.perf_counter()
            if ix<0:
                tag=-1-ix;ans=impl.update(work,a,slot,phases,tag%3,tag>=3)
            else:
                m,k,n,B,_=cases[ix];ans=impl.baseline((left[B],salt,m,k,n),work)
            ready(ans);ms=1000*(time.perf_counter()-tic)
            if i and ix>=0:out['rows'][ix]['ms'].append(ms)
            elif ix<0:times[tag]=ms
        if not i:continue
        # Fresh independent offsets for variance; the precomputed timing schedule is not recycled here.
        def random_slot():
            idx=rng.permutation(128)[:64];off=rng.integers(16,size=(2,32,64),dtype=np.int32)
            return (j.asarray(idx[:32],j.int32),j.asarray(idx[32:],j.int32),j.asarray(off[0]),j.asarray(off[1]),j.asarray(rng.integers(16),j.int32))
        sa,sb,sc,sd=[random_slot() for _ in range(4)];mom=[]
        for part in (1,2):
            def raw(data,s):
                return impl.fp32(impl.raw_block(state.model,data,s[0],s[2],s[4],part))
            mom.extend(np.asarray(strict.moments(raw(a,sa),raw(b,sb),pred,part)).tolist())
        mom.extend(np.asarray(norm_moments(impl.discrepancy(state,c,sc,phases),impl.discrepancy(state,d,sd,phases))).tolist())
        out['components'].append(mom);out['times'].append(times.tolist())
        for row,(m,k,n,B,_) in zip(out['rows'],cases):row['pairs'].append(float(pair(state,left[B],right[B],jax.random.key(73958+i),dummy,m,k,n)))
        if i%4==0:publish(fs,ROOT+f'/prepared-block-pricing-v1/{dev.split()[1]}-s{step}.json',out)
    run.log({'cycles':64});run.finish()

if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(int(sys.argv[1]))

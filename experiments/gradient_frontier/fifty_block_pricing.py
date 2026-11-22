"""Price each realized branch of the corrected implementation against both frozen same-board panels."""
import os
import subprocess
import sys
import time
import fsspec
import haliax as h
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fifty_blocks_strict as fixed,fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import configurations,update
from .independent_audits import copy_state,forecast
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready


def main(step,nested=0):
    from . import nested_block_draw
    impl=nested_block_draw if nested else fixed
    configure_storage();fs=fsspec.filesystem('s3')
    pool,meta=population(fs);state,src=recover(step,fs);fused_attention.install()
    device=jax.devices()[0].device_kind
    board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg=dict(**meta,step=step,checkpoint=src,hardware=device,board=board,algorithm=403 if nested else 402)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='strict-block-conditional-clocks',id=os.urandom(6).hex(),mode='online',config=cfg)
    cases=sorted({c for p in (4,5) for c in configurations(p) if c[0]!=402})
    rng=np.random.default_rng(np.random.SeedSequence([163521,step]))
    arrays=lambda ids:tuple(jnp.asarray(v[ids]) for v in pool)
    dest=ROOT+f'/fifty-block-pricing-v1/{device.split()[1]}-s{step}.json'
    if nested:dest=dest.replace('fifty-block-pricing-v1','nested-block-pricing-v1')
    out=dict(**cfg,wandb=run.url,times=[],components=[],rows=[dict(config=c,ms=[],pairs=[]) for c in cases])
    dummy=jnp.arange(1);pred=ready(forecast(state,10))
    for i in range(65):
        indices=ready(jnp.asarray(rng.permutation(128)[:64],jnp.int32))
        data=ready(arrays(rng.integers(len(pool[0]),size=128)))
        bsizes=sorted({c[3] for c in cases}|{32})
        a,b=[ready({bs:arrays(ix[:bs]) for bs in bsizes}) for ix in rng.integers(len(pool[0]),size=(2,32))]
        views=ready({bs:tuple(x[:bs] for x in data) for bs in bsizes})
        costs=np.zeros(6)
        for idx in rng.permutation(np.arange(-6,len(cases))):
            work=copy_state(state);tic=time.perf_counter()
            if idx<0:
                tag=int(-1-idx)
                ans=impl.update(work,data,(tag%3,tag>=3,indices),75391+i)
            else:
                c=cases[idx]
                ans=update(work,views[c[3]],None,c,75391+i)
            ready(ans);ms=1000*(time.perf_counter()-tic)
            if i and idx>=0:out['rows'][idx]['ms'].append(ms)
            elif idx<0:costs[tag]=ms
        if not i:continue
        key=jax.random.key(13718+step*100+i);k1,k2,k3,k4=jax.random.split(key,4)
        moments=[]
        if nested:
            ga=impl.measurement(state,a[32],k1);gb=impl.measurement(state,b[32],k2)
        for part in (1,2):
            def draw(d,k):
                u,v=jax.random.split(k)
                return fixed.raw_block(state,d,u,part,jax.random.randint(v,(),0,16,dtype=jnp.int32))
            raws=(ga[part-1],gb[part-1]) if nested else (draw(a[32],k1),draw(b[32],k2))
            moments.extend(np.asarray(fixed.moments(*raws,pred,part)).tolist())
        c,d=[arrays(ix) for ix in rng.integers(len(pool[0]),size=(2,32))]
        moments.extend(np.asarray(norm_moments(impl.discrepancy(state,c,k3),impl.discrepancy(state,d,k4))).tolist())
        for row,config in zip(out['rows'],cases):
            m,k,n,bs,_=config;row['pairs'].append(float(pair(state,a[bs],b[bs],key,dummy,m,k,n)))
        out['times'].append(costs.tolist());out['components'].append(moments)
        if i%4==0:publish(fs,dest,out)
    run.log({'cycles':64});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

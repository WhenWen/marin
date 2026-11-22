"""Vary the sample count and vocabulary correction resolution in an individually purchased parameter block."""
import functools
import os
import sys
import time
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import block_audits,fifty_blocks_strict as strict,fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_crossover import configurations,read_np,update
from .independent_audits import copy_state,forecast
from .native_eval import recover
from .run import ROOT,ready


@eqx.filter_jit
def draw(state,data,key,part,k,n):
    g=block_audits.draw(state,data,key,k,n,part,0)
    return tuple(x for path,x in jax.tree_util.tree_leaves_with_path(g) if block_audits.readout(path)==(part==2))


def candidate(state,full,indices,key,part,b,k,n):
    base,nextkey=jax.random.split(state.training_key)
    raw=draw(state,tuple(z[indices[:b]] for z in full),jax.random.fold_in(base,key),part,k,n) if part else None
    return strict.advance((raw,part,None,nextkey),state)


def main(step,part):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs);fused_attention.install()
    pool=tuple(z[512:] for z in read_np(fs,ROOT+'/inputs/native-packed.npz'))
    settings=[(b,k,n) for b in (16,32,64,128) for k,n in ((1024,16),(4096,32),(4096,64),(12288,64))]
    controls=sorted({c for panel in (4,5) for c in configurations(panel) if c[0]!=402})
    cfg=dict(step=step,part=part,source=src,device=jax.devices()[0].device_kind,methods=settings,comparison='conditional lower bound without native audits')
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='search-individual-block-resolutions',id=os.urandom(6).hex(),mode='online',config=cfg)
    out=dict(**cfg,no_query_ms=[],wandb=run.url,rows=[dict(config=c,ms=[],pairs=[]) for c in controls],cases=[dict(config=c,ms=[],components=[]) for c in settings])
    pred=ready(forecast(state,10));dest=ROOT+f'/block-resolution-v1/s{step}-part{part}.json';dummy=j.arange(1)
    for i in range(33):
        rng=np.random.default_rng(np.random.SeedSequence([918317,step,i]))
        sample=lambda indices:tuple(z[indices] for z in pool)
        a,b=[ready(sample(ids)) for ids in rng.integers(len(pool[0]),size=(2,128))]
        sel=ready(j.asarray(rng.permutation(128),j.int32))
        sizes=sorted({q[0] for q in settings}|{q[3] for q in controls})
        left,right=[ready({s:tuple(v[:s] for v in d) for s in sizes}) for d in (a,b)]
        ka,kb=jax.random.split(jax.random.key(step*100+i))
        for idx in rng.permutation(len(settings)+len(controls)+1):
            work=copy_state(state);tic=time.perf_counter()
            if idx<len(settings):
                bs,k,n=settings[idx];ans=candidate(work,a,sel,i,part,bs,k,n)
            elif idx==len(settings):ans=candidate(work,a,sel,i,0,0,0,0)
            else:
                c=controls[idx-len(settings)-1];ans=update(work,left[c[3]],None,c,i)
            ready(ans);ms=1000*(time.perf_counter()-tic)
            if i:
                if idx<len(settings):out['cases'][idx]['ms'].append(ms)
                elif idx==len(settings):out['no_query_ms'].append(ms)
                else:out['rows'][idx-len(settings)-1]['ms'].append(ms)
        if not i:continue
        for row,(bs,k,n) in zip(out['cases'],settings):
            row['components'].append(np.asarray(strict.moments(draw(state,left[bs],ka,part,k,n),draw(state,right[bs],kb,part,k,n),pred,part)).tolist())
        for row,(m,k,n,bs,_) in zip(out['rows'],controls):
            row['pairs'].append(float(pair(state,left[bs],right[bs],ka,dummy,m,k,n)))
        if i%4==0:publish(fs,dest,out)
    run.log({'cycles':32});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

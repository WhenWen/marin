"""Learn layerwise control coefficients for nested short-context and long-context gradient queries."""
import functools
import os
import subprocess
import sys
import time

import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from experiments.hybrid_transfer.storage import configure_storage
from rigging.secrets import require_secret
from . import block_audits,fused_attention,prepared_blocks as exact
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_blocks_strict import moments as high_moments
from .fifty_crossover import configurations
from .independent_audits import copy_state,forecast
from .lowmem import stats
from .native_eval import recover
from .nested_forecast import norm_moments
from .run import ROOT,ready
from .tail import trunk


def short_loss(model,data,ids,starts,part,length,k):
    m=jax.tree_util.tree_map_with_path(lambda path,a:a if block_audits.readout(path)==(part==2) else jax.lax.stop_gradient(a),model)
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),m)
    tokens,weights,segments=data
    pos=starts[:,None]*length+j.arange(length)
    grab=lambda x:x[ids[:,None],pos]
    target=grab(j.roll(tokens,-1,1))
    observed=tuple(grab(x) for x in (tokens,weights,segments))
    hidden=trunk(m,observed)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    logits=hidden@w[:k].T
    positive=j.einsum('btd,btd->bt',hidden,w[target]).astype(j.float32)
    z,*_=stats(logits,positive,target,.5)
    return ((z-positive)*observed[1]).sum()*(tokens.shape[1]/length)/(len(ids)*1023.)


@eqx.filter_jit
def cheap(model,data,slot,starts,part,length,k):
    grad=eqx.filter_grad(short_loss)(model,data,slot[0],starts,part,length,k)
    return tuple(v.astype(j.bfloat16) for path,v in jax.tree_util.tree_leaves_with_path(grad) if block_audits.readout(path)==(part==2))


def high(model,data,slot,part):
    return exact.raw_block(model,data,slot[0],slot[2],slot[4],part)


@eqx.filter_jit
def moments(state,low,hi,part):
    pred=[v for path,v in jax.tree_util.tree_leaves_with_path(forecast(state,10)) if block_audits.readout(path)==(part==2)]
    rows=[]
    for p,a,b in zip(pred,low,hi):
        a=a.astype(j.float32)-p;b=b.astype(j.float32)-p
        rows.append(j.stack([j.vdot(a,a),j.vdot(b,b),j.vdot(a,b)]))
    return j.stack(rows)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def advance(inputs,state):
    low,hi,part,coef,delta=inputs
    pred=forecast(state,10)
    if part:
        q,p=(.6,.3) if part==1 else (.2,.075)
        seq=iter(zip(low,coef,hi if hi is not None else [None]*len(low)))
        def apply(path,prior):
            if block_audits.readout(path)!=(part==2):return prior
            a,w,b=next(seq)
            adj=w*(a.astype(j.float32)-prior)
            return prior+adj/q if b is None else prior+adj/q+(b.astype(j.float32)-prior-adj)/p
        pred=jax.tree_util.tree_map_with_path(apply,pred)
    if delta is not None:pred=jax.tree.map(lambda x,d:x+d/.002,pred,delta)
    return state.take_step(pred,key=jax.random.split(state.training_key)[1])[0]


def main(step,length,k):
    configure_storage();fs=fsspec.filesystem('s3');pool,meta=population(fs)
    state,source=recover(step,fs);fused_attention.install()
    board=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg=dict(**meta,step=step,source=source,length=length,k=k,board=board,device=jax.devices()[0].device_kind)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='coupled-short-gradient-controls',id=os.urandom(6).hex(),mode='online',config=cfg)
    plans,phases,prep=exact.prepare(190153+step)
    rng=np.random.default_rng(387169+step)
    cases=sorted({c for panel in (4,5) for c in configurations(panel) if c[0]!=402})
    dest=ROOT+f'/coupled-block-control/s{step}-l{length}-k{k}.json'
    out=dict(**cfg,wandb=run.url,prep_ms=prep,timings=[],moments=[],native=[],rows=[dict(config=c,ms=[],pairs=[]) for c in cases])
    for i in range(25):
        a,b,c,d=ready([tuple(j.asarray(x[ids]) for x in pool) for ids in rng.integers(len(pool[0]),size=(4,128))])
        start=ready(j.asarray(rng.integers(1024//length,size=(4,32)),j.int32))
        def random_slot():
            ids=rng.permutation(128)[:64]
            offset=rng.integers(16,size=(2,32,64),dtype=np.int32)
            return (j.asarray(ids[:32],j.int32),j.asarray(ids[32:],j.int32),j.asarray(offset[0]),j.asarray(offset[1]),j.asarray(rng.integers(16),j.int32))
        slots=ready([random_slot() for _ in range(4)])
        # Independently sample phases and offsets in each new pair, independently of the calibration pair.
        obs=[];fits=[];mean_sq=[];cal=time.perf_counter()
        for part in (1,2):
            hi=[high(state.model,x,s,part) for x,s in zip((a,b),slots)]
            lo=[cheap(state.model,x,s,u,part,length,k) for x,s,u in zip((a,b),slots,start)]
            stats2=[np.asarray(moments(state,l,g,part)) for l,g in zip(lo,hi)]
            obs.append(np.mean(stats2,0).tolist())
            mean_sq.append(float(high_moments(exact.fp32(hi[0]),exact.fp32(hi[1]),forecast(state,10),part)[1]))
            fits.append(j.asarray(np.clip(np.mean(stats2,0)[:,2]/np.maximum(1e-12,np.mean(stats2,0)[:,0]),-3,3),j.float32))
        ready(fits);cal_ms=1000*(time.perf_counter()-cal)
        if i==1:
            out['cal_ms']=cal_ms
            learned=fits
        coef=fits if i==0 else learned
        corrections=[exact.discrepancy(state,x,s,phases) for x,s in zip((c,d),slots[2:])]
        nn=np.asarray(norm_moments(*corrections)).tolist()
        sizes=sorted({q[3] for q in cases})
        left,right=ready([{sz:tuple(x[:sz] for x in z) for sz in sizes} for z in (a,b)])
        timevec=np.zeros(10);salt=j.asarray(15492+i,j.uint32)
        for ix in rng.permutation(np.arange(-10,len(cases))):
            ix=int(ix);work=copy_state(state);tic=time.perf_counter()
            if ix<0:
                tag=-ix-1;kind=tag%5;part=0 if kind==0 else 1 if kind<=2 else 2
                l=cheap(work.model,a,slots[0],start[0],part,length,k) if part else None
                g=high(work.model,a,slots[0],part) if kind in (2,4) else None
                delta=exact.discrepancy(work,a,slots[0],phases) if tag>=5 else None
                ans=advance((l,g,part,coef[part-1] if part else None,delta),work)
            else:
                m,kv,n,B,_=cases[ix];ans=exact.baseline((left[B],salt,m,kv,n),work)
            ready(ans);elapsed=1000*(time.perf_counter()-tic)
            if ix<0:timevec[tag]=elapsed
            elif i>=2:out['rows'][ix]['ms'].append(elapsed)
        if i>=2:
            out['timings'].append(timevec.tolist());out['moments'].append(obs);out['native'].append(nn+mean_sq)
            for row,(m,kv,n,B,_) in zip(out['rows'],cases):row['pairs'].append(float(pair(state,left[B],right[B],jax.random.key(17291+i),j.arange(1),m,kv,n)))
            if i%4==0:publish(fs,dest,out)
        if i==1:out['coefficients']=[np.asarray(v).tolist() for v in learned]
    run.log({'cycles':len(out['timings'])});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

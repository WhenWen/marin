"""Same-card causal ridge vs no prediction, including live gate selection, native donation and baseline scaling."""
import json
import os
import sys
import time
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .diagnostic_artifacts import publish
from . import fused_attention,online_coin
from .contest import pair
from .independent_audits import regular,residual,copy_state,donated_baseline,forecast
from .native_eval import recover
from .nested_forecast import norm_moments
from .roulette_training import learned_update,update
from .run import ROOT,ready
from .subspace_forecast import components,native_audit


def main(step,rep):
    configure_storage()
    fs=fsspec.filesystem('s3')
    state,src=recover(step,fs)
    previous=state if step==4001 else recover(step-1,fs)[0]
    history=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
    aux=ready(online_coin.prepare(state,previous,history,step))
    del previous
    prediction=ready(online_coin.prediction(state,*aux))
    zero=ready(forecast(state,0))
    original=Attention.__call__
    fused_attention.install()
    p,r,B,k,n=.2,.002,32,12288,64
    manifest='native-confirm3' if rep==1 else 'native-confirm5'
    file=Path('/tmp/learned-gate-pool.npz')
    fs.get_file(ROOT+'/inputs/'+manifest+'.npz',str(file))
    meta=json.loads(fs.cat_file(ROOT+'/inputs/'+manifest+'.json'))
    import hashlib
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['sha256']
    with np.load(file) as z:pool=tuple(z[name] for name in ('tokens','weights','segments'))
    seed=813540+step+rep
    rng=np.random.default_rng(seed)
    cases=[(m,kv,nv,b) for m,kv,nv in ((61,64,4),(61,256,8),(61,1024,16),(61,4096,32),(61,8192,32),
            (61,8192,64),(61,12288,64),(62,16384,64),(7,16384,64))
           for b in (1,2,3,4,6,8,12,16,24,32,48,64)]
    cases.extend((0,0,0,b) for b in (1,2,4,8))
    rows=[dict(method=m,k=kv,n=nv,b=b,ms=[],pair=[]) for m,kv,nv,b in cases]
    dev=jax.devices()[0].device_kind
    cfg=dict(step=step,rep=rep,checkpoint=src,seed=seed,B=B,k=k,n=n,p=p,r=r,holdout=manifest,sha=meta['sha256'],device=dev)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='paired-learned-gating-native',id=os.urandom(5).hex(),config=cfg)
    dest=ROOT+'/paired-learned-gate/'+f'{dev.split()[1]}-s{step}-r{rep}.json'
    out=dict(**cfg,wandb=run.url,rows=rows,times=[],components=[],live=[])
    dummy=j.arange(1)
    def sample(size):
        ix=rng.integers(len(pool[0]),size=size)
        return tuple(j.asarray(x[ix]) for x in pool)
    def delta(modelstate,data,key):
        Attention.__call__=original
        try:gold=native_audit(modelstate.model,data,B)
        finally:fused_attention.install()
        return residual(modelstate,data,key,k,n,gold)
    def gated(modelstate,data,opkey,tag,learn):
        key,next_key=jax.random.split(opkey)
        c1,c2,sel,gk,dk=jax.random.split(key,5)
        primary=bool(jax.random.bernoulli(c1,p))
        precision=bool(jax.random.bernoulli(c2,r))
        if tag<4:primary,precision=bool(tag&1),bool(tag&2)
        chosen=jax.random.permutation(sel,128) if primary or precision else None
        a=tuple(x[chosen[:B]] for x in data) if primary else None
        d=delta(modelstate,tuple(x[chosen[B:2*B]] for x in data),dk) if precision else None
        if learn:
            prev,gram,cross=aux
            args=(a,d,gk,next_key,p,r,(jax.tree.map(lambda x:x.copy(),prev),gram,cross))
            ready(learned_update(args,modelstate))
        else:ready(update((a,d,gk,next_key,p,r,0),modelstate))
        return int(primary)+2*int(precision)
    for i in range(73):
        a,b,ext,ext2,u,v=ready(tuple(sample(sz) for sz in (B,B,B,B,128,128)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(seed+i),4)
        left,right=ready(tuple({sz:tuple(x[:sz] for x in dat) for sz in {q[-1] for q in cases}} for dat in (u,v)))
        elapsed=np.zeros((2,4));live=np.zeros((2,2))
        for ix in rng.permutation(np.arange(-10,len(cases))):
            work=copy_state(state)
            ix=int(ix)
            opkey=ready(jax.random.key(seed+i*200+(ix if ix>=0 else (-ix-1)%5)))
            tic=time.perf_counter()
            if ix<0:
                learn,tag=divmod(-ix-1,5)
                branch=gated(work,u,opkey,tag,bool(learn))
            else:
                m,kv,nv,sz=cases[ix]
                gk,_=jax.random.split(opkey)
                ready(donated_baseline((left[sz],gk,dummy,m,kv,nv),work))
            ms=1000*(time.perf_counter()-tic)
            if i:
                if ix>=0:rows[ix]['ms'].append(ms)
                elif tag==4:live[learn]=[branch,ms]
                else:elapsed[learn,tag]=ms
        if not i:continue
        g1=regular(state,a,ka,k,n);g2=regular(state,b,kb,k,n)
        vz=components(g1,g2,zero);vl=components(g1,g2,prediction)
        prec=norm_moments(delta(state,ext,kc),delta(state,ext2,kd))
        out['times'].append(elapsed.tolist());out['live'].append(live.tolist())
        out['components'].append(np.asarray(j.concatenate((vz,vl[1:],prec))).tolist())
        for row,(m,kv,nv,sz) in zip(rows,cases):
            row['pair'].append(float(pair(state,left[sz],right[sz],ka,dummy,m,kv,nv)))
        if i%6==0:publish(fs,dest,out)
    run.log({'cycles':72});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

"""Fuse runs of predictable steps, cross measuring complete ten-update windows and equally fused batch controls."""
import functools
import json
from math import comb
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
from . import fused_attention,online_coin
from .burst_fusion import burst,error
from .diagnostic_artifacts import publish
from .head_to_head import gradient
from .independent_audits import copy_state,residual
from .native_eval import recover
from .roulette_training import learned_update,update
from .run import ROOT,ready
from .subspace_forecast import native_audit


@functools.partial(eqx.filter_jit,donate='all-except-first')
def base_window(args,state):
    data,k,n,unroll=args
    arr,static=eqx.partition(state,eqx.is_array)
    def one(current,d):
        cur=eqx.combine(current,static)
        key,nextkey=jax.random.split(cur.training_key)
        g=gradient(cur,d,key,j.arange(1),61,k,n)
        nxt=cur.take_step(g,key=nextkey)[0]
        return eqx.filter(nxt,eqx.is_array),None
    a,_=jax.lax.scan(one,arr,data,unroll=unroll)
    return eqx.combine(a,static)


def main(step,learn,band):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    original=Attention.__call__;fused_attention.install()
    if learn:
        prev=state if step==4001 else recover(step-1,fs)[0]
        hist=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
        aux=ready(online_coin.prepare(state,prev,hist,step));del prev
    else:aux=None
    temp=Path('/tmp/packed-windows-data.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(temp))
    with np.load(temp) as file:pool=tuple(file[k] for k in ('tokens','weights','segments'))
    p=.2 if band==0 else .375;r=.002;dev=jax.devices()[0].device_kind;seed=623815+step+band
    rng=np.random.default_rng(seed)
    cases=[(k,n,b,u) for k,n in ((4096,32),(8192,32),(12288,64)) for b in (1,2,4,8,16,32) for u in (1,2)]
    cfg=dict(step=step,learn=learn,band=band,p=p,r=r,seed=seed,source=src,device=dev,holdout='native-confirm3',length=10)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='coalesced-native-windows-and-batch-scaling',id=os.urandom(5).hex(),config=cfg)
    rows=[dict(k=k,n=n,b=b,unroll=u,ms=[]) for k,n,b,u in cases]
    out=dict(**cfg,wandb=run.url,rows=rows,gate_ms=[],patterns=[],checks={})
    dest=ROOT+f'/packed-windows/{dev.split()[1]}-s{step}-learn{learn}-band{band}.json'
    def active(work,d,tag):
        cur,aux0=work
        key,nextkey=jax.random.split(cur.training_key)
        _,_,select,gk,dk=jax.random.split(key,5)
        ids=jax.random.permutation(select,128)
        data=tuple(v[ids[:32]] for v in d) if tag&1 else None
        delta=None
        if tag&2:
            x=tuple(v[ids[32:64]] for v in d)
            Attention.__call__=original
            try:gold=native_audit(cur.model,x,32)
            finally:fused_attention.install()
            delta=residual(cur,x,dk,12288,64,gold)
        if learn:
            previous,gram,cross=aux0
            nxt,ax=learned_update((data,delta,gk,nextkey,p,r,(jax.tree.map(lambda x:x.copy(),previous),gram,cross)),cur)
            return ready((nxt,ax))
        return ready((update((data,delta,gk,nextkey,p,r,0),cur),None))
    def with_schedule(work,all_data,pattern,packed):
        i=0
        while i<10:
            if pattern[i]:
                d=tuple(x[i] for x in all_data)
                work=active(work,d,int(pattern[i]));i+=1
            else:
                length=1
                if packed:
                    while i+length<10 and pattern[i+length]==0:length+=1
                work=ready(burst((length,bool(learn),1),work));i+=length
        return work
    def make_data():
        ids=rng.integers(len(pool[0]),size=(10,128))
        return ready(tuple(j.asarray(a[ids]) for a in pool))
    # Boundary checks exercise both an exact audit and contiguous zero-gradient spans.
    warm=make_data()
    for length in range(1,11):ready(burst((length,bool(learn),1),copy_state((state,aux))))
    for pat in ([0,0,1,0,2,0,0,0,3,0],[1,0,0,0,0,0,0,0,0,0]):
        l=with_schedule(copy_state((state,aux)),warm,pat,False);rr=with_schedule(copy_state((state,aux)),warm,pat,True)
        err=np.asarray(error(l,rr));out['checks'][''.join(map(str,pat))]=dict(different=int(err[:,0].sum()),max_abs=float(err[:,1].max()))
        assert int(l[0].step)==step+10 and int(rr[0].step)==step+10
        del l,rr
    publish(fs,dest,out)
    # Stratified timings conditional on J count 0,1,>=2 give expected precision cost without waiting for rare draws.
    mass=np.array([comb(10,j0)*r**j0*(1-r)**(10-j0) for j0 in range(11)])
    for it in range(81):
        data=make_data()
        subsets=ready({b:tuple(a[:,:b] for a in data) for _,_,b,_ in cases})
        gen_seed=int(rng.integers(2**31-1))
        measures=np.zeros(3);patterns=[]
        for ix in rng.permutation(np.arange(-3,len(cases))):
            ix=int(ix);work=copy_state((state,aux)) if ix<0 else copy_state(state)
            tic=time.perf_counter()
            if ix<0:
                stratum=-ix-1;g=np.random.default_rng(gen_seed)
                pattern=(g.random(10)<p).astype(int)
                nj=stratum if stratum<2 else int(g.choice(np.arange(2,11),p=mass[2:]/mass[2:].sum()))
                pattern[g.permutation(10)[:nj]]+=2
                nxt=with_schedule(work,data,pattern,True)
            else:
                k,n,b,u=cases[ix]
                nxt=ready(base_window((subsets[b],k,n,u),work))
            elapsed=1000*(time.perf_counter()-tic)
            if it:
                if ix<0:measures[stratum]=elapsed;patterns.append((stratum,pattern.tolist()))
                else:rows[ix]['ms'].append(elapsed)
            del nxt
        if it:out['gate_ms'].append(measures.tolist());out['patterns'].append(sorted(patterns))
        if it%8==0:publish(fs,dest,out)
    run.log({'cycles':80});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

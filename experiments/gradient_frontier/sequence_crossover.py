"""Time complete counterfactual ten-update windows from a fixed native state with on-device stochastic gates."""
import functools
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
from . import debias,fused_attention,online_coin
from .causal_feedback import slim
from .head_to_head import gradient
from .independent_audits import copy_state,forecast,residual
from .native_eval import recover
from .run import ROOT,ready
from .subspace_forecast import native_audit

original_attention=Attention.__call__


def gate_step(state,aux,data,external,policy,p):
    basekey,nextkey=jax.random.split(state.training_key)
    key=jax.random.fold_in(basekey,external)
    c1,c2,select,gk,dk=jax.random.split(key,5)
    primary=jax.random.bernoulli(c1,p);precision=jax.random.bernoulli(c2,.002)
    # Explicit static branch variants keep the expensive exact native CE out of the common path.
    def branch(tag):
        def body(_):
            chosen=jax.random.permutation(select,128) if tag else None
            a=tuple(q[chosen[:32]] for q in data) if tag&1 else None
            delta=None
            if tag&2:
                ext=tuple(q[chosen[32:64]] for q in data)
                Attention.__call__=original_attention
                try:gold=native_audit(state.model,ext,32)
                finally:fused_attention.install()
                delta=residual(state,ext,dk,12288,64,gold)
            if policy=='ridge':
                nxt,aa,bb=online_coin.step(state,a,delta,gk,12288,64,p,.002,*aux,next_key=nextkey)
                nextaux=(slim(eqx.filter(state.model,eqx.is_inexact_array)),aa,bb)
            else:
                grad=forecast(state,0)
                if a is not None:
                    raw=debias.draw(state,a,gk,12288,64,61,0,j.arange(1))
                    grad=jax.tree.map(lambda x:x/p,raw)
                if delta is not None:grad=jax.tree.map(lambda x,d:x+d/.002,grad,delta)
                nxt=state.take_step(grad,key=nextkey)[0];nextaux=aux
            return eqx.filter((nxt,nextaux),eqx.is_array)
        return body
    event=primary.astype(j.int32)+2*precision.astype(j.int32)
    return jax.lax.switch(event,[branch(i) for i in range(4)],None),event


@functools.partial(eqx.filter_jit,donate='all-except-first')
def kernel(inputs,carry):
    data,externals,policy,k,n,b,p=inputs
    arrays,static=eqx.partition(carry,eqx.is_array)
    def step(arr,obs):
        state,aux=eqx.combine(arr,static)
        batch,external=obs
        if policy in ('zero','ridge'):return gate_step(state,aux,batch,external,policy,p)
        key,nextkey=jax.random.split(state.training_key)
        grad=gradient(state,batch,jax.random.fold_in(key,external),j.arange(1),61,k,n)
        nextstate=state.take_step(grad,key=nextkey)[0]
        return eqx.filter((nextstate,aux),eqx.is_array),j.array(0,j.int32)
    arrays,events=jax.lax.scan(step,arrays,(data,externals))
    return eqx.combine(arrays,static),events


def main(step,band,run_seed):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(step,fs)
    fused_attention.install()
    file=Path('/tmp/sequence-cross.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(file))
    with np.load(file) as z:pool=tuple(z[q] for q in ('tokens','weights','segments'))
    dev=jax.devices()[0].device_kind
    seed=598200+step+run_seed
    gen=np.random.default_rng(seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfg=dict(step=step,band=band,seed=seed,rep=run_seed,checkpoint=src,holdout='native-confirm3',device=dev,length=10)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='device-ten-update-crossover',id=os.urandom(5).hex(),config=cfg)
    cases=[('zero',12288,64,32,.2),('ridge',12288,64,32,.2)]
    sizes=(1,2,4,8,16,32) if band==0 else (2,4,8,12,24,48)
    cases.extend(('base',k,n,b,1.) for k,n in ((4096,32),(8192,32),(12288,64)) for b in sizes)
    if band:cases.extend((policy,12288,64,32,p) for policy in ('zero','ridge') for p in (.35,.5))
    prev=state if step==4001 else recover(step-1,fs)[0]
    history=json.loads(fs.cat_file(ROOT+'/causal-prediction/feedback.json'))
    aux=ready(online_coin.prepare(state,prev,history,step))
    rows=[dict(policy=m,k=k,n=n,b=b,p=p,ms=[],events=[]) for m,k,n,b,p in cases]
    out=dict(**cfg,wandb=run.url,rows=rows,cold=[],native_step_verified=False)
    dest=ROOT+'/sequence-crossover/'+f'{dev.split()[1]}-s{step}-band{band}-r{run_seed}.json'
    for i in range(97):
        ids=gen.integers(len(pool[0]),size=(10,128))
        data=ready(tuple(j.asarray(v[ids]) for v in pool))
        external=ready(j.asarray(gen.integers(0,2**31-1,size=10,dtype=np.int32)))
        batches=ready({b:tuple(x[:,:b] for x in data) for b in sizes})
        for ix in gen.permutation(len(rows)):
            m,k,n,b,p=cases[ix]
            work=copy_state((state,aux if m=='ridge' else None))
            inputs=(data if m!='base' else batches[b],external,m,k,n,b,p)
            tic=time.perf_counter();final,events=ready(kernel(inputs,work));ms=1000*(time.perf_counter()-tic)
            assert int(final[0].step)==step+10
            if i:
                rows[ix]['ms'].append(ms);rows[ix]['events'].append(np.asarray(events).tolist())
            else:out['cold'].append(dict(index=int(ix),ms=ms))
            del final
        if i%8==0:
            out['native_step_verified']=True;publish(fs,dest,out)
    run.log({'cycles':96});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

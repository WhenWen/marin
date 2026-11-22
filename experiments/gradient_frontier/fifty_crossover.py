"""Randomized same-device crossovers of actual fifty-update MuonW trajectories, with joint durable commits."""
import dataclasses
import functools
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.checkpoint import load_checkpoint,save_checkpoint
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention,strict_crossfit
from .group_regression import fit
from .independent_audits import copy_state
from .native_eval import objective,recover
from .run import ROOT,ready
from .subspace_forecast import native_audit
from .unbiased_check import native

ORIGINAL=Attention.__call__
START=4001
END=4051


def configurations(panel):
    # Independent curves on each physical board. Each batch uses the same ordered original training stream.
    if panel==0:base=((61,4096,32),(61,8192,64),(61,12288,64))
    elif panel==1:base=((7,16384,64),(62,16384,64),(61,8192,32))
    else:raise ValueError(panel)
    return [(172,4096,128,32,32)]+[(m,k,n,b,0) for m,k,n in base for b in (8,16,32,48,64)]


def name(c):return '-'.join(map(str,c))


def read_np(fs,path,check=None):
    raw=fs.cat_file(path)
    if check:assert hashlib.sha256(raw).hexdigest()==check
    with np.load(io.BytesIO(raw)) as z:return tuple(j.asarray(z[k]) for k in ('tokens','weights','segments'))


def native_call(f,*args):
    Attention.__call__=ORIGINAL
    try:return f(*args)
    finally:fused_attention.install()


@eqx.filter_jit
def eval_ce(model,data):return native(model,data)


@functools.partial(eqx.filter_jit,donate='all-except-first')
def baseline(args,state):
    data,key,nextkey,m,k,n=args
    g=eqx.filter_grad(objective)(state.model,data,j.arange(1),key,(100000*m+k,n))
    return state.take_step(g,key=nextkey)[0]


def estimator(state,data,key,bank,c,forced=None):
    _,k,n,b,den=c;T=data[0].shape[1];length=T//n
    basekey,phasekey,coin=jax.random.split(key,3)
    offsets=strict_crossfit.offsets(basekey,b,n,T)
    phase=jax.random.randint(phasekey,(),0,length)
    regular=strict_crossfit.draw(state.model,data,basekey,k,n,bank,offsets,phase)
    hit=bool(jax.random.bernoulli(coin,1./den)) if forced is None else forced
    delta=None
    if hit:
        gold=native_call(native_audit,state.model,data,b)
        total=strict_crossfit.draw(state.model,data,basekey,k,n,bank,offsets,j.asarray(0))
        for i in range(1,length):
            total=strict_crossfit.add(total,strict_crossfit.draw(state.model,data,basekey,k,n,bank,offsets,j.asarray(i)))
        delta=strict_crossfit.subtract_mean(gold,total,length)
    return regular,delta,hit


def update(state,data,bank,c,seed,forced=None):
    m,k,n,b,den=c;rawkey,nextkey=jax.random.split(state.training_key);key=jax.random.fold_in(rawkey,seed)
    if m!=172:return baseline((data,key,nextkey,m,k,n),state),False
    g,d,hit=estimator(state,data,key,bank,c,forced)
    return strict_crossfit.advance((g,d,1./den,nextkey),state),hit


def digest(state):
    hsh=hashlib.sha256()
    for path,v in jax.tree_util.tree_leaves_with_path(state.saveable_state):
        a=np.asarray(v);hsh.update(str(path).encode());hsh.update(str(a.shape).encode());hsh.update(str(a.dtype).encode());hsh.update(a.tobytes())
    return hsh.hexdigest()


def train(mode,seed,panel):
    configure_storage();fs=fsspec.filesystem('s3');seed=int(seed);panel=int(panel)
    device=jax.devices()[0].device_kind;gpu_id=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    root=ROOT+f'/fifty-crossover-v1/{device.split()[1]}-p{panel}-s{seed}'
    uid=uuid.uuid4().hex[:12];cfgs=configurations(panel)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    def read(path):return json.loads(fs.cat_file(path))
    def write(path,x):fs.pipe(path,json.dumps(x,allow_nan=False).encode())
    if mode=='logging':
        r=wandb.init(entity='marin-community',project='gradient-variance-frontier',id=uid,mode='online',config=dict(seed=seed,panel=panel,phase=mode))
        r.log({'preflight/logging':1});r.finish();write(root+'/logging.json',dict(run_id=uid));return
    assert fs.exists(root+'/logging.json')
    if mode=='main':assert read(root+'/verified.json')['bitwise_match']
    stage='replay' if mode in ('check','verify') else 'main';prefix=root+'/'+stage
    initial,src=recover(START,fs);fused_attention.install()
    manifest=read(ROOT+'/fifty-stream/manifest.json');assert len(manifest['files'])==50
    files={v['step']:v for v in manifest['files']}
    def load_data(step):
        rec=files[step];return read_np(fs,rec['path'],rec['archive_sha256'])
    # The fixed four validation documents are not used in the variance analysis or calibration.
    valid=tuple(a[:4] for a in read_np(fs,ROOT+'/inputs/native-confirm2.npz'))
    entries=sorted(fs.glob(prefix+'/commits/*.json'));last=read(entries[-1]) if entries else None
    start=last['next_step'] if last else START
    if mode=='verify':last=read(prefix+'/commits/4002.json');start=4002
    end=4003 if mode in ('check','verify') else END
    if start>=end:return
    if not last and fs.exists(prefix+'/started.json'):
        # Retrying before any training checkpoint is safe only from the recorded origin.
        assert read(prefix+'/started.json')['origin']==src
    write(prefix+'/started.json',dict(origin=src,start=START))
    def restore(c):
        if last is None:return copy_state(initial)
        p=last['checkpoints'][name(c)]
        md=read(p.removeprefix('s3://')+'/metadata.json')
        assert md['config']==list(c) and md['next_step']==start and md['seed']==seed and md['data_position']==128*start
        s=load_checkpoint(eqx.filter_eval_shape(lambda: initial),p,axis_mapping={},mesh=jax.sharding.Mesh(np.asarray(jax.devices()).reshape(-1,1),('data','model')),allow_partial=False)
        assert int(s.step)==int(s.opt_state.count)==start and int(s.opt_state.hyperparams_states['learning_rate'].count)==start
        return s
    states={c:restore(c) for c in cfgs}
    if mode=='verify' and last is None:raise AssertionError('Missing required checkpoint')
    bank_path=root+'/bank.json'
    if fs.exists(bank_path):
        bank_meta=read(bank_path);raw=fs.cat_file(bank_meta['path']);assert hashlib.sha256(raw).hexdigest()==bank_meta['sha256']
        with np.load(io.BytesIO(raw)) as z:
            bank=tuple(jax.lax.bitcast_convert_type(j.asarray(z[f'a{i}']),j.bfloat16) if i>=3 else j.asarray(z[f'a{i}']) for i in range(5))
    else:
        assert not last and mode=='check'
        cal=tuple(a[:4] for a in read_np(fs,ROOT+'/inputs/native-packed.npz'))
        tic=time.perf_counter();bank=ready(fit(initial.model,cal,4096,113));cold=(time.perf_counter()-tic)*1000
        tic=time.perf_counter();bank=ready(fit(initial.model,cal,4096,113));prep=(time.perf_counter()-tic)*1000
        buf=io.BytesIO();np.savez(buf,**{f'a{i}':np.asarray(v).view(np.uint16) if i>=3 else np.asarray(v) for i,v in enumerate(bank)})
        raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest();path=root+'/banks/'+sha+'.npz';fs.pipe(path,raw)
        bank_meta=dict(path=path,sha256=sha,origin=src,prep_ms=prep,cold_prep_ms=cold);write(bank_path,bank_meta)
    if last:assert last['bank']==bank_meta['sha256']
    runs={};history=[];times={}
    for c in cfgs:
        cfg=dict(config=c,seed=seed,phase=mode,origin=src,previous_commit=None if not last else last['checkpoints'][name(c)],restored_next_step=start,trainer='MuonW',board=gpu_id,device=device,calibration=bank_meta,planned_steps=50)
        r=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fifty-actual-crossover-'+mode,id=uuid.uuid4().hex[:12],
                     mode='online',reinit='create_new',name=f'{device.split()[1]}-{name(c)}-s{seed}-{mode}',config=cfg);runs[c]=r
    inputs=load_data(start)
    for c in cfgs:
        data=tuple(v[:c[3]] for v in inputs);tic=time.perf_counter()
        for hit in ((False,True) if c[0]==172 else (False,)):
            work=copy_state(states[c]);tmp,_=update(work,data,bank,c,seed,hit);ready(tmp);del tmp
        times[name(c)]=1000*(time.perf_counter()-tic)
    write(prefix+'/attempts/'+uid+'/lineage.json',dict(mode=mode,start=start,board=gpu_id,origin=src,restore=None if not last else last['checkpoints'],compile_warmup_ms=times,wandb={name(c):r.url for c,r in runs.items()}))
    for step in range(start,end):
        data=load_data(step);by_b=ready({b:tuple(v[:b] for v in data) for b in {c[3] for c in cfgs}})
        order=np.random.default_rng(np.random.SeedSequence([seed,step])).permutation(len(cfgs));batch_rows=[]
        for index in order:
            c=cfgs[int(index)];batch=by_b[c[3]];state=states[c]
            train_ce=float(native_call(eval_ce,state.model,batch))
            tic=time.perf_counter();nxt,hit=update(state,batch,bank,c,seed);ready(nxt);ms=1000*(time.perf_counter()-tic)
            assert int(nxt.step)==int(nxt.opt_state.count)==step+1
            val=float(native_call(eval_ce,nxt.model,valid));states[c]=nxt
            row=dict(config=c,step=step,ms=ms,precision_audit=hit,train_ce=train_ce,validation_ce=val,lr=float(nxt.opt_state.hyperparams['learning_rate']),
                     adam_lr=float(nxt.opt_state.hyperparams['adam_lr']),data_sha256=files[step]['data_sha256'],board=gpu_id,order=int(np.where(order==index)[0][0]),wandb=runs[c].url)
            runs[c].log({'training/ce':train_ce,'validation/ce':val,'time/update_ms':ms,'precision_audit':int(hit)},step=step)
            batch_rows.append(row)
        history.extend(batch_rows)
        with Path('/tmp',uid+'-loss.jsonl').open('a') as fd:
            for row in batch_rows:fd.write(json.dumps(row)+'\n')
        write(prefix+f'/attempts/{uid}/steps/{step}.json',batch_rows)
        if mode!='main' or (step+1-START)%5==0 or step+1==end:
            cp={};digests={}
            for c in cfgs:
                path=f's3://{prefix}/states/{uid}/{name(c)}/step-{step}'
                md=dict(next_step=step+1,data_position=(step+1)*128,seed=seed,config=list(c),origin=src,bank_sha256=bank_meta['sha256'])
                save_checkpoint(states[c].saveable_state,step,path,is_temporary=False,metadata=md)
                assert read(path.removeprefix('s3://')+'/metadata.json')['next_step']==step+1
                cp[name(c)]=path
                if stage=='replay':digests[name(c)]=digest(states[c])
            receipt=dict(next_step=step+1,checkpoints=cp,digests=digests,bank=bank_meta['sha256'],attempt=uid,board=gpu_id,rows=history)
            if mode=='verify':
                prior=read(prefix+'/commits/4003.json');assert cp!=prior['checkpoints'] and digests==prior['digests']
                extract=lambda rr:{tuple(v['config']):(v['data_sha256'],v['precision_audit'],v['train_ce'],v['validation_ce'],v['lr'],v['adam_lr']) for v in rr['rows'] if v['step']==4002}
                assert extract(receipt)==extract(prior)
                write(root+'/verified.json',dict(bitwise_match=True,original=prior['attempt'],replay=uid,digests=digests,restored_from=last['checkpoints']))
            else:write(prefix+f'/commits/{step+1}.json',receipt)
            history=[]
        print('CROSSOVER STEP',mode,step,'arms',len(batch_rows),flush=True)
    for r in runs.values():r.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        train(*sys.argv[1:])

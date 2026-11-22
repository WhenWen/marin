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
    if panel==13:return [(m,12288,64,32,500) for m in (402,420,422)]+configurations(6)[1:]
    if panel==10:return configurations(6)
    if panel==12:return [(408,12288,64,32,500),*configurations(6)[1:]]
    if panel==8:
        return [(406,12288,64,32,500),(61,1024,16,1,0),(62,16384,64,2,0),(61,12288,64,4,0),(62,16384,64,8,0),(61,12288,64,12,0)]
    if panel==7:
        return [(m,12288,64,32,500) for m in (411,412,402,413,414)]+[(61,1024,16,1,0),(62,16384,64,1,0),(62,16384,64,2,0),(61,12288,64,4,0),(7,16384,64,6,0),(62,16384,64,8,0),(61,12288,64,12,0)]
    if panel==6:
        # Secondary timing pilot; the primary panels retain the full preregistered envelope.
        return [(402,12288,64,32,500),(61,1024,16,1,0),(62,16384,64,2,0),(61,12288,64,4,0),(62,16384,64,8,0)]
    if panel in (2,3,4,5):
        primary=(402 if panel>=4 else 401,12288,64,32,500)
        if panel in (2,4):
            candidates=((61,12288,64),(62,16384,64),(7,16384,64))
            extra=((61,1024,16,1,0),(61,4096,32,1,0),(61,8192,32,2,0),(61,4096,64,2,0))
        else:
            candidates=((61,256,8),(61,4096,32),(61,8192,32))
            extra=((61,64,4,1,0),(61,1024,16,2,0),(61,12288,64,4,0),(62,16384,64,8,0))
        return [primary]+[(m,k,n,b,0) for m,k,n in candidates for b in (1,2,3,4,6,8,12)]+list(extra)
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
    from . import fifty_blocks,fifty_blocks_strict,block_rate_grid,prepared_blocks,samplewise_blocks,muon_safe_prediction
    initial,src=recover(START,fs);fused_attention.install()
    block_panel=panel in (2,3,4,5,6,7,8,10,12,13)
    block_impl=fifty_blocks_strict if panel>=4 else fifty_blocks
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
    if block_panel:
        schedule,sched_ms=fifty_blocks.schedule(seed)
        sha=hashlib.sha256(b''.join(bytes([tag,hit])+np.asarray(idx).tobytes() for tag,hit,idx in schedule)).hexdigest()
        bank=None;bank_meta=dict(sha256=sha,origin=src,prep_ms=sched_ms,cold_prep_ms=sched_ms)
        schedules={402:schedule,401:schedule,408:schedule}
        if panel==13:
            schedules.update({420:schedule,422:schedule})
            if fs.exists(bank_path):
                saved=read(bank_path);blob=fs.cat_file(saved['coefficients'])
                assert hashlib.sha256(blob).hexdigest()==saved['coef_sha']
                bank=tuple(j.asarray(v,j.float32) for v in json.loads(blob))
                fit_ms=saved['fit_ms'];coef_sha=saved['coef_sha']
            else:
                assert mode=='check' and not last
                cal=read_np(fs,ROOT+'/inputs/native-packed.npz')
                a,b=ready([tuple(v[n:n+32] for v in cal) for n in (0,32)])
                bank=ready(muon_safe_prediction.calibrate(initial,a,b))
                tic=time.perf_counter();bank=ready(muon_safe_prediction.calibrate(initial,a,b));fit_ms=1000*(time.perf_counter()-tic)
                blob=json.dumps([float(v) for v in bank],allow_nan=False).encode();coef_sha=hashlib.sha256(blob).hexdigest()
                fs.pipe(root+'/banks/'+coef_sha+'.json',blob)
            sha=hashlib.sha256((sha+coef_sha).encode()).hexdigest()
            bank_meta.update(sha256=sha,coefficients=root+'/banks/'+coef_sha+'.json',coef_sha=coef_sha,fit_ms=fit_ms,
                             prep_by_method={402:sched_ms,420:sched_ms,422:sched_ms+fit_ms})
        if panel==8:
            slots,phases,prep_ms=prepared_blocks.prepare(seed)
            host=[np.stack([np.asarray(plan[k]) for plan in slots]) for k in range(5)]
            prep_sha=hashlib.sha256(b''.join(v.tobytes() for v in host)).hexdigest()
            sha=hashlib.sha256((sha+prep_sha).encode()).hexdigest()
            target=root+'/banks/'+prep_sha+'.npz'
            if fs.exists(bank_path):
                saved=read(bank_path)
                assert saved['sha256']==sha and saved['slot_path']==target
                with np.load(io.BytesIO(fs.cat_file(target))) as restored:
                    assert all(np.array_equal(a,restored[f'a{k}']) for k,a in enumerate(host))
            else:
                buffer=io.BytesIO()
                np.savez(buffer,**{f'a{k}':a for k,a in enumerate(host)})
                fs.pipe(target,buffer.getvalue())
            bank_meta.update(sha256=sha,slot_path=target,prep_ms=prep_ms+sched_ms,cold_prep_ms=prep_ms+sched_ms)
            schedules[406]=[(tag,hit,slot) for (tag,hit,_),slot in zip(schedule,slots)]
        if panel==7:
            hashes={402:sha};costs={402:sched_ms}
            for method,rates in block_rate_grid.RATES.items():
                if method==402:continue
                ss,ms=fifty_blocks.schedule(seed,*rates)
                schedules[method]=ss;costs[method]=ms
                hashes[method]=hashlib.sha256(b''.join(bytes([tag,hit])+np.asarray(idx).tobytes() for tag,hit,idx in ss)).hexdigest()
            sha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
            bank_meta.update(sha256=sha,prep_by_method=costs,hashes=hashes)
        if fs.exists(bank_path):
            saved=read(bank_path);assert saved['sha256']==sha;bank_meta=saved
        else:write(bank_path,bank_meta)
    elif fs.exists(bank_path):
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
    def block_update(c,state,data,gate):
        if c[0] in (420,422):return muon_safe_prediction.update(state,data,gate,seed,bank,8 if c[0]==420 else -1)
        if c[0]==408:return fifty_blocks_strict.update(state,data,gate,seed,samplewise_blocks.discrepancy)
        if c[0]==406:
            part,hit,slot=gate
            return prepared_blocks.update(state,data,slot,phases,part,hit)
        if c[0] in (401,402):return block_impl.update(state,data,gate,seed)
        return block_rate_grid.update(state,data,gate,seed,c[0])
    def control_update(state,data,c,forced=None):
        if panel==8:
            m,k,n,_,_=c
            return prepared_blocks.baseline((data,j.asarray(seed,j.uint32),m,k,n),state),False
        return update(state,data,bank,c,seed,forced)
    def is_block(c):return c[0] in (401,406,408,420,422) or c[0] in block_rate_grid.RATES
    inputs=load_data(start)
    for c in (() if mode=="verify" else cfgs):
        data=tuple(v[:c[3]] for v in inputs);tic=time.perf_counter()
        if is_block(c):
            for tag in range(3):
                for hit in (False,True):
                    tmp,_=block_update(c,copy_state(states[c]),inputs,(tag,hit,schedules[c[0]][0][2]));ready(tmp);del tmp
        else:
            for hit in ((False,True) if c[0]==172 else (False,)):
                work=copy_state(states[c]);tmp,_=control_update(work,data,c,hit);ready(tmp);del tmp
        times[name(c)]=1000*(time.perf_counter()-tic)
    write(prefix+'/attempts/'+uid+'/lineage.json',dict(mode=mode,start=start,board=gpu_id,origin=src,restore=None if not last else last['checkpoints'],compile_warmup_ms=times,wandb={name(c):r.url for c,r in runs.items()}))
    for step in range(start,end):
        staging_start=time.perf_counter()
        data=load_data(step);by_b=ready({b:tuple(v[:b] for v in data) for b in {c[3] for c in cfgs}})
        shared_staging_ms=1000*(time.perf_counter()-staging_start)
        order=np.random.default_rng(np.random.SeedSequence([seed,step])).permutation(len(cfgs));batch_rows=[]
        for index in order:
            c=cfgs[int(index)];batch=by_b[c[3]];state=states[c]
            ce_start=time.perf_counter()
            train_ce=float(native_call(eval_ce,state.model,batch))
            training_metric_ms=1000*(time.perf_counter()-ce_start)
            tag=0;tic=time.perf_counter()
            if is_block(c):
                gate=schedules[c[0]][step-START]
                if mode in ('check','verify') and step==4002:gate=(2,True,gate[2])
                tag=gate[0];nxt,hit=block_update(c,state,data,gate)
            else:nxt,hit=control_update(state,batch,c)
            ready(nxt);ms=1000*(time.perf_counter()-tic)
            assert int(nxt.step)==int(nxt.opt_state.count)==step+1
            val_start=time.perf_counter()
            val=float(native_call(eval_ce,nxt.model,valid));states[c]=nxt
            validation_metric_ms=1000*(time.perf_counter()-val_start)
            row=dict(config=c,step=step,ms=ms,shared_staging_ms=shared_staging_ms,training_metric_ms=training_metric_ms,validation_metric_ms=validation_metric_ms,precision_audit=hit,block_draw=tag,train_ce=train_ce,validation_ce=val,lr=float(nxt.opt_state.hyperparams['learning_rate']),
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
                extract=lambda rr:{tuple(v['config']):(v['data_sha256'],v['precision_audit'],v.get('block_draw',0),v['train_ce'],v['validation_ce'],v['lr'],v['adam_lr']) for v in rr['rows'] if v['step']==4002}
                assert extract(receipt)==extract(prior)
                proof=dict(bitwise_match=True,original=prior['attempt'],replay=uid,digests=digests,restored_from=last['checkpoints'])
                write(prefix+'/attempts/'+uid+'/verified.json',proof)
                write(root+'/verified.json',proof)
            else:write(prefix+f'/commits/{step+1}.json',receipt)
            history=[]
        print('CROSSOVER STEP',mode,step,'arms',len(batch_rows),flush=True)
    for r in runs.values():r.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        train(*sys.argv[1:])

"""A ten-update read-only fork of the original MuonW training trajectory."""
import dataclasses,hashlib,json,os,sys,uuid,time
from pathlib import Path
import draccus,equinox as eqx,fsspec,haliax as h,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import RunSpec,configuration
from experiments.hybrid_transfer.recovery import scientific_identity
from levanter.checkpoint import save_checkpoint
from levanter.main.train_lm import main as train_lm
from levanter.trainer import Trainer
from levanter.tracker.wandb import WandbConfig
from rigging.secrets import require_secret
from .run import ROOT
from .multifidelity import objective

ORIGIN='s3://marin-us-east-04a/marin/experiments/hybrid-transfer-h100-20261116/state/standard-muonw-w128-d4-lr0.02-s0/checkpoints/fcf97d48183c/step-4000'
TARGET=4011

def trajectory_prefix(method,width=128,depth=4):
    base=ROOT+('/native-muonw' if method=='native' else '/muonw-'+method)
    return base if (width,depth)==(128,4) else base+f'-w{width}-d{depth}'


def main(mode,method="native",width=128,depth=4):
    configure_storage();fs=fsspec.filesystem('s3');run_id=uuid.uuid4().hex[:12]
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    width,depth=int(width),int(depth)
    prefix=trajectory_prefix(method,width,depth)
    origin=ORIGIN
    if (width,depth)!=(128,4):
        assert method=='native'
        parent=ORIGIN.split('/state/')[0]+'/state/'+f'standard-muonw-w{width}-d{depth}-lr0.02-s0/checkpoints'
        matches=sorted(fs.glob(parent.removeprefix('s3://')+'/*/step-4000/metadata.json'))
        assert len(matches)==1, 'Midpoint checkpoint must be unique'
        origin='s3://'+matches[0].removesuffix('/metadata.json')
    def read(p):return json.loads(fs.cat_file(p))
    def publish(p,x):fs.pipe(p,json.dumps(x,allow_nan=False).encode())
    tracker=WandbConfig(entity='marin-community',project='gradient-variance-frontier',group='MuonW-10-'+method+f'-w{width}-d{depth}',
        name=f'{mode}-{run_id}',id=run_id,mode='online',resume='never',background=False,save_code=False)
    if mode=='logging':
        tr=tracker.init(run_id);tr.log_hyperparameters({'phase':'non-training logging check','origin':origin})
        tr.log({'preflight/logging':1},step=0);tr.finish()
        publish(prefix+'/logging/'+run_id+'.json',{'id':run_id,'status':'passed'});return
    assert fs.glob(prefix+'/logging/*.json')
    lineage=read(origin[len('s3://'):].rsplit('/',1)[0]+'/identity.json')
    spec=RunSpec(width,depth,'standard','muonw',.02,0)
    cfg=configuration(spec,run_id,'s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    assert scientific_identity(draccus.encode(cfg))==lineage['scientific_config']
    if mode=='verify':
        receipt=read(prefix+'/completed/4002.json');src=receipt['checkpoint'];expected=4002;end=4003
    else:
        records=sorted(fs.glob(prefix+'/completed/*.json'))
        if records:
            last=read(records[-1]);src=last['checkpoint'];expected=last['next_step']
        else:src=origin;expected=4001
        end=TARGET
    if mode=='check':end=4003
    if expected>=end:return
    assert read(src[len('s3://'):]+'/metadata.json')['step']==expected-1
    cfg=dataclasses.replace(cfg,trainer=dataclasses.replace(cfg.trainer,
        load_checkpoint=True,load_checkpoint_path=src,allow_partial_checkpoint=False,
        checkpointer=dataclasses.replace(cfg.trainer.checkpointer,base_path='s3://'+prefix+'/automatic'),
        tracker=(tracker,*cfg.trainer.tracker[1:])))
    folder=Path('/tmp/frontier-native');folder.mkdir(exist_ok=True)
    fs.get_file(ROOT+'/inputs/windows-confirm.npz',str(folder/'holdout.npz'))
    with np.load(folder/'holdout.npz') as f:val=j.asarray(f['tokens'][:4])
    validation=eqx.filter_jit(lambda m:objective(m,val,j.zeros(1,j.int32),4,'full',1.))
    if method!='native':
        from .native_eval import objective as packed_objective
        from .multifidelity import shortlist
        freq,n,variant=map(int,method.split('-'))
        learned_bank=variant in range(110,118) or variant in range(170,178)
        if learned_bank:
            from .fused_attention import install
            from .native_eval import recover
            if variant in range(170,178):from .online_adjoint import fit,levels
            else:from .group_regression import fit,levels
            install()
            path=prefix+'/group-bank';dest=folder/'learned.npz'
            if fs.exists(path+'.json'):
                manifest=read(path+'.json')
                assert manifest['source']==origin and manifest['method']==method
                fs.get_file(manifest['bytes'],str(dest))
                assert hashlib.sha256(dest.read_bytes()).hexdigest()==manifest['sha256']
                def recover_leaf(arr,i):
                    if arr.dtype.kind=='V':
                        assert i in (3,4) and arr.dtype.itemsize==2
                        return jax.lax.bitcast_convert_type(j.asarray(arr.view(np.uint16)),j.bfloat16)
                    return j.asarray(arr)
                with np.load(dest) as saved:
                    ids=tuple(recover_leaf(saved[f'v{i}'],i) for i in range(manifest['arrays']))
            else:
                assert expected==4001,'A resumed learned estimator requires its persisted calibration'
                fs.get_file(ROOT+'/inputs/native-packed.npz',str(folder/'packed.npz'))
                with np.load(folder/'packed.npz') as data:ref=tuple(j.asarray(data[a][:levels(variant)[0]]) for a in ('tokens','weights','segments'))
                initial,_=recover(4001,fs);ids=fit(initial.model,ref,freq,variant)
                np.savez(dest,**{f'v{i}':np.asarray(x) for i,x in enumerate(ids)})
                sha=hashlib.sha256(dest.read_bytes()).hexdigest();blob=path+'-'+sha+'.npz'
                fs.put_file(str(dest),blob)
                publish(path+'.json',dict(method=method,source=origin,bytes=blob,sha256=sha,arrays=len(ids)))
        elif variant in (5,6,7) or variant in range(60,66):
            from .fused_attention import install
            install()
            ids=j.arange(freq)
        else:
            fs.get_file(ROOT+'/inputs/native-packed.npz',str(folder/'packed.npz'))
            with np.load(folder/'packed.npz') as packed:
                ids=shortlist(j.asarray(packed['tokens'][:512]),freq,128256)
        if not learned_bank:
            raw=np.asarray(ids).astype(np.int32).tobytes()
            dictionary=prefix+'/dictionary.bin'
            if fs.exists(dictionary):assert fs.cat_file(dictionary)==raw
            else:fs.pipe(dictionary,raw)
        original_init=Trainer.__init__
        def init(self,trainer_cfg,optimizer,original,**kwargs):
            def estimate(model,example,*,key=None):
                if key is None:return original(model,example,key=key)
                def arr(v):return v.rearrange(('batch','position')).array
                data=tuple(arr(v) for v in (example.tokens,example.loss_weight,example.attn_mask.segment_ids[0]))
                return packed_objective(model,data,ids,key,(100000*variant+freq,n))
            original_init(self,trainer_cfg,optimizer,estimate,**kwargs)
        Trainer.__init__=init
    def follow(trainer,state,loader):
        assert int(state.step)==int(state.opt_state.count)==expected
        assert int(state.opt_state.hyperparams_states['learning_rate'].count)==expected
        assert int(state.opt_state.hyperparams_states['adam_lr'].count)==expected
        track=dict(parent=src,restored_next_step=expected,optimizer='muonw',gradient_estimator=method,schedule_steps=cfg.trainer.num_train_steps,width=width,depth=depth,
                   data_example_offset=expected*128,run_id=run_id,mode=mode,
                   wandb=f'https://wandb.ai/marin-community/gradient-variance-frontier/runs/{run_id}')
        publish(prefix+'/attempts/'+run_id+'/lineage.json',track)
        trainer.tracker.log_hyperparameters(track)
        print('VERIFIED FULL MUONW RESTORE',track,flush=True)
        loader=iter(loader)
        for step in range(expected,end):
            batch=next(loader)
            # A data fingerprint makes a new-process replay verifiable independently of loss equality.
            data_hash=hashlib.sha256(np.asarray(batch.tokens.array).tobytes()).hexdigest()
            tick=time.perf_counter()
            info=trainer.train_step(state,batch);state=info.state
            jax.block_until_ready(state);update_ms=(time.perf_counter()-tick)*1000
            n=int(state.step)
            assert n==step+1 and int(state.opt_state.count)==n
            cp=f's3://{prefix}/states/{run_id}/step-{step}'
            valid=float(validation(state.model))
            meta=dict(next_step=n,data_seed=42,next_example=n*128,origin=origin,run_id=run_id,
                      data_sha256=data_hash,optimizer='muonw',trainer_fingerprint=lineage['fingerprint'])
            save_checkpoint(state.saveable_state,step,cp,is_temporary=False,metadata=meta)
            assert read(cp[len('s3://'):]+'/metadata.json')['next_step']==n
            entry=dict(meta,checkpoint=cp,loss=float(info.loss),validation=valid,training_wall_ms=update_ms,
                       lr=float(state.opt_state.hyperparams['learning_rate']),adam_lr=float(state.opt_state.hyperparams['adam_lr']),
                       wandb=track['wandb'])
            with (folder/(run_id+'-loss.jsonl')).open('a') as log:log.write(json.dumps(entry)+'\n')
            tracker_key='completed' if mode in ('main','check') else 'verification'
            publish(prefix+'/'+tracker_key+f'/{n}.json',entry)
            trainer.tracker.log({'trajectory/train_loss':float(info.loss),'trajectory/validation_loss':valid},step=step)
            print('MUONW STATE',n,float(info.loss),valid,flush=True)
        return info
    Trainer.train=follow
    train_lm(cfg)
if __name__=='__main__':main(*sys.argv[1:])

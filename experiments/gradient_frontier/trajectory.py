"""A ten-update read-only fork of the original MuonW training trajectory."""
import dataclasses,hashlib,json,os,sys,uuid
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

def main(mode):
    configure_storage();fs=fsspec.filesystem('s3');run_id=uuid.uuid4().hex[:12]
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    prefix=ROOT+'/native-muonw'
    def read(p):return json.loads(fs.cat_file(p))
    def publish(p,x):fs.pipe(p,json.dumps(x,allow_nan=False).encode())
    tracker=WandbConfig(entity='marin-community',project='gradient-variance-frontier',group='native-MuonW-10',
        name=f'{mode}-{run_id}',id=run_id,mode='online',resume='never',background=False,save_code=False)
    if mode=='logging':
        tr=tracker.init(run_id);tr.log_hyperparameters({'phase':'non-training logging check','origin':ORIGIN})
        tr.log({'preflight/logging':1},step=0);tr.finish()
        publish(prefix+'/logging/'+run_id+'.json',{'id':run_id,'status':'passed'});return
    assert fs.glob(prefix+'/logging/*.json')
    lineage=read(ORIGIN[len('s3://'):].rsplit('/',1)[0]+'/identity.json')
    spec=RunSpec(128,4,'standard','muonw',.02,0)
    cfg=configuration(spec,run_id,'s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    assert scientific_identity(draccus.encode(cfg))==lineage['scientific_config']
    if mode=='verify':
        receipt=read(prefix+'/completed/4002.json');src=receipt['checkpoint'];expected=4002;end=4003
    else:
        records=sorted(fs.glob(prefix+'/completed/*.json'))
        if records:
            last=read(records[-1]);src=last['checkpoint'];expected=last['next_step']
        else:src=ORIGIN;expected=4001
        end=TARGET
    if expected==end:return
    assert read(src[len('s3://'):]+'/metadata.json')['step']==expected-1
    cfg=dataclasses.replace(cfg,trainer=dataclasses.replace(cfg.trainer,
        load_checkpoint=True,load_checkpoint_path=src,allow_partial_checkpoint=False,
        checkpointer=dataclasses.replace(cfg.trainer.checkpointer,base_path='s3://'+prefix+'/automatic'),
        tracker=(tracker,*cfg.trainer.tracker[1:])))
    folder=Path('/tmp/frontier-native');folder.mkdir(exist_ok=True)
    fs.get_file(ROOT+'/inputs/windows-confirm.npz',str(folder/'holdout.npz'))
    with np.load(folder/'holdout.npz') as f:val=j.asarray(f['tokens'][:4])
    validation=eqx.filter_jit(lambda m:objective(m,val,j.zeros(1,j.int32),4,'full',1.))
    def follow(trainer,state,loader):
        assert int(state.step)==int(state.opt_state.count)==expected
        assert int(state.opt_state.hyperparams_states['learning_rate'].count)==expected
        assert int(state.opt_state.hyperparams_states['adam_lr'].count)==expected
        track=dict(parent=src,restored_next_step=expected,optimizer='muonw',schedule_steps=7630,
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
            info=trainer.train_step(state,batch);state=info.state
            n=int(state.step)
            assert n==step+1 and int(state.opt_state.count)==n
            cp=f's3://{prefix}/states/{run_id}/step-{step}'
            valid=float(validation(state.model))
            meta=dict(next_step=n,data_seed=42,next_example=n*128,origin=ORIGIN,run_id=run_id,
                      data_sha256=data_hash,optimizer='muonw',trainer_fingerprint=lineage['fingerprint'])
            save_checkpoint(state.saveable_state,step,cp,is_temporary=False,metadata=meta)
            assert read(cp[len('s3://'):]+'/metadata.json')['next_step']==n
            entry=dict(meta,checkpoint=cp,loss=float(info.loss),validation=valid,
                       lr=float(state.opt_state.hyperparams['learning_rate']),adam_lr=float(state.opt_state.hyperparams['adam_lr']),
                       wandb=track['wandb'])
            with (folder/(run_id+'-loss.jsonl')).open('a') as log:log.write(json.dumps(entry)+'\n')
            tracker_key='completed' if mode=='main' else 'verification'
            publish(prefix+'/'+tracker_key+f'/{n}.json',entry)
            trainer.tracker.log({'trajectory/train_loss':float(info.loss),'trajectory/validation_loss':valid},step=step)
            print('MUONW STATE',n,float(info.loss),valid,flush=True)
        return info
    Trainer.train=follow
    train_lm(cfg)
if __name__=='__main__':main(sys.argv[1])

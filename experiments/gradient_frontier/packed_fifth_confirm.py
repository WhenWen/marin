"""Export independent examples directly from the original packed, shuffled training iterator."""
import asyncio,hashlib,json,os
from pathlib import Path
import fsspec,jax,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import RunSpec,configuration
from .run import ROOT

async def build(cfg):
    ds=cfg.data.train_set(cfg.model.max_Pos,cfg.trainer.batch_schedule,key=jax.random.PRNGKey(cfg.data_seed))
    fs=fsspec.filesystem('s3')
    sources=['native-packed','native-confirm','native-confirm2','native-confirm3','native-confirm4']
    old=[]
    for name in sources:
        with fs.open(ROOT+'/inputs/'+name+'.npz','rb') as fd, np.load(fd) as archive:
            old.extend(archive['starts'].tolist())
    forbidden=np.asarray(old)[:,None]//128+np.arange(-2,3)[None,:]
    valid=np.setdiff1d(np.arange(4032,7600,2),forbidden.ravel())
    rng=np.random.default_rng(837368557)
    starts=128*rng.choice(valid,64,replace=False)
    assert np.min(abs(starts[:,None]-np.asarray(old)[None,:]))>256
    batches=[]
    for start in starts:
        examples=await ds.get_batch(range(int(start),int(start)+64))
        def data(ex):
            ids=ex.attn_mask.segment_ids
            return np.asarray(ex.tokens.array),np.asarray(ex.loss_weight.array),np.asarray(ids[0].array)
        batches.extend(data(e) for e in examples)
    return tuple(np.stack(v) for v in zip(*batches)),starts

def main():
    configure_storage();fs=fsspec.filesystem('s3')
    cfg=configuration(RunSpec(128,4,'standard','muonw',.02,0),'diagnostic','s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    (tokens,weights,segments),starts=asyncio.run(build(cfg))
    out=Path('/tmp/native-confirm5.npz');np.savez(out,tokens=tokens,weights=weights,segments=segments,starts=starts)
    meta=dict(source='native shuffled dataset, seed 42',shape=list(tokens.shape),distinct_blocks=len(starts),start_offsets=starts.tolist(),selection_seed=837368557,
              nonzero_counts=np.unique((weights!=0).sum(-1)).tolist(),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    fs.put_file(str(out),ROOT+'/inputs/native-confirm5.npz')
    fs.pipe(ROOT+'/inputs/native-confirm5.json',json.dumps(meta).encode())
    print('PACKED HOLDOUT',meta,flush=True)
if __name__=='__main__':main()

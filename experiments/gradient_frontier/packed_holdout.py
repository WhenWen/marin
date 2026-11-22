"""Export independent examples directly from the original packed, shuffled training iterator."""
import asyncio,hashlib,json,os
from pathlib import Path
import fsspec,jax,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import RunSpec,configuration
from .run import ROOT

async def build(cfg):
    ds=cfg.data.train_set(cfg.model.max_Pos,cfg.trainer.batch_schedule,key=jax.random.PRNGKey(cfg.data_seed))
    rng=np.random.default_rng(942311)
    # Post-trajectory steps keep the benchmark independent of the ten actual updates.
    starts=128*rng.choice(np.arange(4032,7600,2),64,replace=False)
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
    out=Path('/tmp/native-packed.npz');np.savez(out,tokens=tokens,weights=weights,segments=segments,starts=starts)
    meta=dict(source='native shuffled dataset, seed 42',shape=list(tokens.shape),distinct_blocks=len(starts),
              nonzero_counts=np.unique((weights!=0).sum(-1)).tolist(),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    fs.put_file(str(out),ROOT+'/inputs/native-packed.npz')
    fs.pipe(ROOT+'/inputs/native-packed.json',json.dumps(meta).encode())
    print('PACKED HOLDOUT',meta,flush=True)
if __name__=='__main__':main()

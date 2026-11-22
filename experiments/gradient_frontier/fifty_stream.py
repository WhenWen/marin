"""Reconstruct the fifty original training batches and validate their fingerprints against native training."""
import asyncio
import hashlib
import io
import json
import fsspec
import jax
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import RunSpec,configuration
from .run import ROOT


async def build():
    configure_storage();fs=fsspec.filesystem('s3')
    cfg=configuration(RunSpec(128,4,'standard','muonw',.02,0),'diagnostic','s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    ds=cfg.data.train_set(cfg.model.max_Pos,cfg.trainer.batch_schedule,key=jax.random.PRNGKey(cfg.data_seed))
    files=[]
    for step in range(4001,4051):
        examples=await ds.get_batch(range(step*128,(step+1)*128))
        tokens,weights,segments=(np.stack(v) for v in zip(*[(np.asarray(e.tokens.array),np.asarray(e.loss_weight.array),np.asarray(e.attn_mask.segment_ids[0].array)) for e in examples]))
        h=hashlib.sha256(tokens.tobytes()).hexdigest();p=ROOT+f'/native-muonw/completed/{step+1}.json'
        if fs.exists(p):
            entry=json.loads(fs.cat_file(p));assert entry['data_sha256']==h,(step,h,entry['data_sha256'])
        assert np.all(weights.sum(-1)==1023)
        buf=io.BytesIO();np.savez_compressed(buf,tokens=tokens,weights=weights,segments=segments)
        raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest();path=ROOT+f'/fifty-stream/{step}-{sha}.npz'
        fs.pipe(path,raw);files.append(dict(step=step,data_sha256=h,archive_sha256=sha,path=path))
    fs.pipe(ROOT+'/fifty-stream/manifest.json',json.dumps(dict(seed=cfg.data_seed,batch=128,first=4001,last=4050,files=files)).encode())
    print('Verified stream',len(files),flush=True)


if __name__=='__main__':asyncio.run(build())

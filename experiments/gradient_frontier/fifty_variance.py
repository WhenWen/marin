"""Integrate rare audit coins exactly when estimating raw-gradient variance at fifty common native states."""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_crossover import configurations,estimator,read_np
from .group_regression import fit
from .native_eval import recover
from .run import ROOT,ready


@eqx.filter_jit
def moments(ha,da,hb,db):
    norm=lambda grad:sum(j.vdot(v,v) for v in jax.tree.leaves(grad))
    diff=jax.tree.map(lambda a,d,b,e:(a-b)+(d-e),ha,da,hb,db)
    return j.stack((norm(diff)/2,(norm(da)+norm(db))/2))


def main(block,rep):
    configure_storage();fs=fsspec.filesystem('s3');block=int(block);rep=int(rep)
    pool_name='native-confirm3'
    path=ROOT+'/inputs/'+pool_name
    meta=json.loads(fs.cat_file(path+'.json'));raw=fs.cat_file(path+'.npz');assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    with np.load(io.BytesIO(raw)) as z:
        starts=z['starts'];width=64
        mask=(starts+width<=128*(4001-2))|(starts>=128*(4051+2))
        use=np.repeat(mask,width);assert len(use)==len(z['tokens'])
        pool=tuple(z[a][use] for a in ('tokens','weights','segments'))
    assert np.all(pool[1].sum(-1)==1023)
    initial,_=recover(4001,fs);fused_attention.install()
    cal=tuple(a[:4] for a in read_np(fs,ROOT+'/inputs/native-packed.npz'))
    bank=ready(fit(initial.model,cal,4096,113));dummy=j.arange(1)
    combinations=sorted(set(configurations(0)+configurations(1)));main_cfg=next(c for c in combinations if c[0]==172)
    baselines=[c for c in combinations if c[0]!=172]
    device=jax.devices()[0].device_kind;gpu_id=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg=dict(block=block,rep=rep,board=gpu_id,device=device,holdout=meta['sha256'],sequences=len(pool[0]),excluded_blocks=starts[~mask].tolist(),r=1/main_cfg[-1],
             method='paired rare-event conditional variance',phase='discovery')
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fifty-state-variance',id=os.urandom(6).hex(),config=cfg)
    sizes=sorted({c[3] for c in combinations});candidate_B=main_cfg[3]
    for step in range(4001+5*block,4006+5*block):
        dst=ROOT+f'/fifty-variance-v1/{device.split()[1]}-s{step}-r{rep}.json'
        if fs.exists(dst) and json.loads(fs.cat_file(dst)).get('complete'):continue
        state,src=recover(step,fs);out=dict(**cfg,step=step,checkpoint=src,wandb=run.url,components=[],rows=[dict(config=c,pairs=[]) for c in baselines],complete=False)
        gen=np.random.default_rng(np.random.SeedSequence([836172,step,rep]))
        for i in range(16):
            arrays=[]
            for _ in range(2):
                ids=gen.integers(len(pool[0]),size=max(sizes))
                arrays.append({b:tuple(j.asarray(v[ids[:b]]) for v in pool) for b in sizes})
            a,b=arrays;key=jax.random.fold_in(jax.random.key(rep+713598),step*100+i);k1,k2=jax.random.split(key)
            ha,da,_=estimator(state,a[candidate_B],k1,bank,main_cfg,True)
            hb,db,_=estimator(state,b[candidate_B],k2,bank,main_cfg,True)
            out['components'].append(np.asarray(moments(ha,da,hb,db)).tolist())
            for q in out['rows']:
                m,k,n,bs,_=q['config']
                q['pairs'].append(float(pair(state,a[bs],b[bs],key,dummy,m,k,n)))
            if (i+1)%4==0:
                out['complete']=i==15;publish(fs,dst,out)
        run.log({'state':step,'pairs':16})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):main(*sys.argv[1:])

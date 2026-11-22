"""Paired full-model variance on sealed text, with block and native-audit coins integrated."""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import fsspec
import haliax as h
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import block_audits, fifty_blocks, fused_attention
from .contest import pair
from .diagnostic_artifacts import publish
from .fifty_crossover import configurations, native_call
from .independent_audits import forecast
from .native_eval import recover
from .nested_forecast import norm_moments
from .subspace_forecast import native_audit
from .run import ROOT, ready


def population(fs,strict=False):
    protocol = json.loads(Path(__file__).with_name('block_strict_confirmation_protocol.json' if strict else 'block_confirmation_protocol.json').read_text())
    path = ROOT+'/inputs/'+protocol['pool']
    raw = fs.cat_file(path+'.npz')
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == protocol['sha256'] == json.loads(fs.cat_file(path+'.json'))['sha256']
    with np.load(io.BytesIO(raw)) as z:
        starts = z['starts']
        mask = (starts+64 <= 128*(4001-2)) | (starts >= 128*(4051+2))
        use = np.repeat(mask, 64)
        arrays = tuple(z[k][use] for k in ('tokens','weights','segments'))
    assert len(use) == 4096 and np.all(arrays[1].sum(-1)==1023)
    return arrays, dict(holdout=digest, sequences=len(arrays[0]), excluded=starts[~mask].tolist())


def discrepancy(state, data, key):
    gold = native_call(native_audit, state.model, data, fifty_blocks.B)
    return block_audits.strict(state,data,key,fifty_blocks.K,fifty_blocks.N,gold)


def main(block, rep, strict=0):
    from . import fifty_blocks_strict as fixed,samplewise_blocks as single
    configure_storage()
    fs = fsspec.filesystem('s3')
    pool, provenance = population(fs,strict)
    fused_attention.install()
    cases = sorted({c for panel in (2,3) for c in configurations(panel) if c[0]!=401})
    sizes = sorted({c[3] for c in cases} | {fifty_blocks.B})
    gpu = jax.devices()[0].device_kind
    board = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cfg = dict(**provenance,device=gpu, board=board, block=block, rep=rep, dataset='native-confirm6' if strict else 'native-confirm5', method=408 if strict==2 else 402 if strict else 401, paired_seed=653927)
    os.environ['WANDB_API_KEY'] = require_secret('WANDB_API_KEY')
    run = wandb.init(entity='marin-community',project='gradient-variance-frontier',group='fifty-block-confirmation',id=os.urandom(6).hex(),mode='online',config=cfg)
    dummy = jnp.arange(1)
    folder='fifty-samplewise-block-variance-v1' if strict==2 else 'fifty-block-strict-variance-v1' if strict else 'fifty-block-variance-v1'
    for step in range(4001+5*block,4006+5*block):
        dst = ROOT+f'/{folder}/{gpu.split()[1]}-s{step}-r{rep}.json'
        if fs.exists(dst) and json.loads(fs.cat_file(dst)).get('complete'):
            continue
        state, src = recover(step,fs)
        pred = ready(forecast(state,10))
        out = dict(**cfg,step=step,checkpoint=src,wandb=run.url,components=[],rows=[dict(config=c,pairs=[]) for c in cases],complete=False)
        for i in range(16):
            rng = np.random.default_rng(np.random.SeedSequence([653927,step,rep,i]))
            ids = rng.integers(len(pool[0]),size=(4,max(sizes)))
            def batch(row,b):
                return tuple(jnp.asarray(v[ids[row,:b]]) for v in pool)
            left, right = ({b:batch(row,b) for b in sizes} for row in (0,1))
            key = jax.random.fold_in(jax.random.key(402731+rep),step*100+i)
            key_a,key_b = jax.random.split(key)
            key_c,key_d = jax.random.split(jax.random.fold_in(key,9))
            B,K,N = fifty_blocks.B,fifty_blocks.K,fifty_blocks.N
            values = []
            for part in (1,2):
                if strict:
                    def measure(batch,draw_key):
                        base_key,phase_key=jax.random.split(draw_key)
                        phase=jax.random.randint(phase_key,(),0,batch[0].shape[1]//N,dtype=jnp.int32)
                        return fixed.raw_block(state,batch,base_key,part,phase)
                    values.append(np.asarray(fixed.moments(measure(left[B],key_a),measure(right[B],key_b),pred,part)))
                else:
                    g1=block_audits.grad(state,left[B],key_a,K,N,part)
                    g2=block_audits.grad(state,right[B],key_b,K,N,part)
                    values.append(np.asarray(block_audits.component(g1,g2,pred,part)))
            repair=single.discrepancy if strict==2 else fixed.discrepancy if strict else discrepancy
            values.append(np.asarray(norm_moments(repair(state,batch(2,B),key_c),repair(state,batch(3,B),key_d))))
            out['components'].append(np.concatenate(values).tolist())
            for idx in rng.permutation(len(cases)):
                m,k,n,b,_ = cases[idx]
                out['rows'][idx]['pairs'].append(float(pair(state,left[b],right[b],key,dummy,m,k,n)))
            if (i+1)%4==0:
                out['complete'] = i==15
                publish(fs,dst,out)
        run.log({'state':step,'pairs':16})
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

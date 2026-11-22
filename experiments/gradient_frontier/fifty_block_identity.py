"""Exhaust all block allocations, precision coins, cyclic phases, and data-panel assignments."""
import os
import sys
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from . import block_audits, fifty_blocks, fused_attention
from .diagnostic_artifacts import publish
from .fifty_block_variance import population
from .fifty_crossover import native_call
from .independent_audits import forecast
from .native_eval import recover
from .subspace_forecast import native_audit
from .run import ROOT, ready


@eqx.filter_jit
def emitted(state, data, key, part, phase, delta):
    pred = forecast(state,10)
    if part:
        raw = block_audits.draw(state,data,key,fifty_blocks.K,fifty_blocks.N,part,phase)
        p = fifty_blocks.PT if part==1 else fifty_blocks.PR
        pred = jax.tree_util.tree_map_with_path(lambda path,a,b:a+(b-a)/p if block_audits.readout(path)==(part==2) else a,pred,raw)
    if delta is not None:
        pred = jax.tree.map(lambda a,b:a+b/fifty_blocks.R,pred,delta)
    return pred


def main(step,strict=0):
    from . import fifty_blocks_strict as fixed, prepared_blocks as prepared
    configure_storage()
    fs=fsspec.filesystem('s3')
    pool, meta=population(fs,strict==2)
    state, src=recover(step,fs)
    fused_attention.install()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='enumerate-block-estimator',id=os.urandom(6).hex(),mode='online',config=dict(**meta,step=step,origin=src))
    out=dict(**meta,algorithm={0:401,1:402,2:402,3:406}[strict],step=step,source=src,wandb=run.url,rows=[],outcomes_per_pair=192,host_accumulator='float64',device=jax.devices()[0].device_kind)
    gen=np.random.default_rng(np.random.SeedSequence([239265,step]))
    B,K,N=fifty_blocks.B,fifty_blocks.K,fifty_blocks.N
    pt,pr,r=fifty_blocks.PT,fifty_blocks.PR,fifty_blocks.R
    for i in range(6):
        ids=gen.integers(len(pool[0]),size=(2,B))
        batches=[tuple(jnp.asarray(v[ix]) for v in pool) for ix in ids]
        keys=jax.random.split(jax.random.key(9321+step*10+i))
        if strict==3:
            full_ids=gen.integers(len(pool[0]),size=(2,128))
            full=[tuple(jnp.asarray(v[ix]) for v in pool) for ix in full_ids]
            slots,rotations,_=prepared.prepare(239265+step*6+i)
            slots=slots[:2]
            # Exchange both the data and its cyclic-offset assignment for the independent correction.
            slots=[(a,a,c,c,e) for a,_,c,_,e in slots]
            batches=[tuple(x[slot[0]] for x in panel) for panel,slot in zip(full,slots)]
            ids=np.array([ix[np.asarray(s[0])] for ix,s in zip(full_ids,slots)])
        gold=[ready(native_call(native_audit,state.model,d,B)) for d in batches]
        if strict==3:
            corr=[ready(prepared.discrepancy(state,d,s,rotations)) for d,s in zip(full,slots)]
        else:
            corr=[ready(fixed.discrepancy(state,d,k) if strict else block_audits.strict(state,d,k,K,N,g)) for d,k,g in zip(batches,keys,gold)]
        refs=[.5*(np.asarray(x,np.float64)+np.asarray(y,np.float64)) for x,y in zip(*map(jax.tree.leaves,gold))]
        acc=[np.zeros_like(a) for a in refs]
        for side in range(2):
            for part,p in enumerate((1-pt-pr,pt,pr)):
                # A skipped raw query is identical on all sixteen phases.
                phases=16 if part else 1
                for phase in range(phases):
                    value=fixed.raw_block(state,batches[side],keys[side],part,jnp.asarray(phase,jnp.int32)) if strict in (1,2) and part else None
                    if strict==3 and part:
                        slot=slots[side]
                        value=prepared.raw_block(state.model,full[side],slot[0],slot[2],rotations[phase],part)
                    for hit,w in ((False,1-r),(True,r)):
                        raw=fixed.emitted(state,value,part,corr[1-side] if hit else None) if strict else emitted(state,batches[side],keys[side],part,jnp.asarray(phase),corr[1-side] if hit else None)
                        weight=.5*p*w/phases
                        for a,b in zip(acc,jax.tree.leaves(raw)):
                            a+=np.asarray(b,dtype=np.float64)*weight
        err=[(a-b).ravel() for a,b in zip(acc,refs)]
        relative=float(np.sqrt(sum(np.dot(a,a) for a in err)/sum(np.dot(v.ravel(),v.ravel()) for v in refs)))
        mx=float(max(np.max(np.abs(a)) for a in err))
        out['rows'].append(dict(panel=i,relative_l2=relative,maximum_coordinate=mx,sample_ids=ids.tolist()))
        folder={0:'fifty-block-identity-v1',1:'fifty-block-strict-identity-v1',2:'fifty-block-strict-sealed-identity-v1',3:'prepared-block-identity-v1'}[strict]
        publish(fs,ROOT+f'/{folder}/s{step}.json',out)
        run.log(dict(relative_l2=relative,maximum_coordinate=mx))
    run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

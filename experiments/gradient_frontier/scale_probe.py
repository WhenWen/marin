"""Search actual midpoint MuonW checkpoints for low-cost learnable mean gradients."""
import json
import os
import sys
import time
from pathlib import Path
import draccus
import equinox as eqx
import fsspec
import haliax as h
import jax
import jax.numpy as j
import numpy as np
import wandb
from levanter.checkpoint import load_checkpoint
from levanter.trainer_state import TrainerState
from levanter.layers.attention import Attention
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import RunSpec,configuration
from experiments.hybrid_transfer.recovery import scientific_identity
from . import debias,fused_attention
from .subspace_forecast import native_audit,components
from .nested_forecast import large_control,difference,add_control,norm_moments
from .head_to_head import update
from .contest import pair
from .temporal import momentum
from .run import ROOT,ready


def recover_scale(width,depth,fs,step=4001):
    spec=RunSpec(width,depth,'standard','muonw',.02,0)
    cfg=configuration(spec,'diagnostic','s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    root=f'marin-us-east-04a/marin/experiments/hybrid-transfer-h100-20261116/state/standard-muonw-w{width}-d{depth}-lr0.02-s0/checkpoints'
    paths=sorted(fs.glob(root+'/*/step-4000/metadata.json'));assert paths
    ck=paths[0].removesuffix('/metadata.json')
    identity=json.loads(fs.cat_file(ck.rsplit('/',1)[0]+'/identity.json'))
    assert scientific_identity(draccus.encode(cfg))==identity['scientific_config']
    assert cfg.model.max_Pos.size==1024
    opt=cfg.optimizer.build(cfg.trainer.num_train_steps)
    def initialize():
        model=cfg.model.build(h.Axis('vocab',128256),key=jax.random.key(0))
        return TrainerState.init(opt,model,key=jax.random.PRNGKey(0),mp=cfg.trainer.mp)
    src='s3://'+ck
    if step!=4001:
        receipt=json.loads(fs.cat_file(ROOT+f'/native-muonw-w{width}-d{depth}/completed/{step}.json'))
        assert receipt['next_step']==step and receipt['origin']==src
        src=receipt['checkpoint']
    state=load_checkpoint(eqx.filter_eval_shape(initialize),src,allow_partial=False,mesh=jax.sharding.Mesh(np.asarray(jax.devices()).reshape(-1,1),('data','model')),axis_mapping={})
    assert int(state.step)==int(state.opt_state.count)==step
    return state,src,cfg


@eqx.filter_jit
def fit_scalars(state,grad):
    m=momentum(state)
    return j.stack([j.vdot(x,y)/j.maximum(1e-18,j.vdot(x,x)) if x.size<3_000_000 else j.array(0.) for x,y in zip(jax.tree.leaves(m),jax.tree.leaves(grad))])


def predict(state,coef):
    m=momentum(state);leaves,td=jax.tree.flatten(m)
    return jax.tree.unflatten(td,[c*x if x.size<3_000_000 else j.zeros_like(x) for x,c in zip(leaves,coef)])


@eqx.filter_jit
def ahead(state,coef,a,p,key):
    pred=predict(state,coef)
    grad=pred if a is None else jax.tree.map(lambda x,y:x+(y-x)/p,pred,a)
    return state.take_step(grad,key=key)[0]


def main(width,depth,K,micro,cal):
    configure_storage();fs=fsspec.filesystem('s3');state,src,cfg=recover_scale(width,depth,fs)
    tmp=Path('/tmp/scale-pool.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:
        hold=tuple(z[n][512:] for n in ('tokens','weights','segments'))
        learn=tuple(j.asarray(z[n][:max(1,cal)]) for n in ('tokens','weights','segments'))
    original=Attention.__call__;fused_attention.install()
    def native(d,mic):
        Attention.__call__=original
        try:return native_audit(state.model,d,mic)
        finally:fused_attention.install()
    def residual(d,key):
        _,avg=debias.audit(state,d,key,12288,64,61,j.arange(1))
        return difference(native(d,micro),avg)
    if cal:
        def fit():return fit_scalars(state,native(learn,cal))
        coeff=ready(fit());prep=[]
        for _ in range(6):
            tick=time.perf_counter();coeff=ready(fit());prep.append((time.perf_counter()-tick)*1000)
    else:coeff=j.zeros(len(jax.tree.leaves(eqx.filter(state.model,eqx.is_inexact_array))));prep=[0.]
    seed=9678+depth+width+cal;gen=np.random.default_rng(seed)
    def draw(b):
        ii=gen.integers(len(hold[0]),size=b);return tuple(j.asarray(v[ii]) for v in hold)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    cfglog=dict(step=4001,width=width,depth=depth,K=K,micro=micro,cal=cal,checkpoint=src,seed=seed,
                device=jax.devices()[0].device_kind,schedule=cfg.trainer.num_train_steps,params=sum(q.size for q in jax.tree.leaves(eqx.filter(state.model,eqx.is_inexact_array))))
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='midpoint-size-causal-audits',id=os.urandom(5).hex(),config=cfglog)
    cases=[(m,k,n,b) for m,k,n in ((61,4096,16),(61,8192,32),(61,12288,64),(62,16384,64),(7,16384,64)) for b in (1,2,3,4,6,8,12,16,24,32)]
    rows=[dict(method=m,k=k,n=n,b=b,pair=[],ms=[]) for m,k,n,b in cases]
    out=dict(**cfglog,prep=prep,wandb=run.url,rows=rows,times=[],components=[])
    dst=ROOT+f'/scale-probe/w{width}-d{depth}-K{K}-m{micro}-c{cal}.json'
    pred=predict(state,coeff);dummy=j.arange(1)
    for i in range(25):
        a,b,large,ext,other,U,V=ready(tuple(draw(v) for v in (micro,micro,K,micro,micro,32,32)))
        l,r=ready(tuple({B:tuple(x[:B] for x in d) for _,_,_,B in cases} for d in (U,V)))
        ka,kb,kc,kd=jax.random.split(jax.random.key(seed+i),4);ts=[0.]*3
        def execute(c):
            if c>=0:
                m,k,n,B=cases[c];return update(state,l[B],ka,dummy,m,k,n)
            audit=None if c==-1 else large_control(state,large,kb,micro)
            if c==-3:audit=add_control(audit,residual(ext,kc),.125)
            return ahead(state,coeff,audit,.25,ka)
        for c in gen.permutation(np.arange(-3,len(cases))):
            tick=time.perf_counter();ready(execute(c));ms=1000*(time.perf_counter()-tick)
            if i:
                if c<0:ts[-c-1]=ms
                else:rows[c]['ms'].append(ms)
        if not i:continue
        v=components(large_control(state,a,ka,micro),large_control(state,b,kb,micro),pred)[0]
        er=components(native(a,micro),native(b,micro),pred)[1]
        rem=norm_moments(residual(ext,kc),residual(other,kd))
        out['times'].append(ts);out['components'].append(np.asarray(j.concatenate((j.stack((v,er)),rem))).tolist())
        for row,(m,k,n,B) in zip(rows,cases):row['pair'].append(float(pair(state,l[B],r[B],ka,dummy,m,k,n)))
        if i%4==0:fs.pipe(dst,json.dumps(out).encode())
    run.log({'raw_risk':np.mean(out['components'],axis=0)[1]});run.finish()


if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))),h.axis_mapping({}):
        main(*map(int,sys.argv[1:]))

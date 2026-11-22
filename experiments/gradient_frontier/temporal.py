"""Predict gradients from the original optimizer moments and use independent full-support correction rounds."""
import json,os,sys,time
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .group_regression import fit,levels
from .head_to_head import update,pair_stats
from .run import ROOT,ready
from .fused_attention import install


def momentum(state):
    leaves,tree=jax.tree_util.tree_flatten_with_path(eqx.filter(state.model,eqx.is_inexact_array))
    buffers={}
    for path,x in jax.tree_util.tree_leaves_with_path(state.opt_state):
        key=jax.tree_util.keystr(path)
        if '.momentum_buffer.' in key:tail=key.split('.momentum_buffer',1)[1];factor=.05
        elif '.mu.' in key:tail=key.split('.mu',1)[1];factor=1.
        else:continue
        if tail in buffers:raise ValueError(tail)
        buffers[tail]=(x,factor)
    values=[]
    for path,x in leaves:
        k=jax.tree_util.keystr(path)
        if k not in buffers:raise ValueError('Missing historical first moment for '+k+'; observed '+str(list(buffers)[:2]))
        v,factor=buffers[k];assert v.shape==x.shape
        values.append(v.astype(j.float32)*factor)
    return jax.tree.unflatten(tree,values)


def audit(model,data,key,bank,kind,micro):
    K,t=data[0].shape
    config=(0,0) if kind==0 else (kind*100000+(12288 if kind==61 else 4096),64)
    def calc(d,ke):return eqx.filter_grad(objective)(model,d,bank,ke,config)
    if K<=micro:return calc(data,key)
    assert K%micro==0
    groups=jax.tree.map(lambda x:x.reshape(-1,micro,t),data)
    ke=jax.random.split(key,K//micro)
    initial=calc(jax.tree.map(lambda x:x[0],groups),ke[0])
    def add(i,total):return jax.tree.map(j.add,total,calc(jax.tree.map(lambda x:x[i],groups),ke[i]))
    total=jax.lax.fori_loop(1,len(ke),add,initial)
    return jax.tree.map(lambda a:a/len(ke),total)


@eqx.filter_jit
def take(state,data,key,bank,kind,micro,correct,p):
    pred=momentum(state)
    if correct:
        full=audit(state.model,data,key,bank,kind,micro)
        grad=jax.tree.map(lambda x,g:x+(g-x)/p,pred,full)
    else:grad=pred
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def moments(state,a,b,key,bank,kind,micro):
    ka,kb=jax.random.split(key)
    x=audit(state.model,a,ka,bank,kind,micro);y=audit(state.model,b,kb,bank,kind,micro);h=momentum(state)
    def dot(u,v):return sum(j.vdot(p,q) for p,q in zip(jax.tree.leaves(u),jax.tree.leaves(v)))
    diff=jax.tree.map(j.subtract,x,y)
    u=jax.tree.map(j.subtract,x,h);v=jax.tree.map(j.subtract,y,h)
    return j.stack((dot(diff,diff)*.5,dot(u,v),dot(x,y),dot(h,h),.5*(dot(x,h)+dot(y,h))))


def main(step,K,micro,kind,rep):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    tmp=Path('/tmp/temporal-data.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:
        hold=tuple(z[a][512:] for a in ('tokens','weights','segments'))
        train=tuple(j.asarray(z[a][:levels(113)[0]]) for a in ('tokens','weights','segments'))
    bank=j.arange(1);prep=[0.]
    if kind==113:
        original=state if step==4001 else recover(4001,fs)[0];bank=ready(fit(original.model,train,4096,113));prep=[]
        for _ in range(6):
            t=time.perf_counter();ready(fit(original.model,train,4096,113));prep.append(1000*(time.perf_counter()-t))
    seed=48717+step+rep;rng=np.random.default_rng(seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    config=dict(step=step,K=K,micro=micro,kind=kind,source=src,rep=rep)
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='historical-moment-audits',id=os.urandom(5).hex(),config=config)
    data=dict(**config,wandb=run.url,prep=prep,device=jax.devices()[0].device_kind,moments=[],rows=[])
    dst=ROOT+f'/temporal/{jax.devices()[0].device_kind.split()[1]}-s{step}-K{K}-m{micro}-v{kind}-r{rep}.json'
    cases=[(m,b) for m in (3,61) for b in (2,4,8,12,16,24,32)]+[(-1,0),(-2,K)]
    timing={c:[] for c in cases};vs={c:[] for c in cases if c[0]>0};dummy=j.arange(1)
    def get():
        idx=rng.integers(len(hold[0]),size=max(32,K));return {b:tuple(j.asarray(x[idx[:b]]) for x in hold) for b in sorted({K,8,*(c[1] for c in cases if c[1])})}
    def shot(c,pool,key):
        m,b=c
        if m<0:return take(state,pool[K],key,bank,kind,micro,m==-2,.25)
        return update(state,pool[b],key,dummy,m,32768 if m==3 else 12288,0 if m==3 else 64)
    for i in range(48):
        left,right=ready((get(),get()));key=jax.random.key(seed+i)
        # With independent sequences and per-sequence correction randomness, trace covariance scales as 1/B.
        refB=min(8,K)
        data['moments'].append(np.asarray(moments(state,left[refB],right[refB],key,bank,kind,micro)).tolist())
        for m,b in vs:
            k,n=(32768,0) if m==3 else (12288,64)
            vs[(m,b)].append(float(pair_stats(state,left[b],right[b],key,dummy,m,k,n,False)[0]))
        if i==0:
            for c in cases:ready(shot(c,left,key))
        for idx in rng.permutation(len(cases)):
            c=cases[idx];t=time.perf_counter();ready(shot(c,left,key));timing[c].append(1000*(time.perf_counter()-t))
        if (i+1)%12==0:
            data['rows']=[dict(method=m,b=b,ms=timing[(m,b)],pair=vs.get((m,b),[])) for m,b in cases];fs.pipe(dst,json.dumps(data).encode())
    run.log({'predictor_norm2':np.mean(data['moments'],0)[3]});run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

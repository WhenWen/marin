"""Compare true target estimators and random-negative sampled softmax on identical GPU/data draws."""
import json, os, sys, time
from pathlib import Path
import equinox as eqx
import fsspec, jax, jax.numpy as j, numpy as np, wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover
from .confirm_goal import go
from .fused_attention import install
from .run import ROOT, ready
from .tail import trunk


NEGATIVE_METHODS=(1,2,3,4,201,202,203,204,205,206)

def negative_loss(model,data,key,k,method):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=trunk(m,data); tokens,weight,_=data; y=j.roll(tokens,-1,axis=1)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    # IID log-uniform negative draws shared across tokens. Multiplicities remain in the sum.
    L=j.log1p(float(w.shape[0]))
    if method>=201:
        tokens_per_bank={201:4096,202:8192,203:16384,204:128,205:64,206:512}[method]
        b,t,d=x.shape;flat=x.reshape(-1,d);labels=y.reshape(-1);total=len(labels)
        width=min(tokens_per_bank,total);pad=(-total)%width;groups=(total+pad)//width
        xx=j.pad(flat,((0,pad),(0,0))).reshape(groups,width,d)
        yy=j.pad(labels,((0,pad),)).reshape(groups,width)
        u=(j.arange(k)[None,:]+jax.random.uniform(key,(groups,k)))/k
        ids=j.minimum(j.floor(j.expm1(L*u)).astype(j.int32),w.shape[0]-1)
        proposal=j.log1p(1./(1.+ids))/L
        negative=j.einsum('gtd,gkd->gtk',xx,w[ids]).astype(j.float32)-j.log(k*proposal)[:,None,:]
        negative=j.where(ids[:,None,:]==yy[:,:,None],-j.inf,negative)
        pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
        nz=jax.nn.logsumexp(negative,-1).reshape(-1)[:total].reshape(b,t)
        return ((j.logaddexp(pos,nz)-pos)*weight).sum()/weight.sum()
    grouped=method in (2,4)
    stratified=method in (3,4)
    draws=jax.random.uniform(key,(tokens.shape[0] if grouped else 1,k))
    u=(j.arange(k)[None,:]+draws)/k if stratified else draws
    idx=j.minimum(j.floor(j.expm1(L*u)).astype(j.int32),w.shape[0]-1)
    proposal=j.log1p(1./(1.+idx))/L
    if grouped:
        negative=j.einsum('btd,bkd->btk',x,w[idx]).astype(j.float32)
    else:
        negative=(x@w[idx[0]].T).astype(j.float32)
    negative=negative-j.log(k*proposal[:,None,:])
    negative=j.where(idx[:,None,:]==y[:,:,None],-j.inf,negative)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    loss=j.logaddexp(positive,jax.nn.logsumexp(negative,-1))-positive
    return (loss*weight).sum()/weight.sum()


def gradient(state,data,key,bank,method,k,n):
    if method in NEGATIVE_METHODS: return eqx.filter_grad(negative_loss)(state.model,data,key,k,method)
    if method==0:return go(state,data,bank,key,'base',(0,0))[0]
    return go(state,data,bank,key,'candidate',(100000*method+k,n))[0]


@eqx.filter_jit
def update(state,data,key,bank,method,k,n):
    g=gradient(state,data,key,bank,method,k,n)
    return state.take_step(g,key=key)[0]


@eqx.filter_jit
def pair_stats(state,left,right,key,bank,method,k,n,estimate_bias=True):
    ka,kb=jax.random.split(key)
    ga=gradient(state,left,ka,bank,method,k,n)
    gb=gradient(state,right,kb,bank,method,k,n)
    variance=sum(j.vdot(a-b,a-b) for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)))*.5
    if method not in NEGATIVE_METHODS or not estimate_bias:return j.stack((variance,j.zeros_like(variance)))
    ra=gradient(state,left,ka,bank,0,0,0);rb=gradient(state,right,kb,bank,0,0,0)
    bias2=sum(j.vdot(a-x,b-y) for a,x,b,y in zip(*map(jax.tree.leaves,(ga,ra,gb,rb))))
    return j.stack((variance,bias2))


def main(step,method,k,n,rep,*batches):
    configure_storage();fs=fsspec.filesystem('s3');install()
    state,origin=recover(step,fs)
    f=Path('/tmp/pair-held.npz');fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(f))
    with np.load(f) as z:hold=tuple(z[a] for a in ('tokens','weights','segments'))
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='sampled-softmax-baseline-frontier',id=os.urandom(5).hex(),config=dict(step=step,method=method,k=k,n=n,rep=rep,batches=batches))
    if 90<=method<=97 or 100<=method<=107 or 110<=method<=117 or 120<=method<=127 or 130<=method<=137 or 140<=method<=147 or 150<=method<=157 or 160<=method<=167 or 170<=method<=177:
        if method<=97:from .learned_adjoint import fit,levels
        elif method<=107:from .neighbor_adjoint import fit,levels
        elif method<=117:from .group_regression import fit,levels
        elif method<=127:from .balanced import fit,levels
        elif method<=137:from .product_control import fit,levels
        elif method<=147:from .cross_calibration import fit,levels
        elif method<=157:from .sobolev_control import fit,levels
        elif method<=167:from .compact_group import fit,levels
        else:from .online_adjoint import fit,levels
        f=Path('/tmp/fit.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
        initial=state if step==4001 else recover(4001,fs)[0]
        with np.load(f) as z:train=tuple(j.asarray(z[a][:levels(method)[0]]) for a in ('tokens','weights','segments'))
        bank=ready(fit(initial.model,train,k,method));setup=[]
        for _ in range(7):
            t=time.perf_counter();bank=ready(fit(initial.model,train,k,method));setup.append(1000*(time.perf_counter()-t))
    else:bank=j.arange(k);setup=[0.]
    device=jax.devices()[0].device_kind
    dst=ROOT+f'/head-to-head/{device.split()[1]}-s{step}-v{method}-k{k}-n{n}-r{rep}.json'
    out=dict(step=step,device=device,source=origin,method=method,k=k,n=n,rep=rep,wandb=run.url,setup_ms=setup,results=[])
    def load(indices):return tuple(j.asarray(a[indices]) for a in hold)
    for b in batches:
        seed=rep+step*41;rng=np.random.default_rng(seed)
        row=dict(batch=b,paired=[],time_ms=[],bias_measured=(method in NEGATIVE_METHODS and rep<70))
        for i in range(64):
            left,right=[load(rng.integers(len(hold[0]),size=b)) for _ in range(2)]
            if i==0:ready(update(state,left,jax.random.key(seed),bank,method,k,n))
            key=jax.random.key(seed+i)
            row['paired'].append(np.asarray(pair_stats(state,left,right,key,bank,method,k,n,rep<70)).tolist())
            ready(left);t=time.perf_counter();ready(update(state,left,key,bank,method,k,n));row['time_ms'].append(1000*(time.perf_counter()-t))
        out['results'].append(row);fs.pipe(dst,json.dumps(out).encode())
        run.log({f'variance/batch_{b}':np.mean(row['paired'],axis=0)[0]})
    run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

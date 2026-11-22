"""Multilevel full-vocabulary gradients using shared target draws on short and original contexts."""
import json,os,sys,time
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover
from .run import ROOT,ready
from .fused_attention import install
from .tail import trunk


def starts(key,b,t,J,q):
    width=t//J
    return j.arange(J)[None,:]*width+q*jax.random.randint(key,(b,J),0,width//q)


def loss(model,data,chosen,context,q,short):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    tok,wt,seg=data;b,t=tok.shape;ng=chosen.shape[1]
    at=chosen[:,:,None]+j.arange(q)
    targets=j.take_along_axis(j.roll(tok,-1,axis=1),at.reshape(b,-1),axis=1)
    weights=j.take_along_axis(wt,at.reshape(b,-1),axis=1)
    if short:
        ix=chosen[:,:,None]+j.arange(q-context,q)
        index=ix.reshape(b,-1).clip(0,t-1)
        xx=j.take_along_axis(tok,index,axis=1).reshape(b*ng,context)
        segment=j.take_along_axis(seg,index,axis=1).reshape(b*ng,context)
        segment=j.where(ix.reshape(b*ng,context)<0,-1,segment)
        hidden=trunk(m,(xx,j.ones_like(xx,j.float32),segment))[:,-q:,:].reshape(b,-1,128)
    else:
        xx=trunk(m,data);hidden=j.take_along_axis(xx,at.reshape(b,-1,1),axis=1)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    scores=(hidden@w.T).astype(j.float32);z=jax.nn.logsumexp(scores,-1)
    ce=z-j.take_along_axis(scores,targets[:,:,None],axis=-1)[:,:,0]
    return (ce*weights).sum()*(t/(ng*q))/wt.sum()


def mixed(model,data,key,b,J,context,q,scale):
    M,t=data[0].shape;sel=starts(key,M,t,J,q)
    small=jax.tree.map(lambda a:a[:b],data);extra=jax.tree.map(lambda a:a[b:],data)
    c=scale*(1-b/M)
    return loss(model,small,sel[:b],context,q,False)+c*(loss(model,extra,sel[b:],context,q,True)-loss(model,small,sel[:b],context,q,True))


@eqx.filter_jit
def timed(state,data,key,b,J,L,q):
    grad=eqx.filter_grad(mixed)(state.model,data,key,b,J,L,q,1.)
    return state.take_step(grad,key=key)[0]


@eqx.filter_jit
def diagnostic(state,a,b,key,J,L,q):
    keys=jax.random.split(key)
    def two(dat,k):
        n,t=dat[0].shape;sel=starts(k,n,t,J,q)
        fn=eqx.filter_grad(loss)
        return fn(state.model,dat,sel,L,q,False),fn(state.model,dat,sel,L,q,True)
    aa,ha=two(a,keys[0]);bb,hb=two(b,keys[1])
    vals=[]
    for g,h,gg,hh in zip(*map(jax.tree.leaves,(aa,ha,bb,hb))):
        x=g-gg;y=h-hh;vals.append(.5*j.stack((j.vdot(x,x),j.vdot(y,y),j.vdot(x,y))))
    return j.stack(vals)


def main(step,J,L,q,b,M):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    file=Path('/tmp/context-labels.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(file))
    with np.load(file) as z:hold=tuple(z[a][512:] for a in ('tokens','weights','segments'))
    seed=23953+step;rng=np.random.default_rng(seed)
    conf=dict(step=step,J=J,L=L,q=q,b=b,M=M,source=src,seed=seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='context-multilevel',id=os.urandom(5).hex(),config=conf)
    conf.update(pairs=[],timing=[],wandb=run.url,device=jax.devices()[0].device_kind)
    from .head_to_head import update,pair_stats
    dummy=j.arange(1)
    # Time both current baselines in the same process. The matched full-gradient pair observations are direct.
    cases=[(m,k,n,B) for m,k,n in ((3,32768,0),(61,12288,64)) for B in sorted({max(2,b//2),b,2*b,3*b,4*b})]
    cases.append((200,J,q,M));times={v:[] for v in cases};basepair={v:[] for v in cases[:-1]}
    def draw():
        ii=rng.integers(len(hold[0]),size=max(M,4*b));return {B:tuple(j.asarray(a[ii[:B]]) for a in hold) for B in set([4]+[c[3] for c in cases])}
    def dispatch(c,x,key):
        m,k,n,B=c
        return timed(state,x,key,b,J,L,q) if m==200 else update(state,x,key,dummy,m,k,n)
    def persist():
        conf['timing']=[dict(method=m,k=k,n=n,B=B,ms=times[(m,k,n,B)],pair=basepair.get((m,k,n,B),[])) for m,k,n,B in cases]
        fs.pipe(ROOT+f'/context-mlmc/s{step}-j{J}-L{L}-q{q}-b{b}-M{M}.json',json.dumps(conf).encode())
    for i in range(48):
        left,right=ready((draw(),draw()));key=jax.random.key(seed+i)
        conf['pairs'].append(np.asarray(diagnostic(state,left[4],right[4],key,J,L,q)).tolist())
        for c in cases[:-1]:
            m,k,n,B=c;basepair[c].append(float(pair_stats(state,left[B],right[B],key,dummy,m,k,n,False)[0]))
        if i==0:
            for c in cases:ready(dispatch(c,left[c[-1]],key))
        for ix in rng.permutation(len(cases)):
            c=cases[ix];t=time.perf_counter();ready(dispatch(c,left[c[-1]],key));times[c].append(1000*(time.perf_counter()-t))
        if (i+1)%12==0:persist()
    run.log({'median_ms':np.median(times[cases[-1]])});run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

"""Preregistered exact-batch, paired benchmark on a disjoint packed-data population."""
from collections import defaultdict
import hashlib,json,os,sys,time
from pathlib import Path
import equinox as eqx,fsspec,haliax as h,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .run import ROOT,ready
from .native_eval import recover,objective
from .tail import trunk
from .fused_attention import install

def pieces(x,w,y,weight):
    logits=(x@w.T).astype(j.float32)
    z=jax.nn.logsumexp(logits,-1)
    loss=((z-j.take_along_axis(logits,y[:,:,None],-1)[:,:,0])*weight).sum()/weight.sum()
    return loss,(x,w,y,weight,j.exp(logits-z[:,:,None]))
@jax.custom_vjp
def exact(x,w,y,weight):return pieces(x,w,y,weight)[0]
def exact_bwd(res,v):
    x,w,y,wt,p=res;b,t,d=x.shape
    error=p.at[j.arange(b)[:,None],j.arange(t)[None,:],y].add(-1)
    error=(error*(v*wt/wt.sum())[:,:,None]).astype(x.dtype)
    return error@w,error.reshape((-1,w.shape[0])).T@x.reshape((-1,d)),None,None
exact.defvjp(pieces,exact_bwd)

def loss(model,data,ids,key,method,c):
    if method in ('base','candidate'):
        return objective(model,data,ids,key,(0,0) if method=='base' else c)
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model);x=trunk(m,data)
    tokens,weight,seg=data
    if method=='native':
        from levanter.models.loss import maybe_fused_next_token_loss
        axes=(h.Axis('batch',tokens.shape[0]),h.Axis('position',tokens.shape[1]))
        retval=maybe_fused_next_token_loss(m.Pos,m.Embed,m.Vocab,h.named(x,(*axes,m.Embed)),m.get_lm_head(),
                  h.named(tokens,axes),loss_weight=h.named(weight,axes),logsumexp_weight=0.)
        return retval.array if hasattr(retval,'array') else retval
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    return exact(x,w,j.roll(tokens,-1,axis=1),weight)

def go(state,data,ids,key,method,c):
    g=eqx.filter_grad(loss)(state.model,data,ids,key,method,c)
    nxt,u=state.take_step(g,key=key)
    return g,u,nxt
@eqx.filter_jit
def one(state,data,ids,key,method,c):
    return go(state,data,ids,key,method,c)[2]
@eqx.filter_jit
def pair(state,a,b,ids,key,method,c):
    k1,k2=jax.random.split(key)
    g,u,_=go(state,a,ids,k1,method,c)
    g2,u2,_=go(state,b,ids,k2,method,c)
    def distance(x,y):
        return sum(j.vdot(i-jj,i-jj) for i,jj in zip(jax.tree.leaves(x),jax.tree.leaves(y)))*.5
    return j.stack((distance(g,g2),distance(u,u2)))

def main(step,k,n,v,trial,mode,B=16):
    configure_storage();fs=fsspec.filesystem('s3');install()
    key_prefix=ROOT+('/inputs/native-packed' if mode==2 else '/inputs/native-confirm3' if mode in (5,6) else ('/inputs/native-confirm2' if mode in (3,4) else '/inputs/native-confirm'));f=Path('/tmp/nat-confirm.npz')
    fs.get_file(key_prefix+'.npz',str(f));meta=json.loads(fs.cat_file(key_prefix+'.json'))
    assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
    with np.load(f) as z:hold=tuple(z[a][512:] if mode==2 else z[a] for a in ('tokens','weights','segments'))
    assert np.unique(hold[1].sum(1)).tolist()==[1023.]
    state,src=recover(step,fs)
    # Natural-ID candidate has zero external preprocessing, and the implementation uses a contiguous slice.
    assert v in (5,6,7) or 11<=v<=37 or 40<=v<=46 or 50<=v<=55 or 60<=v<=65 or 70<=v<=78 or 80<=v<=87 or 90<=v<=97
    if v in range(90,98):
        from .learned_adjoint import fit,levels
        initial=state if step==4001 else recover(4001,fs)[0]
        train_path=Path("/tmp/adjoint-calibration.npz")
        fs.get_file(ROOT+"/inputs/native-packed.npz",str(train_path))
        with np.load(train_path) as train:
            ref=tuple(j.asarray(train[a][:levels(v)[0]]) for a in ("tokens","weights","segments"))
        make=lambda:fit(initial.model,ref,k,v)
    elif v in range(40,47) or v in range(50,56):
        if v in range(50,56):
            from .virtual import fit
        else:
            from .cluster_head import fit
        initial=state if step==4001 else recover(4001,fs)[0]
        with np.load(f) as z:ref=tuple(j.asarray(z[a][:4]) for a in ('tokens','weights','segments'))
        make=lambda:fit(initial.model,ref,k,v)
    else:make=lambda:j.arange(k)
    t=time.perf_counter();ids=ready(make());setup=(time.perf_counter()-t)*1000
    # Charge one steady-state device initialization. Track first-dispatch time separately.
    preps=[]
    for _ in range(32):
        t=time.perf_counter();ready(make());preps.append(1000*(time.perf_counter()-t))
    seed=391730+100*step+trial;rng=np.random.default_rng(seed);c=(100000*v+k,n)
    code=os.urandom(5).hex();os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='independent-packed-confirmation',
        id=code,name=f'{step}-k{k}-n{n}-v{v}-trial{trial}-m{mode}',config=dict(state=src,seed=seed,batch=B,data=meta,mode=mode))
    out=dict(step=step,k=k,n=n,v=v,trial=trial,seed=seed,mode=mode,source=src,wandb=run.url,
             device=jax.devices()[0].device_kind,dataset=meta,batch=B,baseline_pair_sizes=[max(1,B//2-1),B//2],first_setup_ms=setup,preprocess_ms=preps,pairs=[],timing=[])
    family='goal-confirm-h100' if 'H100' in jax.devices()[0].device_kind else 'goal-confirm'
    dst=ROOT+f'/{family}/k{k}-n{n}-v{v}-s{step}-t{trial}-m{mode}'+(f'-b{B}' if mode in (3,4) or B!=16 else '')+'.json'
    def get(idx):return tuple(j.asarray(a[idx]) for a in hold)
    if mode in (0,2,3,5):
        for it in range(96 if mode==2 else 192):
            ix=rng.integers(len(hold[0]),size=(2,B))
            a,b=[get(jj) for jj in ix];key=jax.random.PRNGKey(seed+it)
            vals=[]
            for name,size in [('candidate',B),('base',max(1,B//2-1)),('base',B//2)]:
                aa,bb=[jax.tree.map(lambda x:x[:size],z) for z in (a,b)]
                vals.append(np.asarray(pair(state,aa,bb,ids,key,name,c)).tolist())
            out['pairs'].append(vals)
        fs.pipe(dst,json.dumps(out).encode())
    # An exact full-softmax custom backward and the ordinary autodiff share the selected attention backend.
    # The original native streamed CE is measured in a separate crossover against the same candidate.
    sizes=(range(max(1,B//4),max(5,3*B//4+1)) if mode in (0,2,3,5) else tuple(range(max(1,B//2-2),max(3,B//2+2))))
    if B>=32 and mode in (2,3,5):
        sizes=sorted({B//4,B//3,*(B//2+i for i in (-6,-4,-2,-1,0,1,2,4,6)),3*B//4})
    methods=('base','manual') if mode in (0,2,3,5) else ('base','native')
    cases=[(method,b) for b in sizes for method in methods]+[('candidate',B)]
    pools=[]
    for it in range(82):
        seq=rng.integers(len(hold[0]),size=B)
        pools.append(({b:get(seq[:b]) for b in {b for _,b in cases}},jax.random.PRNGKey(seed+300+it)))
    ready(pools)
    for name,b in cases:
        for pool,key in pools[:2]:ready(one(state,pool[b],ids,key,name,c))
    times=defaultdict(list)
    for pool,key in pools[2:]:
        for i in rng.permutation(len(cases)):
            name,b=cases[i];t=time.perf_counter()
            ready(one(state,pool[b],ids,key,name,c));times[(name,b)].append(1000*(time.perf_counter()-t))
    out['timing']=[dict(method=name,b=b,ms=ms) for (name,b),ms in times.items()]
    fs.pipe(dst,json.dumps(out).encode());run.log({'timing/candidate_ms':np.median(times[('candidate',B)])});run.finish()
if __name__=='__main__':
    arguments=list(map(int,sys.argv[1:]))
    if arguments[5] in (1,4,6):
        with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))), h.axis_mapping({}):
            main(*arguments)
    else:main(*arguments)

"""Measure paired gradient and MuonW-update noise on independently saved optimizer states."""
import functools,hashlib,json,os,sys,time
from collections import defaultdict
from pathlib import Path
import equinox as eqx,fsspec,haliax as h,jax,jax.numpy as j,numpy as np,wandb
from jax.flatten_util import ravel_pytree
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention_mask import AttentionMask
from levanter.trainer_state import TrainerState
from experiments.hybrid_transfer.train import RunSpec,configuration
from experiments.hybrid_transfer.storage import configure_storage
from rigging.secrets import require_secret
from .trajectory import ORIGIN
from .multifidelity import shortlist
from .run import ROOT,ready

def recover(step,fs):
    cfg=configuration(RunSpec(128,4,'standard','muonw',.02,0),'diagnostic','s3://marin-us-east-02a/marin','s3://marin-us-east-04a/marin/experiments',1,8)
    opt=cfg.optimizer.build(cfg.trainer.num_train_steps)
    def template():
        model=cfg.model.build(h.Axis('vocab',128256),key=jax.random.PRNGKey(0))
        return TrainerState.init(opt,model,key=jax.random.PRNGKey(0),mp=cfg.trainer.mp)
    example=eqx.filter_eval_shape(template)
    src=ORIGIN if step==4001 else json.loads(fs.cat_file(ROOT+f'/native-muonw/completed/{step}.json'))['checkpoint']
    mesh=jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))
    with jax.set_mesh(mesh):
        state=load_checkpoint(example,src,allow_partial=False,mesh=mesh,axis_mapping={})
    assert int(state.step)==int(state.opt_state.count)==step
    return state,src

def objective(model,data,ids,key,config):
    code,n=config
    tokens,weight,seg=data
    rows,t=tokens.shape
    axes=(h.Axis('batch',rows),h.Axis('position',t))
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=m.embeddings.embed(h.named(tokens,axes))
    mask=AttentionMask.causal().with_segment_ids(h.named(seg,axes))
    for layer in m.transformer.layers.unstacked():x=layer(x,mask)
    x=m.transformer.norm(x).rearrange(('batch','position','embed')).array
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    y=j.roll(tokens,-1,axis=1)
    if code//100000 in range(180,188):
        from .antithetic_vocab import objective as antithetic
        return antithetic(x,w,y,weight,key,config,ids)
    if code//100000 in range(170,178):
        from .online_adjoint import objective as online_adjoint
        return online_adjoint(x,w,y,weight,key,config,ids)
    if code//100000 in range(160,168):
        from .compact_group import objective as compact
        return compact(x,w,y,weight,key,config,ids)
    if code//100000 in range(150,158):
        from .sobolev_control import objective as sobolev
        return sobolev(x,w,y,weight,key,config,ids)
    if code//100000 in range(140,148):
        from .cross_calibration import objective as cross_calibrated
        return cross_calibrated(x,w,tokens,y,weight,key,config,ids)
    if code//100000 in range(130,138):
        from .product_control import objective as product
        return product(x,w,y,weight,key,config,ids)
    if code//100000 in range(120,128):
        from .balanced import objective as balanced
        return balanced(x,w,tokens,y,weight,key,config,ids)
    if code//100000 in range(110,118):
        from .group_regression import objective as grouped
        return grouped(x,w,y,weight,key,config,ids)
    if code//100000 in range(100,108):
        from .neighbor_adjoint import objective as knn_adjoint
        return knn_adjoint(x,w,tokens,y,weight,key,config,ids)
    if code//100000 in range(90,98):
        from .learned_adjoint import objective as learned_adjoint
        return learned_adjoint(x,w,tokens,y,weight,key,config,ids)
    if code//100000 in range(80,88):
        from .flash_head import objective as flash_head
        return flash_head(x,w,y,weight,key,config)
    if code//100000 in range(60,66) or code//100000 in range(70,79):
        from .lowmem import objective as shared_stats
        return shared_stats(x,w,y,weight,key,config)
    if code//100000 in range(50,56):
        from .virtual import objective as virtual_vocab
        return virtual_vocab(x,w,y,weight,key,config,ids)
    if code//100000 in range(40,47):
        from .cluster_head import objective as cluster_vocab
        return cluster_vocab(x,w,y,weight,key,config,ids)
    if code//100000 in range(30,38):
        from .batch_head import objective as adaptive_vocab
        return adaptive_vocab(x,w,y,weight,key,config)
    if not code:
        logits=(x@w.T).astype(j.float32)
        all_loss=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,y[:,:,None],-1)[:,:,0]
        return (all_loss*weight).sum()/weight.sum()
    k=code%100000
    global_pps=code//100000 in (21,22,23,24,25,26)
    natural=global_pps or code//100000 in (5,6,7,11,12,13,14,15,16,17,18,19,20)
    tail_fit=code//100000 in (8,9,10)
    selected=(ids[0] if tail_fit else ids)[:k]
    head=(x@(w[:k] if natural else w[selected]).T).astype(j.float32)
    positive=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z=jax.nn.logsumexp(head,-1)
    if code>=100000:
        member=(j.arange(w.shape[0])<k if natural else j.zeros((w.shape[0],),bool).at[selected].set(True))
        if code//100000 in (3,4):
            local=j.sort(y[:,::4],axis=1)
            unique=j.concatenate((j.ones_like(local[:,:1],dtype=bool),local[:,1:]!=local[:,:-1]),axis=1)
            extra=unique & ~member[local]
            local_scores=j.einsum('btd,bsd->bts',x,w[local]).astype(j.float32)
            z=j.logaddexp(z,jax.nn.logsumexp(j.where(extra[:,None,:],local_scores,-1e9),axis=-1))
            idx=jax.vmap(j.searchsorted)(local,y).clip(0,local.shape[1]-1)
            covers=member[y] | (j.take_along_axis(local,idx,1)==y)
        else:covers=member[y]
        z=j.logaddexp(z,j.where(covers,-1e9,positive))
    rank_tail=code//100000 in (16,17,18,19,20,24,25)
    if rank_tail:
        factor={16:0.5,17:1.,18:2.,19:4.,20:8.,24:1.,25:2.}[code//100000]
        last=jax.nn.logsumexp(head[:,:,k//2:],-1)
        inband=(y>=k//2)&(y<k)
        excluded=j.where(inband,j.exp(j.minimum(0.,positive-last)),0.)
        extrap=last+j.log(j.maximum(1.-excluded,1e-5))+j.log(factor)
        z=j.logaddexp(z,extrap)
        tail_mass=j.exp(extrap-z)
    sampled_tail=code//100000 in (11,12,13,14,15)
    if sampled_tail:
        stride={11:16,12:32,13:64,14:128,15:256}[code//100000]
        tailids=j.arange(k+stride//2,w.shape[0],stride)
        sparse=(x@w[k+stride//2::stride].T).astype(j.float32)
        sparse=j.where(y[:,:,None]==tailids[None,None,:],-1e9,sparse)
        ztail=jax.nn.logsumexp(sparse,-1)+j.log(float(stride))
        z=j.logaddexp(z,ztail)
        tail_mass=j.exp(ztail-z)
    if tail_fit:
        from .tail import estimate
        z,tail_mass=estimate(x,head,z,positive,member[y],ids)
    width=t//n
    if global_pps:
        if rank_tail:
            miss=tail_mass
        else:
            outer=jax.nn.logsumexp(head[:,:,k//2:],-1)
            miss=j.minimum(1.,j.exp(outer-z))
        top=j.maximum(j.max(head,-1),positive)
        diffuse=1-j.minimum(1.,j.exp(top-z))
        if code//100000==22:score=.02+miss+.15*diffuse
        elif code//100000==23:score=.02+j.sqrt(miss)
        else:score=.035+j.sqrt(miss)+.15*diffuse
        prob=jax.lax.stop_gradient(score/score.sum(-1,keepdims=True))
        order=(j.argsort(y,axis=-1) if code//100000==26 else None)
        pr=prob if order is None else j.take_along_axis(prob,order,axis=1)
        u=(j.arange(n)[None,:]+jax.random.uniform(key,(rows,n)))/n
        pos=jax.vmap(lambda cdf,val:j.searchsorted(cdf,val,side='right',method='scan_unrolled'))(j.cumsum(pr,axis=-1),u)
        pos=j.minimum(pos,t-1)
        if order is not None:pos=j.take_along_axis(order,pos,axis=1)
        iw=1./(n*j.take_along_axis(prob,pos,axis=1))
    elif code//100000 in (2,4,6,7,9,10,11,12,13,14,15,16,17,18,19,20):
        top=j.maximum(j.max(head,-1),positive)
        if sampled_tail or rank_tail:score=.04+j.sqrt(tail_mass)
        elif tail_fit:score=0.015+(tail_mass if code//100000==9 else j.sqrt(tail_mass))
        elif code//100000==7:
            outer=jax.nn.logsumexp(head[:,:,k//2:],-1)
            score=0.035+j.sqrt(j.minimum(1.,j.exp(outer-z)))+0.15*(1-j.minimum(1.,j.exp(top-z)))
        else:score=1.1-j.minimum(1.,j.exp(top-z))
        s=jax.lax.stop_gradient(score.reshape((rows,n,width)))
        p=s/s.sum(-1,keepdims=True)
        if natural or tail_fit:
            u=jax.random.uniform(key,(rows,n,1))
            i=j.minimum((u>=j.cumsum(p,axis=-1)).sum(-1),width-1)
        else:i=jax.random.categorical(key,j.log(p),axis=-1)
        iw=1./j.take_along_axis(p,i[:,:,None],-1)[:,:,0]
    else:
        i=jax.random.randint(key,(rows,n),0,width);iw=width
    if not global_pps:pos=i+j.arange(n)[None,:]*width
    xs=j.take_along_axis(x,pos[:,:,None],1)
    full_logits=(xs@w.T).astype(j.float32)
    full_y=j.take_along_axis(y,pos,axis=1)
    exact=jax.nn.logsumexp(full_logits,-1)-j.take_along_axis(full_logits,full_y[:,:,None],axis=-1)[:,:,0]
    cheap=z-positive
    delta=(exact-j.take_along_axis(cheap,pos,1))*j.take_along_axis(weight,pos,1)*iw
    return ((cheap*weight).sum()+delta.sum())/weight.sum()

def grad_update(state,data,ids,key,config):
    grad=eqx.filter_grad(objective)(state.model,data,ids,key,config)
    next_state,update=state.take_step(grad,key=key)
    return grad,update,next_state

@eqx.filter_jit
def compare(state,a,b,ids,ka,kb,config):
    ga,ua,_=grad_update(state,a,ids,ka,config)
    gb,ub,_=grad_update(state,b,ids,kb,config)
    def diff(x,y):
        d=jax.tree.map(lambda l,r:l-r,x,y)
        return sum(j.vdot(l,l) for l in jax.tree.leaves(d))*0.5
    return j.stack((diff(ga,gb),diff(ua,ub)))

@eqx.filter_jit
def full_step(state,data,ids,key,config):
    return grad_update(state,data,ids,key,config)[2]

def main(step,freq,n,variant,flash=0):
    if flash:
        from .fused_attention import install
        install()
        if flash==2:
            from .fused_attention import install_projections
            install_projections()
    configure_storage();fs=fsspec.filesystem('s3')
    data_path=ROOT+'/inputs/native-packed'
    meta=json.loads(fs.cat_file(data_path+'.json'))
    folder=Path('/tmp/frontier-state-eval');folder.mkdir(exist_ok=True)
    f=folder/'data.npz';fs.get_file(data_path+'.npz',str(f))
    assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
    with np.load(f) as z:data=tuple(z[k] for k in ('tokens','weights','segments'))
    # Separate calibration from entirely different native batches used for variance measurement.
    cal=j.asarray(data[0][:512]);hold=tuple(z[512:] for z in data)
    assert np.unique(hold[1].sum(1)).size==1
    state,source=recover(step,fs)
    if variant in (8,9,10):
        from .tail import fit
        initial=state if step==4001 else recover(4001,fs)[0]
        refs=tuple(j.asarray(a[:16]) for a in data)
        make=lambda x:fit(initial.model,refs,x,freq,8)
    else:
        make=functools.partial(shortlist,k=freq,vocab=128256)
    ids=ready(make(cal))
    prep=[]
    for _ in range(32):
        t=time.perf_counter();ready(make(cal));prep.append(1000*(time.perf_counter()-t))
    seed=87311+step;rng=np.random.default_rng(seed); ident=os.urandom(5).hex()
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='native-MuonW-packed-assay',
                   id=ident,name=f'packed-{step}-k{freq}-n{n}-v{variant}',config=dict(source=source,step=step,seed=seed,k=freq,n=n,variant=variant,holdout=meta))
    candidate=(100000*variant+freq,n);base=(0,0);B=4
    out=dict(step=step,k=freq,n=n,variant=variant,source=source,wandb=run.url,prep_ms=prep,pairs=[])
    def data_for(indices):return tuple(j.asarray(z[indices]) for z in hold)
    for i in range(96):
        a,b=[data_for(rng.integers(len(hold[0]),size=B)) for _ in range(2)]
        k1,k2=jax.random.split(jax.random.PRNGKey(seed+i))
        out['pairs'].append([np.asarray(compare(state,a,b,ids,k1,k2,c)).tolist() for c in (base,candidate)])
    family={0:'native-assay',1:'native-flash-assay',2:'native-packed-projections'}[flash]
    path=ROOT+f'/{family}/k{freq}-n{n}-v{variant}-s{step}.json'
    fs.pipe(path,json.dumps(out).encode())
    cases=[(base,b) for b in (1,2,3,4,6,8,12,16,24,32)]+[(candidate,b) for b in (4,8,12,16,24,32)]
    ranks=sorted({b for _,b in cases})
    samples=[]
    for i in range(50):
        draws=rng.integers(len(hold[0]),size=max(ranks));samples.append(({b:data_for(draws[:b]) for b in ranks},jax.random.PRNGKey(seed+100+i)))
    ready(samples)
    for c,b in cases:
        for x,key in samples[:2]:ready(full_step(state,x[b],ids,key,c))
    timing=defaultdict(list)
    for x,key in samples[2:]:
        for idx in rng.permutation(len(cases)):
            c,b=cases[idx];t=time.perf_counter();ready(full_step(state,x[b],ids,key,c))
            timing[(c,b)].append(1000*(time.perf_counter()-t))
    out['timing']=[dict(method=('base' if c==base else 'candidate'),b=b,ms=ms) for (c,b),ms in timing.items()]
    fs.pipe(path,json.dumps(out).encode());run.log({'variance/gradient_ratio':np.mean(np.array(out['pairs'])[:,1,0])/np.mean(np.array(out['pairs'])[:,0,0])});run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

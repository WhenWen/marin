"""On-device crossover of learned estimators with the sampled-softmax and debiased batch frontiers."""
from collections import defaultdict
import hashlib,json,os,sys,time
from pathlib import Path
import equinox as eqx
import fsspec,jax,jax.numpy as j,numpy as np,wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover
from .head_to_head import gradient,update
from .fused_attention import install
from .run import ROOT,ready
from .group_regression import fit,levels


@eqx.filter_jit
def pair(state,a,b,key,bank,m,k,n):
    ka,kb=jax.random.split(key)
    ga=gradient(state,a,ka,bank,m,k,n);gb=gradient(state,b,kb,bank,m,k,n)
    return sum(j.vdot(x-y,x-y) for x,y in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)))*.5


def main(step,trial,Bcandidate=16,method=113,size=4096,count=64):
    configure_storage();fs=fsspec.filesystem('s3');install();state,source=recover(step,fs)
    screening=trial>=70
    if method!=113:
        if method<100:from .learned_adjoint import fit,levels
        elif method<110:from .neighbor_adjoint import fit,levels
        elif method<120:from .group_regression import fit,levels
        elif method<130:from .balanced import fit,levels
        elif method<140:from .product_control import fit,levels
        elif method<150:from .cross_calibration import fit,levels
        elif method<160:from .sobolev_control import fit,levels
        elif method<170:from .compact_group import fit,levels
        else:from .online_adjoint import fit,levels
    else:from .group_regression import fit,levels
    pop='native-packed' if screening else 'native-confirm4'
    tmp=Path('/tmp/contest-data.npz');fs.get_file(ROOT+'/inputs/'+pop+'.npz',str(tmp))
    meta=json.loads(fs.cat_file(ROOT+'/inputs/'+pop+'.json'))
    assert hashlib.sha256(tmp.read_bytes()).hexdigest()==meta['sha256']
    with np.load(tmp) as z:hold=tuple(z[a][512:] if screening else z[a] for a in ('tokens','weights','segments'))
    fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:train=tuple(j.asarray(z[a][:levels(method)[0]]) for a in ('tokens','weights','segments'))
    origin=state if step==4001 else recover(4001,fs)[0]
    bank=ready(fit(origin.model,train,size,method));prep=[]
    for _ in range(12):
        t=time.perf_counter();bank=ready(fit(origin.model,train,size,method));prep.append(1000*(time.perf_counter()-t))
    cases=[(method,size,count,Bcandidate)]
    sizes=(2,3,4,5,6,8,10,12) if Bcandidate==8 else (4,6,8,10,12,16,20,24)
    # Distinct baseline families keep vocabulary draw noise and GPU occupancy in the comparison.
    for m,k,n in [(3,16384,0),(3,32768,0),(4,4096,0),(61,12288,64),(62,16384,64),(7,16384,64)]:
        for b in sizes:
            cases.append((m,k,n,b))
    cases.extend((4,1024,0,b) for b in sizes[::2])
    cases.extend((0,0,0,b) for b in sizes[:5])
    if screening:
        bs=sorted({max(1,Bcandidate//2),max(1,3*Bcandidate//4),Bcandidate,5*Bcandidate//4,3*Bcandidate//2,2*Bcandidate})
        cases=cases[:1]+[(m,k,n,b) for m,k,n in ((3,32768,0),(61,12288,64)) for b in bs]
    device=jax.devices()[0].device_kind
    seed=59348+step*113+trial;gen=np.random.default_rng(seed)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='matched-sampled-softmax-crossover',id=os.urandom(5).hex(),config=dict(step=step,trial=trial,seed=seed,source=source,holdout=meta,device=device))
    dst=(ROOT+f'/screen-contest/{device.split()[1]}-s{step}-v{method}-k{size}-n{count}-b{Bcandidate}-r{trial}.json' if screening else ROOT+f'/matched-contest/{device.split()[1]}-s{step}-r{trial}.json')
    rows={key:dict(method=key[0],k=key[1],n=key[2],b=key[3],pair=[],ms=[]) for key in cases}
    dummy=j.arange(1);banks={k:bank if k==cases[0] else dummy for k in cases}
    def save(stage):
        fs.pipe(dst,json.dumps(dict(step=step,trial=trial,device=device,source=source,holdout=meta,wandb=run.url,prep=prep,stage=stage,rows=list(rows.values()))).encode())
    def batches():
        ix=gen.integers(len(hold[0]),size=max(b for _,_,_,b in cases))
        return {b:tuple(j.asarray(a[ix[:b]]) for a in hold) for b in {case[3] for case in cases}}
    # Compile before creating any timed observation. Execute all methods in randomized order each cycle.
    a,b=ready((batches(),batches()));key=jax.random.key(seed)
    for case in cases:
        m,k,n,B=case
        ready(pair(state,a[B],b[B],key,banks[case],m,k,n))
        ready(update(state,a[B],key,banks[case],m,k,n))
    for it in range(36 if screening else 72):
        a,b=ready((batches(),batches()));key=jax.random.key(seed+it+1)
        for idx in gen.permutation(len(cases)):
            case=cases[idx];m,k,n,B=case
            rows[case]['pair'].append(float(pair(state,a[B],b[B],key,banks[case],m,k,n)))
        for idx in gen.permutation(len(cases)):
            case=cases[idx];m,k,n,B=case;t=time.perf_counter()
            ready(update(state,a[B],key,banks[case],m,k,n));rows[case]['ms'].append(1000*(time.perf_counter()-t))
        if (it+1)%12==0:save('partial')
    save('complete')
    run.log({'variance/learned':np.mean(rows[cases[0]]['pair'])});run.finish()
if __name__=='__main__':main(*map(int,sys.argv[1:]))

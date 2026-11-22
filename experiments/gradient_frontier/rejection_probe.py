"""Check certified envelopes for exact context-dependent categorical draws on a real vocabulary."""
import json
import sys
from pathlib import Path
import equinox as eqx
import fsspec
import jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .fused_attention import install
from .native_eval import recover
from .run import ROOT,ready
from .tail import trunk


@eqx.filter_jit
def geometry(model,cal,k,width,blocks):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array.astype(j.float32)
    h=trunk(m,cal).reshape(-1,w.shape[1]).astype(j.float32)
    ev,rot=j.linalg.eigh(h.T@h/len(h))
    logits=h[::32]@w.T
    importance=(jax.nn.softmax(logits,axis=1)**2).mean(0)
    rank=j.argsort(-importance);head=rank[:k];others=rank[k:]
    sig=logits[:16,others];bits=sig>j.median(sig,axis=1,keepdims=True)
    order=others[j.argsort((bits*(2**j.arange(16))[:,None]).sum(0))]
    ng=len(others)//width
    assert len(others)%width==0
    table=(w[order]@rot).reshape(ng,width,-1)
    center=table.mean(1);res=(table-center[:,None,:]).reshape(ng,width,blocks,-1)
    radius=j.max(j.linalg.norm(res,axis=-1),axis=1)
    return w,head,order,rot,center,radius


@eqx.filter_jit
def measure(model,data,bank,width,blocks):
    w,head,order,rot,center,radius=bank
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    allh=trunk(m,data).reshape(-1,w.shape[1]).astype(j.float32)
    h=allh[::8]
    def one(x):
        exact=x@w.T
        z=jax.nn.logsumexp(exact,-1)
        head_z=jax.nn.logsumexp(exact[:,head],-1)
        tail=exact[:,order].reshape(len(x),len(center),width)
        best=tail.max(-1)
        xr=x@rot
        norm=j.linalg.norm(xr.reshape(len(x),blocks,-1),axis=-1)
        upper=xr@center.T+norm@radius.T
        def acceptance(bound):
            total=j.logaddexp(head_z,jax.nn.logsumexp(bound+j.log(float(width)),axis=-1))
            return z-total
        headmass=j.exp(head_z-z)
        return j.stack((acceptance(upper),acceptance(best),headmass,j.max(best-upper,axis=1)),-1)
    return jax.lax.map(one,h.reshape(-1,32,w.shape[1])).reshape(-1,4)


def main(step,k,width,blocks):
    configure_storage();fs=fsspec.filesystem('s3');install();state,src=recover(step,fs)
    local=Path('/tmp/reject-grad.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(local))
    with np.load(local) as z:pool=tuple(z[n] for n in ('tokens','weights','segments'))
    cal=tuple(j.asarray(a[:4]) for a in pool)
    bank=ready(geometry(state.model,cal,k,width,blocks))
    rng=np.random.default_rng(129835);rows=[]
    for _ in range(32):
        idx=rng.integers(512,len(pool[0]),size=4)
        d=tuple(j.asarray(a[idx]) for a in pool)
        rows.append(np.asarray(measure(state.model,d,bank,width,blocks)))
    allv=np.concatenate(rows)
    result=dict(step=step,k=k,width=width,blocks=blocks,contexts=len(allv),source=src,
                log_acceptance_quantiles=np.quantile(allv[:,0],[0,.1,.5,.9,1]).tolist(),
                oracle_log_acceptance_quantiles=np.quantile(allv[:,1],[0,.1,.5,.9,1]).tolist(),
                mean_head_mass=float(allv[:,2].mean()),max_actual_minus_bound=float(allv[:,3].max()),
                log_mean_proposals=float(np.logaddexp.reduce(-allv[:,0].astype(np.float64))-np.log(len(allv))))
    fs.pipe(ROOT+f'/certified-rejection/s{step}-k{k}-w{width}-b{blocks}.json',json.dumps(result).encode())
    print(result,flush=True)


if __name__=='__main__':main(*map(int,sys.argv[1:]))

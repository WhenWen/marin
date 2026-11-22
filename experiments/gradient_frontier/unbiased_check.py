"""Condition on a real packed batch and check estimator mean against its exact full gradient."""
import sys,json
from pathlib import Path
import equinox as eqx,fsspec,haliax as h,jax,jax.numpy as j,numpy as np
from levanter.layers.attention_mask import AttentionMask
from levanter.models.lm_model import LmExample
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .run import ROOT
from .multifidelity import shortlist
from jax.flatten_util import ravel_pytree

def native(model,data):
    m=jax.tree.map(lambda x:x.astype(j.bfloat16),model)
    tok,weight,seg=data;axes=(h.Axis('batch',tok.shape[0]),h.Axis('position',tok.shape[1]))
    example=LmExample(h.named(tok,axes),h.named(weight,axes),AttentionMask.causal().with_segment_ids(h.named(seg,axes)))
    val=m.compute_next_token_loss(example,logsumexp_weight=0.)
    return val.array if hasattr(val,'array') else val
@eqx.filter_jit
def native_grad(m,data):return ravel_pytree(eqx.filter_grad(native)(m,data))[0]
@eqx.filter_jit
def gradient(m,data,ids,key,c):return ravel_pytree(eqx.filter_grad(objective)(m,data,ids,key,c))[0]
@jax.jit
def accumulate(curr,g,truth):
    diff=g-truth
    return curr+diff,j.vdot(diff,diff)

def main(s,k,n,v):
    configure_storage();fs=fsspec.filesystem('s3');state,src=recover(s,fs)
    f=Path('/tmp/packed-bias.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(f))
    with np.load(f) as z:
        data=tuple(j.asarray(z[a][2032:2034]) for a in ('tokens','weights','segments'))
        cal=j.asarray(z['tokens'][:512])
    if v in range(110,118) or v in range(140,148):
        from .fused_attention import install
        if v<140:from .group_regression import fit,levels
        else:from .cross_calibration import fit,levels
        install()
        with np.load(f) as z:train=tuple(j.asarray(z[a][:levels(v)[0]]) for a in ('tokens','weights','segments'))
        initial=state if s==4001 else recover(4001,fs)[0]
        ids=fit(initial.model,train,k,v)
    else:ids=shortlist(cal,k,128256)
    key=jax.random.PRNGKey(s)
    true=gradient(state.model,data,ids,key,(0,0))
    ref=native_grad(state.model,data)
    one=float(j.vdot(true,true))
    complete=gradient(state.model,data,ids,key,(100000*v+k,1024))
    floor=float(j.vdot(complete-true,complete-true)/one)**.5
    check=dict(step=s,checkpoint=src,k=k,n=n,v=v,full_norm_squared=one,exhaustive_relative_error=floor,
        native_relative_error=float(j.vdot(true-ref,true-ref)/j.vdot(ref,ref))**.5,
        native_cosine=float(j.vdot(true,ref)/(j.linalg.norm(true)*j.linalg.norm(ref))))
    blocks=[];sq=[]
    for block in range(16):
        cumul=j.zeros_like(true);vs=[]
        for i in range(64):
            g=gradient(state.model,data,ids,jax.random.fold_in(key,64*block+i),(100000*v+k,n))
            cumul,r=accumulate(cumul,g,true);vs.append(r)
        blocks.append(cumul/64);sq.extend(np.asarray(vs).tolist())
    mean=sum(blocks)/len(blocks);nr=len(sq)
    check.update(draws=nr,mean_residual_norm=float(j.linalg.norm(mean)),
        expected_mc_norm=float(np.sqrt(np.mean(sq)/nr)),bias_squared_u_statistic=float((nr*j.vdot(mean,mean)-np.mean(sq))/(nr-1)))
    fs.pipe(ROOT+f'/bias/k{k}-n{n}-v{v}-s{s}.json',json.dumps(check).encode());print(check,flush=True)
if __name__=='__main__':
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()).reshape((-1,1)),('data','model'))):
        main(*map(int,sys.argv[1:]))

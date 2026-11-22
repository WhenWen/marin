"""Train linear synthetic gradients at the three dominant trunk projections and test on unseen sequences."""
import json,sys,time
from pathlib import Path
import equinox as eqx
import fsspec,haliax as h,jax,jax.numpy as j,numpy as np
from jax.experimental.pallas.ops.gpu.attention import mha
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .fused_attention import install
from .run import ROOT,ready


def tapped(m,data,noise):
    tok,_,seg=data;b,t=tok.shape
    axes=(h.Axis('batch',b),h.Axis('position',t))
    x=m.embeddings.embed(h.named(tok,axes));record=[]
    def arr(v):return v.rearrange(('batch','position',...)).array.reshape(b,t,-1)
    for i,layer in enumerate(m.transformer.layers.unstacked()):
        a=layer.self_attn;nx=layer.input_layernorm(x)
        q,k,v=a._compute_qkv(nx,key=None,pos_ids=None)
        q=q.flatten_axes(('kv_head','q_heads_per_group'),'heads').rearrange(('batch','position','heads','head_size'))
        k=k.rearrange(('batch','position','kv_head','head_size'));v=v.rearrange(k.axes)
        vv=v.array if noise is None else v.array+noise[i,0].reshape(v.array.shape)
        hv=mha(q.array,k.array,vv,seg,sm_scale=a.config.scaling_factor,causal=True)
        ho=a.o_proj(h.named(hv,q.axes).astype(x.dtype));ho=ho.rearrange(('batch','position','embed'))
        if noise is not None:ho=ho+h.named(noise[i,1],ho.axes)
        x=x+layer.config.branch_scale*layer.post_attn_layernorm(ho)
        mx=layer.post_attention_layernorm(x);ml=layer.mlp
        act=ml.act(ml.gate_proj(mx))*ml.up_proj(mx)
        hd=ml.down_proj(act).rearrange(('batch','position','embed'))
        if noise is not None:hd=hd+h.named(noise[i,2],hd.axes)
        x=x+layer.config.branch_scale*layer.post_mlp_layernorm(hd)
        record.append((arr(nx),hv.reshape(b,t,-1),arr(act),arr(ho),arr(hd)))
    return arr(m.transformer.norm(x)),record


def block_features(m,data,x,r,variant):
    tok,weight,seg=data;y=j.roll(tok,-1,1);w=m.get_lm_head().rearrange(('vocab','embed')).array
    wy=w[y];wcur=w[tok];base=[x,wy,wcur,(x*wy)]
    if variant>0:
        # Future labels contribute to earlier causal-attention value gradients.
        for look in (1,4,16):
            nxt=j.roll(wy,-look,axis=1)
            valid=(j.arange(tok.shape[1])+look<tok.shape[1])[None,:]&(seg==j.roll(seg,-look,1))
            base.append(nxt*valid[:,:,None])
    if variant>=2:
        base.extend((r[0],r[3],r[4]))
    return j.concatenate(base,axis=-1).astype(j.bfloat16)


def scales(r):
    def rms(x):return j.sqrt(j.mean(x.astype(j.float32)**2,-1)+1e-6)
    return j.stack((j.ones(r[0].shape[:2],j.float32),rms(r[3]),rms(r[4])),axis=-1)


def teacher(m,data,variant):
    d=m.get_lm_head().rearrange(("vocab","embed")).array.shape[1]
    noise=j.zeros((4,3,*data[0].shape,d),j.bfloat16)
    def loss(e):
        x,rec=tapped(m,data,e);w=m.get_lm_head().rearrange(('vocab','embed')).array
        logits=(x@w.T).astype(j.float32);y=j.roll(data[0],-1,axis=1)
        exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,y[:,:,None],-1)[:,:,0]
        return (exact*data[1]).sum(),(x,rec)
    (_, (x,rec)),der=jax.value_and_grad(loss,has_aux=True)(noise)
    out=[]
    for i,r in enumerate(rec):
        f=block_features(m,data,x,r,variant);scale=scales(r)
        target=j.moveaxis(der[i],0,2).astype(j.float32)*scale[:,:,:,None]
        out.append((f.reshape(-1,f.shape[-1]),target.reshape(-1,3*d)))
    return out


@eqx.filter_jit
def fit(model,data,variant):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    def stats(sub):
        out=[]
        for f,y in teacher(m,sub,variant):
            ones=j.ones((len(f),1),j.bfloat16);f=j.concatenate((ones,f),-1)
            out.append((jax.lax.dot_general(f.T,f,(((1,),(0,)),((),())),preferred_element_type=j.float32),
                        jax.lax.dot_general(f.T,y.astype(j.bfloat16),(((1,),(0,)),((),())),preferred_element_type=j.float32)))
        return jax.tree.map(lambda v:v,out)
    blocks=jax.tree.map(lambda a:a.reshape(-1,2,a.shape[-1]),data)
    first=stats(jax.tree.map(lambda a:a[0],blocks))
    def acc(i,v):return jax.tree.map(j.add,v,stats(jax.tree.map(lambda a:a[i],blocks)))
    sums=jax.lax.fori_loop(1,blocks[0].shape[0],acc,first)
    def solve(xx,xy):
        sd=j.sqrt(j.maximum(.0001,j.diag(xx)/data[0].size))
        cov=xx/sd[:,None]/sd[None,:]
        return (jax.scipy.linalg.solve(cov+.04*data[0].size*j.eye(len(sd)),xy/sd[:,None],assume_a='pos')/sd[:,None]).astype(j.bfloat16)
    return tuple(solve(xx,xy) for xx,xy in sums)


def prediction(model,data,bank,variant):
    m=jax.tree.map(lambda a:jax.lax.stop_gradient(a.astype(j.bfloat16)),model)
    x,rec=tapped(m,data,None);den=data[1].sum()
    scalar=0.
    for layer,r,coef in zip(model.transformer.layers.unstacked(),rec,bank):
        f=block_features(m,data,x,r,variant)
        pred=(f@coef[1:]+coef[0]).reshape(*data[0].shape,3,128)
        pred=(pred.astype(j.float32)/scales(r)[:,:,:,None]).astype(j.bfloat16)
        matrices=(layer.self_attn.v_proj.weight.array,layer.self_attn.o_proj.weight.array,layer.mlp.down_proj.weight.array)
        for jx,(u,w) in enumerate(zip(r[:3],matrices)):
            dg=(pred[:,:,jx,:].reshape(-1,128).T@u.reshape(-1,u.shape[-1])).astype(j.float32)/den
            scalar=scalar+j.sum(jax.lax.stop_gradient(dg).reshape(w.shape)*w)
    return scalar


@eqx.filter_jit
def evaluate(model,left,right,bank,mode,key):
    def gs(data):
        truth=eqx.filter_grad(objective)(model,data,j.arange(1),key,(0,0))
        pred=eqx.filter_grad(prediction)(model,data,bank,mode)
        return truth,pred
    a,b=gs(left);c,d=gs(right)
    out=[]
    for g,h1,g2,h2 in zip(*map(jax.tree.leaves,(a,b,c,d))):
        u=g-g2;v=h1-h2
        out.append(j.stack((j.vdot(u,u),j.vdot(v,v),j.vdot(u,v)))*.5)
    return j.stack(out)


def main(step,ncal,mode):
    configure_storage();fs=fsspec.filesystem('s3');install();state,origin=recover(step,fs)
    tmp=Path('/tmp/layer-cal.npz');fs.get_file(ROOT+'/inputs/native-packed.npz',str(tmp))
    with np.load(tmp) as z:train=tuple(j.asarray(z[a][:ncal]) for a in ('tokens','weights','segments'))
    fs.get_file(ROOT+'/inputs/native-confirm3.npz',str(tmp))
    with np.load(tmp) as z:hold=tuple(z[a] for a in ('tokens','weights','segments'))
    initial=state if step==4001 else recover(4001,fs)[0]
    bank=ready(fit(initial.model,train,mode));cost=[]
    for _ in range(4):
        t=time.perf_counter();bank=ready(fit(initial.model,train,mode));cost.append(1000*(time.perf_counter()-t))
    keys=[jax.tree_util.keystr(p) for p,_ in jax.tree_util.tree_leaves_with_path(eqx.filter(state.model,eqx.is_inexact_array))]
    rng=np.random.default_rng(771+step);pairs=[]
    for i in range(128):
        left,right=[tuple(j.asarray(a[ix]) for a in hold) for ix in rng.integers(len(hold[0]),size=(2,4))]
        pairs.append(np.asarray(evaluate(state.model,left,right,bank,mode,jax.random.key(i))).tolist())
    dst=ROOT+f'/layer-prediction/s{step}-c{ncal}-m{mode}.json'
    fs.pipe(dst,json.dumps(dict(step=step,cal=ncal,mode=mode,params=keys,pairs=pairs,fit_ms=cost,source=origin)).encode())
if __name__=='__main__':main(*map(int,sys.argv[1:]))

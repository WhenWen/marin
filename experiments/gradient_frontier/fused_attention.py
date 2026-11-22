"""Use the same Pallas fused attention in both estimator and baseline; preserve checkpoint projections and scales."""
from jax.experimental.pallas.ops.gpu.attention import mha
import haliax as h
from levanter.layers.attention import Attention

def apply(self,x,mask=None,*,key=None,pos_ids=None):
    q,k,v=self._compute_qkv(x,key=key,pos_ids=pos_ids)
    q=q.flatten_axes(('kv_head','q_heads_per_group'),'heads').rearrange(('batch','position','heads','head_size'))
    k=k.rearrange(('batch','position','kv_head','head_size'))
    v=v.rearrange(('batch','position','kv_head','head_size'))
    seg=mask.segment_ids if mask is not None else None
    segments=None if seg is None else seg[0].rearrange(('batch','position')).array
    value=mha(q.array,k.array,v.array,segments,sm_scale=self.config.scaling_factor,causal=True)
    return self.o_proj(h.named(value,q.axes).astype(x.dtype),key=key)
def install():
    Attention.__call__=apply

def install_projections():
    """Fuse the checkpoint's bias-free QKV and gate/up multiplies in both benchmark arms."""
    import jax.numpy as j
    from levanter.models.llama import LlamaMlp
    def qkv(self,x,*,key=None,pos_ids=None):
        x=x.rearrange((...,'embed'));d=x.array.shape[-1]
        layers=(self.q_proj,self.k_proj,self.v_proj)
        ws=[p.weight.array.reshape((-1,d)) for p in layers]
        sizes=[w.shape[0] for w in ws]
        hidden=x.array@j.concatenate(ws,axis=0).T
        pieces=j.split(hidden,(sizes[0],sizes[0]+sizes[1]),axis=-1)
        def named(v,l):
            outs=l.weight.axes[:-1]
            return h.named(v.reshape((*v.shape[:-1],*(a.size for a in outs))),(*x.axes[:-1],*outs))
        q,k,v=[named(v,l) for v,l in zip(pieces,layers)]
        if self.config.qk_norm is not None:q,k=self.q_norm(q),self.k_norm(k)
        if self.rot_embs is not None:
            if pos_ids is None:pos_ids=h.arange(x.resolve_axis('position'))
            q=self.rot_embs(q,pos_ids).astype(x.dtype);k=self.rot_embs(k,pos_ids).astype(x.dtype)
        return q,k,v
    def mlp(self,x,*,key=None):
        x=x.rearrange((...,'embed'))
        w=j.concatenate((self.gate_proj.weight.array,self.up_proj.weight.array),axis=0)
        a,b=j.split(x.array@w.T,2,axis=-1)
        axes=(*x.axes[:-1],self.gate_proj.weight.axes[0])
        return self.down_proj(self.act(h.named(a,axes))*h.named(b,axes),key=key)
    Attention._compute_qkv=qkv
    LlamaMlp.__call__=mlp

"""Native TransferLM layers with reusable forward features and a full parameter gradient."""
import equinox as eqx
import haliax as h
import jax
import jax.numpy as j
import numpy as np
from jax.flatten_util import ravel_pytree
from levanter.layers.attention_mask import AttentionMask
from experiments.hybrid_transfer.model import TransferConfig

def restore(path):
    config=TransferConfig(hidden_dim=128, intermediate_dim=448, num_layers=4, num_heads=2,
        num_kv_heads=2,max_seq_len=1024,initializer_range=0.02,layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,scan_layers=True)
    tree=jax.eval_shape(lambda:config.build(h.Axis('vocab',128256),key=jax.random.key(0)))
    pairs,struct=jax.tree_util.tree_flatten_with_path(tree)
    with np.load(path) as z:
        leaves=[]
        for path,shape in pairs:
            key='model/'+'/'.join(k.name for k in path if hasattr(k,'name'))
            a=z[key]
            assert a.shape==shape.shape
            leaves.append(j.asarray(a))
        assert len(z.files)==len(leaves)+1 and int(z['step'])==4001
    return struct.unflatten(leaves)

def inputs(x):
    return h.named(x,(h.Axis('batch',x.shape[0]),h.Axis('position',x.shape[1])))

def pooled(hidden):
    return h.mean(hidden.astype(j.float32),axis='position').rearrange(('batch','embed')).array

def forward(model,tokens,depth,loss,stats):
    ids=inputs(tokens[:,:-1])
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=m.embeddings.embed(ids)
    phi=None
    if depth==0:phi=pooled(x)
    if depth==-2:phi=stats[tokens[:,:-1]].mean(1,keepdims=True)
    for k,layer in enumerate(m.transformer.layers.unstacked(),1):
        if not loss and k>depth:break
        x=layer(x,AttentionMask.causal())
        if k==depth:phi=pooled(x)
    if not loss:return phi
    x=m.transformer.norm(x)
    logits=(m.lm_head(x).rearrange(('batch','position','vocab')).array / m.config.width_multiplier).astype(j.float32)
    targets=tokens[:,1:]
    nll=(jax.nn.logsumexp(logits,axis=-1)-j.take_along_axis(logits,targets[:,:,None],axis=2)[:,:,0]).mean()
    return nll,phi

def gradient(m,tokens,depth,stats):
    (nll,feature),g=eqx.filter_value_and_grad(forward,has_aux=True)(m,tokens,depth,True,stats)
    return ravel_pytree(g)[0],feature,nll

batch_grad=eqx.filter_jit(gradient)
features=eqx.filter_jit(lambda m,tokens,depth,stats:forward(m,tokens,depth,False,stats))
per_grad=eqx.filter_jit(lambda m,tokens,stats:jax.vmap(lambda t:gradient(m,t[None],-1,stats)[0])(tokens))

@eqx.filter_jit
def corrected(m,tokens,ref,depth,stats,projection,beta,center):
    g,x,_=gradient(m,tokens,depth,stats)
    bx=x.mean(0)
    if ref is None: diff=bx-center
    else:
        rx=forward(m,ref,depth,False,stats).mean(0)
        diff=(bx-rx)*(ref.shape[0]/(tokens.shape[0]+ref.shape[0]))
    return g-((diff@projection)@beta)

plain=eqx.filter_jit(lambda m,tokens,stats:gradient(m,tokens,-1,stats)[0])

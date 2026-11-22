"""Input-only control variate from singleton-context gradients grouped by previous token."""
import equinox as eqx
import jax
import jax.numpy as j
from jax.flatten_util import ravel_pytree
from .multifidelity import activations,shortlist,objective
from .adaptive_correction import loss as corrected_loss

@eqx.filter_jit
def setup(model,cal,k):
    ids=shortlist(cal,k,128256)
    rank=j.full((128256,),k,dtype=j.int32).at[ids].set(j.arange(k))
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    single=j.stack((ids,ids),axis=1)
    h=activations(m,single)[:,0]
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    expected=(jax.nn.softmax((h@w.T).astype(j.float32),axis=-1).astype(j.bfloat16)@w).astype(j.float32)
    return ids,rank,expected

def statistic(model,tokens,bank,weight=None):
    ids,rank,avg=bank
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    x=tokens[:,:-1];y=tokens[:,1:]
    groups=rank[x].reshape(-1);k=ids.shape[0]
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    a=j.ones(x.shape,j.float32)/x.size if weight is None else weight
    a=a.reshape(-1)
    neg=jax.ops.segment_sum(a,groups,k)
    pos=jax.ops.segment_sum(a[:,None]*w[y.reshape(-1)].astype(j.float32),groups,k)
    cot=jax.lax.stop_gradient(neg[:,None]*avg-pos)
    single=j.stack((ids,ids),axis=1)
    h=activations(m,single)[:,0,:].astype(j.float32)
    return (h*cot).sum()

@eqx.filter_jit
def probe(m,tok,bank):
    return ravel_pytree(eqx.filter_grad(statistic)(m,tok,bank))[0]
@eqx.filter_jit
def mixed(m,tok,bank,head,key,b,power):
    mask=j.ones(tok[:,:-1].shape,j.float32)/tok[:,:-1].size
    mask=mask.at[:b].add(-1./(b*mask.shape[1]))
    def estimate(net):
        return corrected_loss(net,tok[:b],key,head,32,1)+power*statistic(net,tok,bank,mask)
    return ravel_pytree(eqx.filter_grad(estimate)(m))[0]

"""Fit group log partitions and their hidden-state derivatives using true full-vocabulary teacher moments."""
import equinox as eqx
import jax
import jax.numpy as j
from . import group_regression as groups
from .tail import trunk


def levels(kind):
    return (8 if kind in (152,153,156,157) else 2,)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array;x=trunk(m,data).reshape(-1,w.shape[-1]);num,d=x.shape
    width=256
    def importance_step(acc,xx):
        logits=(xx@w.T).astype(j.float32);p=jax.nn.softmax(logits,axis=-1)
        return acc+j.sum(p*p,axis=0),None
    blocks=x.reshape(-1,128,d)
    importance,_=jax.lax.scan(importance_step,j.zeros(w.shape[0],j.float32),blocks)
    ranking=j.argsort(-importance);head=ranking[:k];others=ranking[k:]
    sub=x[::max(1,num//16)][:16]
    keys=(sub@w[others].T).astype(j.float32)
    signature=j.sum((keys>j.median(keys,-1,keepdims=True)).astype(j.int32)*(2**j.arange(16))[:,None],0)
    order=others[j.argsort(signature)];ids=j.pad(order,(0,(-len(order))%width),constant_values=w.shape[0])
    g=len(ids)//width;table=j.pad(w,((0,1),(0,0)))[ids].reshape(g,width,d)
    mask=ids.reshape(g,width)<w.shape[0]
    def moments(acc,xx):
        logits=(xx@table.reshape(-1,d).T).reshape(len(xx),g,width).astype(j.float32)
        logits=j.where(mask,logits,-j.inf);zz=jax.nn.logsumexp(logits,-1)
        p=j.exp(logits-zz[:,:,None]).astype(j.bfloat16)
        # E[w | group,h] is the exact derivative of that group's log partition.
        mu=j.einsum('igv,gvd->igd',p,table).astype(j.float32)
        inputs=groups.phi(xx)
        xx32=xx.astype(j.float32)
        return (acc[0]+inputs.T@zz,acc[1]+mu.sum(0),acc[2]+j.einsum('id,igd->gd',xx32,mu),acc[3]+p.astype(j.float32).sum(0)),None
    init=(j.zeros((2*d+1,g)),j.zeros((g,d)),j.zeros((g,d)),j.zeros((g,width)))
    (xy,mu,xmu,prior),_=jax.lax.scan(moments,init,blocks)
    inputs=groups.phi(x);sd=j.maximum(.01,j.sqrt((inputs*inputs).mean(0)));ff=inputs/sd
    mean_z=xy[0]/num
    rhs=(xy-inputs.sum(0)[:,None]*mean_z)/sd[:,None]
    beta=jax.scipy.linalg.solve(ff.T@ff+.03*num*j.eye(ff.shape[1]),rhs,assume_a='pos')/sd[:,None]
    beta=beta.at[0].add(mean_z)
    mx=x.astype(j.float32).mean(0);sx=(x.astype(j.float32)**2).mean(0);mu=mu/num
    curvature=(xmu/num-mx[None,:]*mu)/j.maximum(.001,sx-mx*mx)[None,:]
    c=.5*curvature.T;b=mu.T-2*mx[:,None]*c
    a=mean_z-mx@b-sx@c
    deriv_fit=j.concatenate((a[None,:],b,c),0)
    mix={150:.25,151:.65,152:.65,153:1.,154:.25,155:.65,156:.65,157:1.}[kind]
    learned=(1-mix)*beta+mix*deriv_fit
    membership=j.full((w.shape[0]+1,),-1,j.int32).at[ids].set(j.repeat(j.arange(g),width))
    return head,ids,membership,(prior/num).astype(j.bfloat16),learned.astype(j.bfloat16)


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000)
    return groups.objective(x,w,y,wt,key,((115 if kind>=154 else 113)*100000+k,n),bank)

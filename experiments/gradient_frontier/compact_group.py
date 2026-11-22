"""Share compact group and singleton probabilities between prediction, sampling and gradient computation."""
from functools import partial
import jax
import jax.numpy as j
from . import group_regression as groups


def parent(kind):
    return (113 if kind%2==0 else 115)


def levels(kind):
    return groups.levels(parent(kind))


def fit(model,data,k,kind):
    return groups.fit(model,data,k,parent(kind))


def forward(head,group,pos,extra):
    dtype=head.dtype
    shift=jax.lax.stop_gradient(j.maximum(j.maximum(head.max(-1),group.max(-1)),pos))
    eh=j.exp((head-shift[:,:,None]).astype(j.float32)).astype(dtype)
    eg=j.exp((group-shift[:,:,None]).astype(j.float32)).astype(dtype)
    ep=j.where(extra,j.exp(pos.astype(j.float32)-shift),0.)
    sg=eg.sum(-1,dtype=j.float32);tot=eh.sum(-1,dtype=j.float32)+sg+ep
    return (shift+j.log(tot),sg/tot,1./tot),(eh,eg,ep,tot)


@jax.custom_vjp
def partition(head,group,pos,extra):
    return forward(head,group,pos,extra)[0]


def reverse(saved,adj):
    eh,eg,ep,tot=saved
    scale=adj[0]/tot
    return ((eh.astype(j.float32)*scale[:,:,None]).astype(eh.dtype),
            (eg.astype(j.float32)*scale[:,:,None]).astype(eg.dtype),scale*ep,None)
partition.defvjp(forward,reverse)


def objective(x,w,y,wt,key,config,bank):
    code,n=config;kind,k=divmod(code,100000);b,t,d=x.shape
    ids,sorted_ids,bucket,prior,beta=bank
    pred=groups.phi(x).astype(j.bfloat16)@beta
    if parent(kind)==115:
        g=prior.shape[0];width=prior.shape[1]
        grouped=j.pad(w,((0,1),(0,0)))[sorted_ids].reshape(g,width,d)
        avg=j.einsum('gw,gwd->gd',prior,grouped)
        pred=pred+jax.lax.stop_gradient(x)@(avg-jax.lax.stop_gradient(avg)).T
    short=x@w[ids].T
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z,mass,top=partition(short,pred,pos,bucket[y]>=0)
    cheap=z-pos
    score=(.04+j.sqrt(mass)+.12*(1-j.minimum(top,1.))) if kind<164 else (.025+mass+.12*(1-j.minimum(top,1.)))
    width=t//n;s=jax.lax.stop_gradient(score.reshape(b,n,width));pr=s/s.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));i=j.minimum((u>=j.cumsum(pr,-1)).sum(-1),width-1)
    sel=j.arange(n)[None,:]*width+i;inv=1/j.take_along_axis(pr,i[:,:,None],-1)[:,:,0]
    h=j.take_along_axis(x,sel[:,:,None],1);target=j.take_along_axis(y,sel,1)
    logits=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    return ((cheap*wt).sum()+((exact-j.take_along_axis(cheap,sel,1))*j.take_along_axis(wt,sel,1)*inv).sum())/wt.sum()

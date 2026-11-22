"""Fit positive vocabulary multiplicities against the exact partition and its directional derivatives."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):
    return (2 if kind<224 else 4,)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    x=trunk(m,data).reshape(-1,w.shape[-1])[::2]
    # Independent of all measurement batches. Each row describes the exact conditional softmax.
    logits=(x@w.T).astype(j.float32);p=jax.nn.softmax(logits,-1)
    energy=j.sum(p*p,axis=0)
    head=jax.lax.top_k(energy,k)[1]
    avail=j.ones(len(w),bool).at[head].set(False)
    tailprob=jax.nn.softmax(j.where(avail,logits,-j.inf),-1)
    mass=j.sqrt(j.maximum(0.,j.sum(tailprob*tailprob,axis=0)))
    M=512 if kind%4<2 else 1024
    # Exponential races sample distinct tail rows. The next arrival bounds their inclusion weights.
    u=jax.random.uniform(jax.random.key(4175),(len(w),),minval=1.e-7)
    priority=mass/(-j.log(u));vals,words=jax.lax.top_k(priority,M+1)
    sel=words[:M];tau=j.maximum(vals[-1],1e-9)
    prior=1/j.maximum(1.e-7,-j.expm1(-mass[sel]/tau))
    design=tailprob[:,sel]*prior
    axes=jax.random.rademacher(jax.random.key(712),(w.shape[-1],8),dtype=j.float32)/j.sqrt(float(w.shape[-1]))
    proj=w@axes.astype(j.bfloat16);proj=proj.astype(j.float32)
    target=(tailprob.astype(j.bfloat16)@proj.astype(j.bfloat16)).astype(j.float32)
    # For each direction, match sum_j p_j (w_j - E[w]) = 0, using the same fitted positive counts.
    diff=proj[sel][None,:,:]-target[:,None,:]
    if kind%2:diff=diff/j.maximum(.01,j.sqrt(j.mean(diff*diff,axis=(0,1),keepdims=True)))
    else:diff=j.zeros_like(diff)
    feats=(design[:,:,None]*diff).transpose(2,0,1).reshape(-1,M)
    G=design.T@design+.12*(feats.T@feats);rhs=design.sum(0)
    reg=.08*j.maximum(j.diag(G),.005)
    G=G+j.diag(reg);rhs=rhs+reg
    # Projected convex quadratic fit with diagonal preconditioning, initialized at sampling weights.
    sd=j.sqrt(j.maximum(j.diag(G),1e-8));A=G/sd[:,None]/sd[None,:];rhs=rhs/sd
    lipschitz=j.maximum(1.,j.max(j.sum(j.abs(A),1)))
    def solve(i,v):
        last,extrap=v
        nxt=j.clip(extrap+(rhs-A@extrap)/lipschitz,.03*sd,12.*sd)
        return nxt,nxt+i/(i+3.)*(nxt-last)
    sol,_=jax.lax.fori_loop(0,32,solve,(sd,sd))
    weights=prior*sol/sd
    known=j.zeros(len(w),bool).at[head].set(True).at[sel].set(True)
    return head,sel,weights,known


def objective(x,w,y,wt,key,config,bank):
    code,n=config;b,t,d=x.shape
    ids,sel,multiple,known=bank;stop=jax.lax.stop_gradient
    head=(x@w[ids].T).astype(j.float32)
    # Only one copy of each real row receives a readout derivative, while x sees the fitted multiplicity.
    ws=w[sel];eff=stop(ws)+(ws-stop(ws))*(1/j.maximum(1.,multiple)).astype(ws.dtype)[:,None]
    tail=(x@eff.T).astype(j.float32)+j.log(multiple)
    zh=jax.nn.logsumexp(head,-1);zt=jax.nn.logsumexp(tail,-1)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    z=j.logaddexp(j.logaddexp(zh,zt),j.where(known[y],-j.inf,pos))
    cheap=z-pos;width=t//n
    p=stop(.04+j.sqrt(j.exp(zt-z))+.12*(1.-j.minimum(1.,j.exp(pos-z)))).reshape(b,n,width)
    p=p/p.sum(-1,keepdims=True)
    u=jax.random.uniform(key,(b,n,1));at=j.minimum((u>=j.cumsum(p,-1)).sum(-1),width-1)
    iw=1/j.take_along_axis(p,at[:,:,None],-1)[:,:,0];selpos=j.arange(n)[None,:]*width+at
    hid=j.take_along_axis(x,selpos[:,:,None],1);target=j.take_along_axis(y,selpos,1)
    logits=(hid@w.T).astype(j.float32);exact=jax.nn.logsumexp(logits,-1)-j.take_along_axis(logits,target[:,:,None],-1)[:,:,0]
    diff=(exact-j.take_along_axis(cheap,selpos,1))*j.take_along_axis(wt,selpos,1)*iw
    return ((wt*cheap).sum()+diff.sum())/wt.sum()

"""Cover the tail log-partition by teacher tangent planes and learn shared positive curvature between anchors."""
import equinox as eqx
import jax
import jax.numpy as j
from .tail import trunk


def levels(kind):return (32 if kind%2 else 8,)


@eqx.filter_jit
def fit(model,data,k,kind):
    m=jax.tree.map(lambda a:a.astype(j.bfloat16),model)
    w=m.get_lm_head().rearrange(('vocab','embed')).array
    h=trunk(m,data).reshape(-1,w.shape[1])[::4];N,D=h.shape
    prob=jax.nn.softmax((h[::max(1,N//128)]@w.T).astype(j.float32),axis=-1)
    ids=jax.lax.top_k((prob*prob).mean(0),k)[1]
    member=j.zeros(len(w),bool).at[ids].set(True)
    def teach(x):
        tail=j.where(member,-j.inf,(x@w.T).astype(j.float32))
        z=jax.nn.logsumexp(tail,-1);p=j.exp(tail-z[:,None])
        return z,(p.astype(j.bfloat16)@w).astype(j.float32)
    z,mu=jax.lax.map(teach,h.reshape(-1,64,D));z=z.ravel();mu=mu.reshape(N,D);h=h.astype(j.float32)
    R=128 if kind in (320,321,324,325) else 256
    first=j.argmin(j.sum((h-h.mean(0))**2,1))
    def add_anchor(i,carry):
        selected,cover,idx=carry;selected=selected.at[i].set(idx)
        tangent=z[idx]+(h-h[idx])@mu[idx]
        cover=j.maximum(cover,tangent)
        gap=j.maximum(.0001,z-cover)
        choice=j.argmax(2*j.log(gap)+jax.random.gumbel(jax.random.fold_in(jax.random.key(7399),i),(N,)))
        return selected,cover,choice
    chosen,_,_=jax.lax.fori_loop(0,R,add_anchor,(j.zeros(R,j.int32),j.full(N,-j.inf),first))
    centers=h[chosen];grad=mu[chosen];bias=z[chosen]-j.sum(centers*grad,axis=1)
    group=j.argmax(h@grad.T+bias,axis=1)
    delta=h-centers[group];response=mu-grad[group]
    cov=delta.T@delta;reg=.05*j.trace(cov)/D+1e-5
    matrix=j.linalg.solve(cov+reg*j.eye(D),delta.T@response)
    ev,Q=j.linalg.eigh((matrix+matrix.T)/2);Q=Q[:,-16:]*j.sqrt(j.maximum(0.,ev[-16:]))
    projected=delta@Q
    raw=z[chosen][group]+(delta*grad[group]).sum(1)+.5*(projected*projected).sum(1)
    count=j.bincount(group,length=R);gap=z-raw
    adj=(jax.ops.segment_sum(gap,group,R)+8*gap.mean())/(count+8)
    anchor_logits=j.where(member,-j.inf,(centers.astype(j.bfloat16)@w.T).astype(j.float32))
    tail_p=jax.nn.softmax(anchor_logits,axis=1)
    top,words=jax.lax.top_k(tail_p,32)
    return ids,centers.astype(j.bfloat16),grad.astype(j.bfloat16),bias,Q.astype(j.bfloat16),adj,words,top.astype(j.bfloat16)


def objective(x,w,y,wt,key,config,bank,forced=None,return_prob=False):
    code,n=config;kind,k=divmod(code,100000);B,T,D=x.shape
    ids,centers,grad,bias,Q,adj,words,top=bank
    stop=jax.lax.stop_gradient
    lookup=(x@grad.T).astype(j.float32)+bias
    cell=stop(j.argmax(lookup,axis=-1));lin=j.take_along_axis(lookup,cell[:,:,None],axis=-1)[:,:,0]
    coord=(x@Q).astype(j.float32)-(centers@Q).astype(j.float32)[cell]
    tail=lin+adj[cell]+.5*j.sum(coord*coord,axis=-1)
    head=jax.nn.logsumexp((x@w[ids].T).astype(j.float32),axis=-1)
    z=j.logaddexp(head,tail);mass=j.exp(tail-z)
    pos=j.einsum('btd,btd->bt',x,w[y]).astype(j.float32)
    cheap=z-pos
    width=T//n;score=.05+j.sqrt(mass)+.15*(1.-j.minimum(1.,j.exp(pos-z)))
    p=stop(score.reshape(B,n,width));p=p/p.sum(-1,keepdims=True)
    at=j.minimum((jax.random.uniform(key,(B,n,1))>=j.cumsum(p,axis=-1)).sum(-1),width-1) if forced is None else forced
    ix=at+j.arange(n)[None,:]*width;ar=j.arange(B)[:,None]
    inv=1./j.take_along_axis(p,at[:,:,None],axis=-1)[:,:,0]
    coeff=wt.at[ar,ix].add(-wt[ar,ix]*inv)/wt.sum()
    h=x[ar,ix];yy=y[ar,ix];scores=(h@w.T).astype(j.float32)
    exact=jax.nn.logsumexp(scores,axis=-1)-j.take_along_axis(scores,yy[:,:,None],axis=-1)[:,:,0]
    total=(coeff*cheap).sum()+(wt[ar,ix]*inv*exact).sum()/wt.sum()
    if kind>=324:
        # Aggregate token contributions before touching the anchor's small readout dictionary.
        vec=jax.ops.segment_sum((coeff[:,:,None]*mass[:,:,None]*x).reshape(-1,D),cell.ravel(),len(centers))
        table=w[words];affine=j.einsum('rd,rkd->rk',stop(vec).astype(table.dtype),table-stop(table)).astype(j.float32)
        total+=(affine*top).sum()
    return (total,p) if return_prob else total

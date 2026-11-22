"""Identify which parameter groups and document norms carry gradient noise."""
import json
import fsspec,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .multifidelity import per_example,shortlist
from .kernels import restore
from .run import ROOT,provenance

def main():
    configure_storage();fs=fsspec.filesystem('s3');folder,_=provenance(fs)
    model=restore(folder/'weights.npz');data=np.load(folder/'windows.npz')['tokens']
    ids=shortlist(j.asarray(data[:1024]),8192,128256)
    groups=[];offset=0
    for path,leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        name='/'.join(k.name for k in path if hasattr(k,'name'))
        groups.append((name,offset,offset+leaf.size));offset+=leaf.size
    @jax.jit
    def moments(g):
        return j.stack([j.sum(g[:,s:e]**2,axis=-1) for _,s,e in groups],axis=-1)
    gsum=0.;tots=[];htots=[]
    for idx in range(2048,4096,4):
        t=j.asarray(data[idx:idx+4])
        g=per_example(model,t,ids,'full')
        p=per_example(model,t,ids,'cheap')
        gsum+=g.sum(0)
        tots.extend(np.array(moments(g)))
        htots.extend(np.array(moments(g-p)))
    n=len(tots);square=np.array(moments((gsum/n)[None]))[0];tots=np.array(tots)
    var=tots.mean(0)-square
    norm=np.sqrt(tots.sum(1));optimal=(norm.mean()**2-square.sum())/var.sum()
    out=dict(optimal_document_importance_relative_variance=float(optimal),
             groups=[dict(name=name,variance=float(var[k]),fraction=float(var[k]/var.sum()),
                        residual_second_moment=float(np.mean(htots,axis=0)[k])) for k,(name,_,_) in enumerate(groups)])
    print(json.dumps(out),flush=True)
    fs.pipe(ROOT+'/diagnostics/groups.json',json.dumps(out).encode())
if __name__=='__main__':main()

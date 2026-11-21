"""Compiler-estimated arithmetic for the exact benchmark shapes, before backend fusion."""
import json
import fsspec,jax
import jax.numpy as j
import numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .count_ops import matrix_flops
from .kernels import restore,plain,corrected
from .run import ROOT,provenance
configure_storage()
fs=fsspec.filesystem('s3')
folder,_=provenance(fs)
m=restore(folder/'weights.npz')
data=j.asarray(np.load(folder/'windows.npz')['tokens'][:128])
stats=j.ones(128256)
bank=j.ones((8,33786496))
rows=[]
def append(lower,b,d,r,ratio):
    count=matrix_flops(lower)
    assert count>0
    rows.append(dict(b=b,depth=d,rank=r,pool_ratio=ratio,matmul_flops=count))
for b in [1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,24,32]:
    append(plain.lower(m,data[:b],stats),b,-1,0,0)
for b in (4,8,16):
    for d in (-2,0,1,4):
        for r in ((1,) if d==-2 else (1,4,8)):
            size=1 if d==-2 else 128
            proj=j.ones((size,r))
            for ratio in ((0,) if d<=0 else (2,4)):
                ref=data[b:ratio*b] if ratio else None
                append(corrected.lower(m,data[:b],ref,d,stats,proj,bank[:r],j.ones(size)),b,d,r,ratio)
fs.pipe(ROOT+'/flops.json',json.dumps(dict(device=jax.devices()[0].device_kind,rows=rows)).encode())
print('FLOP analysis complete',len(rows),flush=True)

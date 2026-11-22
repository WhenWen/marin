"""Count dense GEMM FLOPs on native 1,024-token packed loss and the optimizer."""
import json
import equinox as eqx,fsspec,jax,jax.numpy as j
from experiments.hybrid_transfer.storage import configure_storage
from .native_eval import recover,objective
from .count_ops import matrix_flops
from .run import ROOT

@eqx.filter_jit
def gradient(m,d,ids,key,c):return eqx.filter_grad(objective)(m,d,ids,key,c)
@eqx.filter_jit
def update(s,g,key):return s.take_step(g,key=key)[0]
def main():
    configure_storage();fs=fsspec.filesystem('s3');state,_=recover(4001,fs)
    import levanter.layers.attention as attention
    attention.default_attention_type=lambda: attention.AttentionBackend.VANILLA
    key=jax.random.PRNGKey(0);ids=j.arange(16384);rows=[]
    for b in (1,4,8,16,32):
        data=(j.ones((b,1024),j.int32),j.ones((b,1024),j.float32),j.zeros((b,1024),j.int32))
        for c in ((0,0),(208192,128),(716384,64)):
            rows.append(dict(b=b,method=str(c),
                             dense_flops=matrix_flops(gradient.lower(state.model,data,ids,key,c))))
    out=dict(rows=rows,attention='Dense causal dot-product reference, includes masked entries; fused kernels can skip tiles.')
    try:out['optimizer_dense_flops']=matrix_flops(update.lower(state,state.model,key))
    except ValueError as e:out['optimizer_count_issue']=str(e)
    fs.pipe(ROOT+'/native-matrix-flops.json',json.dumps(out).encode())
    print('NATIVE DENSE ARITHMETIC',out,flush=True)
if __name__=='__main__':main()

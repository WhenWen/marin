"""Measure busy GPU stream time paired with untraced wall clock on a restored optimizer state."""
from collections import defaultdict
from pathlib import Path
import gzip,json,sys,time
import fsspec,jax,jax.numpy as j,numpy as np
from experiments.hybrid_transfer.storage import configure_storage
from .run import ROOT,ready
from .native_eval import recover
from .confirm_goal import one
from .fused_attention import install

def main(s,k,n,v,B,rep):
    configure_storage();fs=fsspec.filesystem('s3');install();state,_=recover(s,fs)
    path=Path('/tmp/device-trace-data.npz');fs.get_file(ROOT+'/inputs/native-confirm2.npz',str(path))
    with np.load(path) as z:hold=tuple(z[a] for a in ('tokens','weights','segments'))
    ids=ready(j.arange(k));c=(v*100000+k,n);rng=np.random.default_rng(s+rep*417)
    sizes=range(max(1,B//4),max(5,3*B//4+1))
    cases=[('base',b) for b in sizes]+[('manual',b) for b in sizes]+[('candidate',B)]
    pools=[]
    for it in range(24):
        ii=rng.integers(len(hold[0]),size=B)
        pools.append(({b:tuple(j.asarray(a[ii[:b]]) for a in hold) for b in set(b for _,b in cases)},jax.random.PRNGKey(s+it)))
    ready(pools)
    for m,b in cases:
        for d,key in pools[:2]:ready(one(state,d[b],ids,key,m,c))
    wall=defaultdict(list)
    for d,key in pools[2:]:
        for i in rng.permutation(len(cases)):
            m,b=cases[i];t=time.perf_counter();ready(one(state,d[b],ids,key,m,c));wall[(m,b)].append(1000*(time.perf_counter()-t))
    dr=Path('/tmp/grad-cupti');dr.mkdir(exist_ok=True)
    with jax.profiler.trace(str(dr),create_perfetto_trace=True):
        for it,(d,key) in enumerate(pools[2:10]):
            for i in rng.permutation(len(cases)):
                m,b=cases[i]
                with jax.profiler.TraceAnnotation(f'FRONTIER-{m}-{b}-{it}'):
                    ready(one(state,d[b],ids,key,m,c))
    files=list(dr.rglob('perfetto_trace.json.gz'));assert len(files)==1,list(dr.rglob('*'))
    with gzip.open(files[0],'rt') as inp:events=json.load(inp)['traceEvents']
    proc={e['pid']:e['args']['name'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    threads={(e['pid'],e['tid']):e['args']['name'] for e in events if e.get('ph')=='M' and e.get('name')=='thread_name'}
    gpu_streams={p for p,t in threads.items() if ('GPU' in proc.get(p[0],'').upper() or 'GB200' in proc.get(p[0],'').upper()) and 'stream' in t.lower()}
    gpu=[(e['ts'],e['ts']+e['dur']) for e in events if e.get('ph')=='X' and (e['pid'],e['tid']) in gpu_streams and e.get('dur',0)>0]
    scopes=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('FRONTIER-')]
    family='device-trace-h100' if 'H100' in jax.devices()[0].device_kind else 'device-trace'
    dst=ROOT+f'/{family}/k{k}-n{n}-v{v}-b{B}-s{s}-r{rep}'
    meta=dict(procs=proc,threads=[(a,b,t) for (a,b),t in threads.items()],gpu_records=len(gpu),scopes=len(scopes))
    print('TRACE META',list(proc.values()),[(t) for p,t in threads.items() if p in gpu_streams],len(gpu),len(scopes),flush=True)
    if not gpu or not scopes:
        fs.pipe(dst+'-metadata.json',json.dumps(meta).encode());fs.put_file(str(files[0]),dst+'.json.gz')
        raise RuntimeError('GPU stream or synchronized host scope not identified')
    dev=defaultdict(list)
    for e in scopes:
        _,m,b,_=e['name'].split('-');lo=e['ts'];hi=lo+e['dur']
        spans=sorted((max(lo,a),min(hi,bb)) for a,bb in gpu if lo<bb and a<hi)
        if not spans:raise ValueError(e['name'])
        end=lo;busy=0.
        for a,bb in spans:
            busy+=max(0.,bb-max(a,end));end=max(end,bb)
        dev[(m,int(b))].append(busy/1000)
    out=dict(step=s,k=k,n=n,v=v,B=B,rep=rep,meta=meta,
             timing=[dict(method=m,b=b,ms=ms,device_ms=dev[(m,b)]) for (m,b),ms in wall.items()])
    fs.pipe(dst+'.json',json.dumps(out).encode());print('GPU BUSY',[(a['method'],a['b'],np.median(a['device_ms'])) for a in out['timing']],flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))

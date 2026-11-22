"""Compare every saved array and the production minibatch after an independent process replay."""
import json
import time
import fsspec
import numpy as np
import tensorstore as ts
from experiments.hybrid_transfer.storage import configure_storage
from levanter.tensorstore_serialization import build_kvstore_spec
from .run import ROOT

def run(method="native"):
    configure_storage()
    fs=fsspec.filesystem('s3')
    prefix=ROOT+('/native-muonw' if method=='native' else '/muonw-'+method)
    def read(path):
        return json.loads(fs.cat_file(path))
    a=read(prefix+'/completed/4003.json')
    b=read(prefix+'/verification/4003.json')
    for key in ('data_sha256','next_step','train_loss','validation','lr','adam_lr','loss'):
        if key in a or key in b: assert a[key]==b[key],key
    assert a['run_id']!=b['run_id']
    ac,bc=[r['checkpoint'] for r in (a,b)]
    def list_arrays(p):
        return {x['path']:x['shape'] for x in read(p.removeprefix('s3://')+'/manifest.json')['arrays']}
    aa=list_arrays(ac)
    assert aa==list_arrays(bc)
    def array(p,name):
        store={'driver':'ocdbt','base':build_kvstore_spec(p+'/'),'path':name}
        return np.asarray(ts.open({'driver':'zarr3','kvstore':store},open=True).result().read().result())
    different=[]
    scalars=0
    for name in sorted(aa):
        x=array(ac,name)
        y=array(bc,name)
        scalars+=x.size
        if x.dtype!=y.dtype or x.tobytes()!=y.tobytes():different.append(name)
    report=dict(arrays=len(aa),elements=int(scalars),different=different,original=a,replay=b,
                bitwise_match=not different)
    fs.pipe(prefix+'/replay-comparison.json',json.dumps(report).encode())
    print('INDEPENDENT FULL-STATE EQUALITY',len(aa),scalars,different,flush=True)
    assert not different
if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

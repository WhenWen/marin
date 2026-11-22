"""Start a distinct physical-board replicate after checking the original panel's complete replay receipt."""
import json
import sys

import fsspec
from experiments.hybrid_transfer.storage import configure_storage
from .fifty_crossover import configurations,name
from .run import ROOT


def bind(seed,panel):
    assert panel==10
    configure_storage();fs=fsspec.filesystem('s3')
    original=ROOT+f'/fifty-crossover-v1/GB200-p6-s{seed}'
    root=ROOT+f'/fifty-crossover-v1/GB200-p{panel}-s{seed}'
    read=lambda path:json.loads(fs.cat_file(path))
    assert fs.exists(root+'/logging.json')
    proof=read(original+'/verified.json')
    ref=read(original+'/replay/commits/4003.json')
    prev=read(original+'/replay/commits/4002.json')
    assert proof['bitwise_match'] and proof['original']==ref['attempt'] and proof['digests']==ref['digests']
    assert proof['restored_from']==prev['checkpoints']
    assert set(ref['digests'])=={name(c) for c in configurations(panel)} and ref['bank']==read(original+'/bank.json')['sha256']
    if not fs.exists(root+'/bank.json'):
        fs.pipe(root+'/bank.json',fs.cat_file(original+'/bank.json'))
    assert read(root+'/bank.json')['sha256']==ref['bank']
    # Source receipt certifies code, seed, schedule and carry. This board runs a fresh trajectory from step 4001.
    receipt=dict(**proof,certificate_path=original+'/verified.json',replica_from_origin=True)
    fs.pipe(root+'/verified.json',json.dumps(receipt).encode())


if __name__=='__main__':bind(*map(int,sys.argv[1:]))

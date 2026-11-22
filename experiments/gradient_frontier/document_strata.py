"""Learn document strata from previous gradients and sample uniformly within equal-size strata."""
import io
import json
import os
import sys

import fsspec
import numpy as np
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .diagnostic_artifacts import publish
from .kernel_probe import ngram_kernel
from .run import ROOT


def strata(indices,coordinates,count):
    """Bisect at medians of the locally most variable projected feature."""
    groups=[indices]
    while len(groups)<count:
        children=[]
        for group in groups:
            points=coordinates[group]
            axis=np.argmax(points.var(axis=0))
            order=group[np.argsort(points[:,axis],kind='stable')]
            children.extend(np.split(order,2))
        groups=children
    return np.asarray(groups)


def score(y,groups,mean):
    # E(||estimate-population_mean||² | pilot pool), including variation of the pool mean.
    values=y[groups]
    center=values.mean(axis=1,keepdims=True)
    return np.square(values.mean(axis=(0,1))-mean).sum()+np.square(values-center).sum(axis=-1).mean()/len(groups)


def main(step):
    configure_storage()
    fs=fsspec.filesystem('s3')
    def load(path):
        with np.load(io.BytesIO(fs.cat_file(path))) as f:
            return {k:f[k] for k in f.files}
    def sketches(which):
        chunks=sorted([load(p) for p in fs.glob(ROOT+f'/document-sketch/native-packed-s{which}-*.npz')],key=lambda q:int(q['begin']))
        ids=np.concatenate([np.arange(int(q['begin']),int(q['begin'])+len(q['norms'])) for q in chunks])
        return ids,np.concatenate([q['sketch'][:,0] for q in chunks]).astype(float)
    ids,Y=sketches(step)
    train_ids,previous=sketches(4003)
    assert np.array_equal(ids,train_ids)
    tokens=load(ROOT+'/inputs/native-packed.npz')['tokens'][ids]
    cfg=dict(step=step,source_step=4003,seed=32971,pool_size=128,target='native trunk CountSketch',repetitions=1024)
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='learned-document-stratification',id=os.urandom(6).hex(),mode='online',config=cfg)
    gram=.5*(ngram_kernel(tokens,1)+ngram_kernel(tokens,2))
    out=dict(**cfg,wandb=run.url,results=[])
    for fold in (0,1):
        train=np.where((ids//64)%2!=fold)[0];test=np.where((ids//64)%2==fold)[0]
        old=previous[train]-previous[train].mean(0)
        _,basis=np.linalg.eigh(old.T@old)
        directions=basis[:,-16:]
        label=old@directions
        fitted=gram[np.ix_(test,train)]@np.linalg.solve(gram[np.ix_(train,train)]+np.eye(len(train)),label)
        current=Y[test];mean=current.mean(0)
        # The second score buys the previous gradient of each prospective document; it is a discovery comparator.
        features={'learned_ngram':fitted,'previous_exact_gradient':previous[test]@directions,'current_gradient_diagnostic':current@directions}
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,fold]))
        costs={(key,b):[] for key in features for b in (2,4,8,16)}
        total=np.square(current-mean).sum(axis=1).mean()
        for i in range(cfg['repetitions']):
            sample=rng.integers(len(test),size=cfg['pool_size'])
            for key,pred in features.items():
                for b in (2,4,8,16):
                    groups=strata(sample,pred,b)
                    costs[key,b].append(score(current,groups,mean))
        for (key,b),summ in costs.items():
            out['results'].append(dict(fold=fold,method=key,queried=b,variance=float(np.mean(summ)),iid_variance=total/b,ratio=(total/b)/np.mean(summ),mc_se=float(np.std(summ,ddof=1)/np.sqrt(len(summ)))))
        publish(fs,ROOT+f'/learned-document-strata/s{step}.json',out)
    run.log({'completed_folds':2});run.finish()


if __name__=='__main__':main(int(sys.argv[1]))

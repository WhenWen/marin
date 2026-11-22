"""Blocked, nested validation of nonparametric document-gradient predictors on true native gradient sketches."""
import io
import json
import os
import sys
from pathlib import Path
import fsspec
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
import wandb
from rigging.secrets import require_secret
from experiments.hybrid_transfer.storage import configure_storage
from .diagnostic_artifacts import publish
from .run import ROOT


def ngram_kernel(tokens,order):
    n,t=tokens.shape
    count=t-order+1;rows=np.repeat(np.arange(n),count)
    code=tokens[:,:count].astype(np.uint64)
    for i in range(1,order):code=code*np.uint64(1000003)+tokens[:,i:i+count].astype(np.uint64)
    cols=((code^(code>>17))%65536).astype(np.int32).reshape(-1)
    counts=sparse.csr_matrix((np.ones(len(cols),np.float32),(rows,cols)),shape=(n,65536))
    counts.sum_duplicates()
    counts.data=np.sqrt(counts.data)
    norms=np.sqrt(np.asarray(counts.multiply(counts).sum(1)).ravel())
    counts=sparse.diags(1./np.maximum(norms,1e-7))@counts
    return (counts@counts.T).toarray().astype(float)


def solve(K,train,test,Y):
    # Four ridge settings share one eigendecomposition.
    G=K[np.ix_(train,train)]
    cross=K[np.ix_(test,train)]
    mu=Y[train].mean(0)
    eig,v=eigh(G,check_finite=False)
    alpha=v.T@(Y[train]-mu);left=cross@v
    return [mu+(left/(np.maximum(0.,eig)+reg))@alpha for reg in (.01,.1,1.,10.)]


def main(step,source_step):
    configure_storage();fs=fsspec.filesystem('s3')
    def arrays(key):
        with np.load(io.BytesIO(fs.cat_file(key))) as z:return {k:z[k] for k in z.files}
    def load(st):
        groups=sorted([arrays(q) for q in fs.glob(ROOT+f'/document-sketch/native-packed-s{st}-*.npz')],key=lambda z:int(z['begin']))
        ix=np.concatenate([np.arange(int(z['begin']),int(z['begin'])+len(z['norms'])) for z in groups])
        feat=np.concatenate([z['forward'] for z in groups])
        response=np.concatenate([z['sketch'][:,0] for z in groups]).astype(float)
        return ix,feat,response
    ids,features,y=load(step);idx,previous,teacher=load(source_step);assert np.array_equal(ids,idx)
    tokens=arrays(ROOT+'/inputs/native-packed.npz')['tokens'][ids]
    cfg=dict(step=step,source=source_step,documents=len(ids),projection_dim=y.shape[1],seed=4373,
             regularization=[.01,.1,1.,10.],targets='true native trunk CountSketch',split='disjoint 64-sequence source blocks')
    os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
    run=wandb.init(entity='marin-community',project='gradient-variance-frontier',group='nonparametric-document-gradient-probe',id=os.urandom(5).hex(),config=cfg)
    unigrams=ngram_kernel(tokens,1);bigrams=ngram_kernel(tokens,2)
    lexical=(unigrams+bigrams)*.5
    result=dict(**cfg,wandb=run.url,folds=[])
    def mse(a,b):return float(np.mean(np.sum((a-b)**2,axis=1)))
    for fold in (0,1):
        test=np.where((ids//64)%2==fold)[0];train=np.where((ids//64)%2!=fold)[0]
        internal=train[(ids[train]//128)%4!=0];val=train[(ids[train]//128)%4==0]
        # Feature standardization uses the inner training partition only; current-state descriptors are available at eval.
        f=features.astype(float).copy();f[train]=previous[train]
        f=(f-f[internal].mean(0))/np.maximum(.0001,f[internal].std(0))
        f=np.clip(f,-6.,6.);f/=np.maximum(.001,np.linalg.norm(f,axis=1,keepdims=True))
        gram=f@f.T
        ygrid=y.copy();ygrid[train]=teacher[train]
        kernels={'hashed unigrams':unigrams,'hashed bigrams':bigrams,'lexical mixture':lexical,
                 'forward cosine':gram,'forward quadratic':((1+gram)/2)**2,'forward RBF':np.exp(2.*(gram-1)),
                 'lexical and forward':(lexical+gram)/2,'nonlinear lexical and forward':np.exp(1.5*(lexical+gram-2))}
        options=[]
        for name,K in kernels.items():
            for i,pred in enumerate(solve(K,internal,val,ygrid)):
                options.append((mse(pred,ygrid[val]),name,i))
        _,name,reg=min(options)
        pred=solve(kernels[name],train,test,ygrid)[reg]
        target=y[test];base=mse(target,ygrid[train].mean(0))
        err=mse(target,pred)
        result['folds'].append(dict(fold=fold,chosen=name,regularization=cfg['regularization'][reg],
                                    holdout_r2=1.-err/base,prediction_mse=err,intercept_mse=base,
                                    cal_docs=len(train),test_docs=len(test),validation_mse=min(options)[0]))
    run.log({'validation/mean_heldout_r2':float(np.mean([z['holdout_r2'] for z in result['folds']]))})
    publish(fs,ROOT+f'/kernel-probe/s{step}-from{source_step}.json',result);run.finish()


if __name__=='__main__':main(*map(int,sys.argv[1:]))

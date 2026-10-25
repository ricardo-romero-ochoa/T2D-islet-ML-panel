#!/usr/bin/env python3
"""Compute a DeLong confidence interval for an external ROC AUC."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    out=np.empty(N); out[J]=T; return out

def fast_delong(preds,label_1_count):
    m=label_1_count; n=preds.shape[1]-m; k=preds.shape[0]
    tx=np.empty((k,m)); ty=np.empty((k,n)); tz=np.empty((k,m+n))
    for r in range(k):
        tx[r]=compute_midrank(preds[r,:m]); ty[r]=compute_midrank(preds[r,m:]); tz[r]=compute_midrank(preds[r])
    aucs=tz[:,:m].sum(axis=1)/(m*n)-(m+1)/(2*n)
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    return aucs,np.atleast_2d(np.cov(v01)/m+np.cov(v10)/n)

def ci(y,score,alpha=.95):
    y=np.asarray(y).astype(int); score=np.asarray(score,float); order=np.argsort(-y); y=y[order]; score=score[order]
    m=int(y.sum()); aucs,cov=fast_delong(score[None,:],m); auc=float(aucs[0]); se=float(np.sqrt(cov[0,0])); z=norm.ppf(1-(1-alpha)/2)
    return auc,max(0,auc-z*se),min(1,auc+z*se),se

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scores',type=Path,required=True); ap.add_argument('--group-col',default='group'); ap.add_argument('--score-col',default='reduced_score'); args=ap.parse_args()
    d=pd.read_csv(args.scores); y=(d[args.group_col].astype(str)=='T2D').astype(int).values; s=d[args.score_col].values
    auc,lo,hi,se=ci(y,s); print(f'AUC={auc:.6f}; DeLong 95% CI={lo:.6f}-{hi:.6f}; SE={se:.6f}')

if __name__=='__main__': main()

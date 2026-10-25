#!/usr/bin/env python3
"""Fully nested stratified 5-fold validation for the GSE164416 ML workflow.

Every outer fold repeats DEG calling, LASSO/SVM-RFE/Random Forest feature
selection, consensus construction, scaling, and model fitting using training
data only. The held-out outer fold is used only for prediction.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
warnings.filterwarnings("ignore")
RANDOM_STATE=42; DEG_FDR=0.01; DEG_LOGFC=1.5; LASSO_FOLDS=10; SVM_FOLDS=5
SVM_STEP=0.1; RF_TREES=500; TOP_N=50; FINAL_N=10

def bh(p):
    p=np.asarray(p); n=len(p); order=np.argsort(p); out=np.empty(n,float)
    ranked=p[order]*n/np.arange(1,n+1); ranked=np.minimum.accumulate(ranked[::-1])[::-1]
    out[order]=np.minimum(ranked,1.0); return out

def deg_genes(Xtr,ytr):
    a=Xtr[ytr==1]; b=Xtr[ytr==0]; fc=a.mean(0)-b.mean(0)
    var1=a.var(0,ddof=1); var0=b.var(0,ddof=1); df=len(a)+len(b)-2
    sp2=((len(a)-1)*var1+(len(b)-1)*var0)/df
    prior_var=np.exp(np.median(np.log(sp2+1e-8))); prior_df=max(3,len(sp2)/10)
    shr=(prior_df*prior_var+df*sp2)/(df+prior_df); se=np.sqrt(shr*(1/len(a)+1/len(b)))
    t=fc/(se+1e-10); p=2*stats.t.sf(np.abs(t),df=df+prior_df); q=bh(p)
    return np.where((q<DEG_FDR)&(np.abs(fc)>=DEG_LOGFC))[0], fc, q

def select_panel(X,y,genes,seed):
    Xs=StandardScaler().fit_transform(X); min_class=min(np.bincount(y))
    cv=StratifiedKFold(n_splits=min(LASSO_FOLDS,max(3,min_class)),shuffle=True,random_state=seed)
    lasso=LogisticRegressionCV(Cs=np.logspace(-4,2,30),cv=cv,penalty='l1',solver='liblinear',max_iter=2000,random_state=seed,scoring='roc_auc',n_jobs=-1).fit(Xs,y)
    lsel=np.abs(lasso.coef_[0])>0; lscore=np.abs(lasso.coef_[0])
    cv2=StratifiedKFold(n_splits=min(SVM_FOLDS,max(3,min_class)),shuffle=True,random_state=seed)
    step=max(1,int(X.shape[1]*SVM_STEP))
    rfe=RFECV(LinearSVC(C=1.0,max_iter=5000,random_state=seed),step=step,cv=cv2,scoring='roc_auc',min_features_to_select=min(5,X.shape[1]),n_jobs=-1).fit(Xs,y)
    ssel=rfe.support_; srank=rfe.ranking_.astype(float)
    cv3=StratifiedKFold(n_splits=min(5,max(3,min_class)),shuffle=True,random_state=seed); imps=[]
    for k,(tr,_) in enumerate(cv3.split(Xs,y)):
        rf=RandomForestClassifier(n_estimators=RF_TREES,max_features='sqrt',n_jobs=-1,random_state=seed+k,class_weight='balanced').fit(Xs[tr],y[tr])
        imps.append(rf.feature_importances_)
    imp=np.mean(imps,0); order=np.argsort(-imp); rsel=np.zeros(len(genes),bool); rsel[order[:min(TOP_N,len(genes))]]=True
    votes=lsel.astype(int)+ssel.astype(int)+rsel.astype(int)
    comp=(lscore/(lscore.max()+1e-10) + 1-(srank-1)/(srank.max()+1e-10) + imp/(imp.max()+1e-10))/3
    cand=np.where(votes>=2)[0]
    if len(cand)<FINAL_N: cand=np.argsort(-comp)[:min(FINAL_N,len(comp))]
    else: cand=cand[np.argsort(-comp[cand])[:FINAL_N]]
    return cand

def build_ensemble(seed):
    return Pipeline([('scaler',StandardScaler()),('clf',VotingClassifier(estimators=[
        ('svm',SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=seed)),
        ('rf',RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=seed,n_jobs=-1)),
        ('lr',LogisticRegression(C=0.1,class_weight='balanced',max_iter=2000,random_state=seed)),
        ('gb',GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=seed))],voting='soft'))])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--expr',type=Path,default=Path('data/processed/GSE164416_expr_normalized.csv'))
    ap.add_argument('--labels',type=Path,default=Path('data/processed/GSE164416_labels.csv'))
    ap.add_argument('--outdir',type=Path,default=Path('results/nested_validation'))
    ap.add_argument('--outer-folds',type=int,default=5)
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    expr=pd.read_csv(args.expr,index_col=0); ldf=pd.read_csv(args.labels)
    labels=pd.Series(ldf.label.values,index=ldf.sample_id.values)
    common=[c for c in expr.columns if c in labels.index and labels[c] in [0,1]]
    expr=expr[common]; y=labels[common].astype(int).values; samples=np.array(common); G=np.array(expr.index); Xall=expr.values.T.astype(float)
    probs=np.zeros(len(y)); pred=np.zeros(len(y),int); fold_rows=[]; sel_rows=[]; fold_metrics=[]
    outer=StratifiedKFold(n_splits=args.outer_folds,shuffle=True,random_state=20261021)
    for fold,(tr,te) in enumerate(outer.split(Xall,y),1):
        Xtr,ytr,Xte,yte=Xall[tr],y[tr],Xall[te],y[te]
        didx,_,q=deg_genes(Xtr,ytr)
        if len(didx)<10: didx=np.argsort(q)[:max(10,min(184,len(q)))]
        pidx=select_panel(Xtr[:,didx],ytr,G[didx],RANDOM_STATE+fold); selected=G[didx][pidx]
        model=build_ensemble(RANDOM_STATE+fold).fit(Xtr[:,didx][:,pidx],ytr)
        pr=model.predict_proba(Xte[:,didx][:,pidx])[:,1]; probs[te]=pr; pred[te]=(pr>=0.5).astype(int)
        f_auc=roc_auc_score(yte,pr)
        fold_metrics.append({'fold':fold,'n_test':len(te),'n_deg_train':len(didx),'auc':f_auc,'panel_genes':';'.join(selected)})
        for j,ix in enumerate(te):
            fold_rows.append({'fold':fold,'test_sample':samples[ix],'true_label':int(y[ix]),'prob_t2d':float(pr[j]),'n_deg_train':len(didx),'n_panel':len(pidx),'panel_genes':';'.join(selected)})
        for g in selected: sel_rows.append({'fold':fold,'gene':g})
    auc=roc_auc_score(y,probs); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    metrics={'nested_5fold_auc':auc,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'f1':f1_score(y,pred),'mcc':matthews_corrcoef(y,pred),'accuracy':accuracy_score(y,pred),'n':len(y),'n_t2d':int(y.sum()),'n_nd':int((y==0).sum())}
    rng=np.random.default_rng(20261021); aucs=[]; pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    for _ in range(5000):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]; aucs.append(roc_auc_score(y[idx],probs[idx]))
    aucs=np.array(aucs); metrics.update({'bootstrap_auc_median':float(np.median(aucs)),'auc_ci_2_5':float(np.quantile(aucs,.025)),'auc_ci_97_5':float(np.quantile(aucs,.975)),'fold_auc_mean':float(np.mean([x['auc'] for x in fold_metrics])),'fold_auc_sd':float(np.std([x['auc'] for x in fold_metrics],ddof=1)),'fold_auc_min':float(np.min([x['auc'] for x in fold_metrics])),'fold_auc_max':float(np.max([x['auc'] for x in fold_metrics]))})
    pd.DataFrame(fold_rows).to_csv(args.outdir/'nested_5fold_predictions_and_panels.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(args.outdir/'nested_5fold_fold_metrics.csv',index=False)
    sel=pd.DataFrame(sel_rows); stab=sel.groupby('gene').agg(n_folds=('fold','nunique')).reset_index(); stab['selection_frequency']=stab.n_folds/args.outer_folds
    stab.sort_values(['selection_frequency','gene'],ascending=[False,True]).to_csv(args.outdir/'nested_feature_selection_stability.csv',index=False)
    pd.DataFrame({'auc':aucs}).to_csv(args.outdir/'nested_auc_bootstrap_distribution.csv',index=False)
    pd.DataFrame([metrics]).to_csv(args.outdir/'nested_5fold_metrics.csv',index=False)
    (args.outdir/'nested_5fold_summary.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Build manuscript Figure 7 from full-cohort descriptive inputs and nested-CV outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--figure-inputs',type=Path,default=Path('results/figure_inputs'))
    ap.add_argument('--nested-dir',type=Path,default=Path('results/nested_validation'))
    ap.add_argument('--outdir',type=Path,default=Path('figures'))
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    single=pd.read_csv(args.figure_inputs/'fig7_single_gene_auc.csv').sort_values('auc')
    venn=json.loads((args.figure_inputs/'fig7_venn_counts.json').read_text())
    pred=pd.read_csv(args.nested_dir/'nested_5fold_predictions_and_panels.csv')
    stab=pd.read_csv(args.nested_dir/'nested_feature_selection_stability.csv')
    final=pd.read_csv(args.figure_inputs/'fig7_final_panel.csv')
    label_col='display_label' if 'display_label' in final.columns else 'gene'
    st=[]
    for _,r in final.iterrows():
        q=stab[stab.gene==r.gene]; st.append((r[label_col],float(q.selection_frequency.iloc[0]) if len(q) else 0.0))
    st=pd.DataFrame(st,columns=['gene','freq']).sort_values('freq')
    fig=plt.figure(figsize=(12,10)); gs=fig.add_gridspec(3,2,height_ratios=[0.9,1.1,1.1],hspace=0.38,wspace=0.28)
    ax=fig.add_subplot(gs[0,:]); ax.axis('off')
    boxes=[(0.03,0.60,0.20,0.22,'Outer training set\n(~80% of donors)'),(0.28,0.60,0.20,0.22,'DEG calling\ntraining samples only'),(0.53,0.60,0.20,0.22,'LASSO + SVM-RFE + RF\ntraining samples only'),(0.78,0.60,0.19,0.22,'10-feature panel +\nensemble fit')]
    for x,y,w,h,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.02',facecolor='#f2f2f2',edgecolor='0.35')); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9)
    for i in range(len(boxes)-1):
        x=boxes[i][0]+boxes[i][2]; y=boxes[i][1]+boxes[i][3]/2; x2=boxes[i+1][0]; ax.annotate('',xy=(x2-0.01,y),xytext=(x+0.01,y),arrowprops=dict(arrowstyle='->',lw=1.2,color='0.35'))
    ax.add_patch(FancyBboxPatch((0.36,0.15),0.28,0.22,boxstyle='round,pad=0.01,rounding_size=0.02',facecolor='#e8f1f8',edgecolor='0.35')); ax.text(0.50,0.26,'Held-out outer fold\nprediction only',ha='center',va='center',fontsize=9)
    ax.annotate('',xy=(0.50,0.39),xytext=(0.875,0.59),arrowprops=dict(arrowstyle='->',lw=1.2,color='0.35')); ax.text(0.02,0.97,'A',fontsize=15,fontweight='bold',va='top'); ax.text(0.50,0.02,'Nested stratified 5-fold cross-validation',ha='center',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[1,0]); ax.set_aspect('equal'); ax.axis('off')
    centers=[(0.39,0.46),(0.61,0.46),(0.50,0.62)]; cols=['#b9d9f0','#bfe5ca','#f7e8a8']
    for (cx,cy),c in zip(centers,cols): ax.add_patch(Circle((cx,cy),0.23,facecolor=c,edgecolor='0.6',alpha=.7))
    positions={'lasso_only':(.24,.42),'svm_only':(.76,.42),'rf_only':(.50,.79),'lasso_svm':(.50,.39),'lasso_rf':(.40,.59),'svm_rf':(.60,.59),'all_three':(.50,.53)}
    for k,(x,y) in positions.items(): ax.text(x,y,str(venn[k]),ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(.28,.18,'LASSO',ha='center',fontsize=9,fontweight='bold'); ax.text(.72,.18,'SVM-RFE',ha='center',fontsize=9,fontweight='bold'); ax.text(.50,.91,'Random Forest',ha='center',fontsize=9,fontweight='bold'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.text(0.01,0.98,'B',transform=ax.transAxes,fontsize=15,fontweight='bold',va='top'); ax.set_title('Full-cohort feature-selection overlap',fontsize=10)
    ax=fig.add_subplot(gs[1,1]); ax.barh(single.display_label,single.auc,color='0.55'); ax.set_xlim(.80,.96)
    for yv,v in enumerate(single.auc): ax.text(v+0.002,yv,f'{v:.3f}',va='center',fontsize=8)
    ax.set_xlabel('Single-feature AUC'); ax.set_title('Full-cohort candidate feature AUCs',fontsize=10); ax.spines[['top','right']].set_visible(False); ax.text(-0.08,1.03,'C',transform=ax.transAxes,fontsize=15,fontweight='bold')
    ax=fig.add_subplot(gs[2,0]); y=pred.true_label.values; p=pred.prob_t2d.values; fpr,tpr,_=roc_curve(y,p); auc=roc_auc_score(y,p)
    ax.plot(fpr,tpr,lw=2.2,label=f'Nested 5-fold AUC = {auc:.3f}'); ax.plot([0,1],[0,1],'--',color='0.55',lw=1); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(loc='lower right',fontsize=8); ax.spines[['top','right']].set_visible(False); ax.set_title('Out-of-fold ROC (selection fully nested)',fontsize=10); ax.text(-0.08,1.03,'D',transform=ax.transAxes,fontsize=15,fontweight='bold')
    summary=json.loads((args.nested_dir/'nested_5fold_summary.json').read_text()); ax.text(0.04,0.82,f"Bootstrap 95% CI: {summary['auc_ci_2_5']:.3f}–{summary['auc_ci_97_5']:.3f}",transform=ax.transAxes,fontsize=8)
    ax=fig.add_subplot(gs[2,1]); ax.barh(st.gene,st.freq,color='0.55'); ax.set_xlim(0,1.05); ax.set_xlabel('Outer-fold selection frequency'); ax.set_title('Feature-selection stability',fontsize=10)
    for yv,v in enumerate(st.freq): ax.text(v+0.02,yv,f'{int(round(v*5))}/5',va='center',fontsize=8)
    ax.spines[['top','right']].set_visible(False); ax.text(-0.08,1.03,'E',transform=ax.transAxes,fontsize=15,fontweight='bold')
    fig.subplots_adjust(left=.09,right=.97,top=.98,bottom=.07)
    fig.savefig(args.outdir/'Figure07_nested.png',dpi=600,bbox_inches='tight'); fig.savefig(args.outdir/'Figure07_nested.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(args.outdir/'Figure07_nested.pdf',bbox_inches='tight')

if __name__=='__main__': main()

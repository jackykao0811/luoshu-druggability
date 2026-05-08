"""
Luoshu Druggability 完整審計腳本 v2
修正 Leave-Family-Out 邏輯：
  改為「留一靶點（LOPO）+ 家族分組報告」
  每次 test 只有一個蛋白（一定有正/負標籤）
  然後按 family 彙整 AUC 分析

作者：Kao, Yao-Kai (NYCU)
日期：2026-05-08
"""

import os, requests, math, time, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import ConvexHull, Delaunay
from sklearn.metrics import (roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import LeaveOneOut

PHI = 1.6180339887
LUO_NUM = {(1,1,1):9,(1,1,0):7,(1,0,1):3,(1,0,0):1,
           (0,1,1):2,(0,1,0):6,(0,0,1):8,(0,0,0):4}
HYDROPHOBIC = {'A','C','F','G','H','I','L','M','P','V','W','Y'}
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
       'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
       'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
       'TYR':'Y','VAL':'V','MSE':'M','HSD':'H','HSE':'H','HSP':'H'}

TARGETS = [
    ("CDK2","1HCL",1,"Kinase"),("EGFR","2GS7",1,"Kinase"),
    ("BRAF","1UWH",1,"Kinase"),("Aurora-A","1MQ4",1,"Kinase"),
    ("BCR-ABL","2HYY",1,"Kinase"),("VEGFR2","2OH4",1,"Kinase"),
    ("SRC","2SRC",1,"Kinase"),("p38-MAPK","1A9U",1,"Kinase"),
    ("CHK1","1NVQ",1,"Kinase"),("GSK3B","1J1B",1,"Kinase"),
    ("ROCK1","2ETK",1,"Kinase"),("PLK1","2OJX",1,"Kinase"),
    ("MET","1R0P",1,"Kinase"),("IGF1R","2OJ9",1,"Kinase"),
    ("FGFR1","2FGI",1,"Kinase"),("LCK","2OF2",1,"Kinase"),
    ("JAK2","2XA4",1,"Kinase"),
    ("HIV-PR","3HVP",1,"Protease"),("Trypsin","1TGS",1,"Protease"),
    ("MMP3","1B8Y",1,"Protease"),("MMP2","1CK7",1,"Protease"),
    ("FXa","1KSN",1,"Protease"),("Renin","2V16",1,"Protease"),
    ("BACE1","1W51",1,"Protease"),("CathB","1CSB",1,"Protease"),
    ("FURIN","5JXH",1,"Protease"),
    ("AChE","1QTI",1,"Enzyme"),("CA-II","1CAM",1,"Enzyme"),
    ("DHFR","3NXO",1,"Enzyme"),("PARP1","3GJW",1,"Enzyme"),
    ("PDE5","1UDT",1,"Enzyme"),("COX-2","3LN1",1,"Enzyme"),
    ("HMGR","1HWK",1,"Enzyme"),("TS","1HVY",1,"Enzyme"),
    ("IMPDH","1NF3",1,"Enzyme"),("DHODH","1D3G",1,"Enzyme"),
    ("InhA","1P44",1,"Enzyme"),("GlcNAc","1NLG",1,"Enzyme"),
    ("DPP4","1N1M",1,"Enzyme"),("ALR2","2FZD",1,"Enzyme"),
    ("HIV-RT","1RTH",1,"Enzyme"),("HIV-IN","1BIS",1,"Enzyme"),
    ("NA","2HT7",1,"Enzyme"),("CP3A4","1TQN",1,"Enzyme"),
    ("PYGB","1GPA",1,"Enzyme"),("INHA","1ZID",1,"Enzyme"),
    ("PPARg","1FM9",1,"NR"),("PPARd","2ZNP",1,"NR"),
    ("AR","2AM9",1,"NR"),("ERa","1SJ0",1,"NR"),
    ("ERb","1X7J",1,"NR"),("GR","1M2Z",1,"NR"),
    ("PR","2OVH",1,"NR"),("THRa","3GWS",1,"NR"),
    ("RXRa","1MV9",1,"NR"),("FXR","3DCT",1,"NR"),
    ("LXRb","1PQ9",1,"NR"),
    ("HSP90","1YET",1,"Chaperone"),("PTPB","2CME",1,"Phosphatase"),
    ("KIF11","2GRF",1,"Kinesin"),("KDM4A","2OQ6",1,"Enzyme"),
    ("BRD4","3MXF",1,"Enzyme"),
    ("ADRB1","2VT4",1,"GPCR"),("ADRB2","2RH1",1,"GPCR"),
    ("CXCR4","3ODU",1,"GPCR"),("DRD3","3PBL",1,"GPCR"),
    ("HRH1","3RZE",1,"GPCR"),
    ("AKT1","3CQW",1,"Kinase"),("AKT2","3D0E",1,"Kinase"),
    ("CDK1","4YC3",1,"Kinase"),("PIK3CA","3ZIM",1,"Kinase"),
    ("MTOR","4JSN",1,"Kinase"),("BTK","3K54",1,"Kinase"),
    ("PDK1","3HRF",1,"Kinase"),
    ("HDAC2","3MAX",1,"Enzyme"),("SIRT1","4ZZJ",1,"Enzyme"),
    ("PRMT5","4GQB",1,"Enzyme"),("IDH1","4UMX",1,"Enzyme"),
    ("DNMT3A","2QRV",1,"Enzyme"),("EZH2","5LS6",1,"Enzyme"),
    ("SETD2","5JLE",1,"Enzyme"),("DOT1L","3UWP",1,"Enzyme"),
    ("WDR5","3EG7",1,"Enzyme"),("MCL1","4HW2",1,"Enzyme"),
    ("PLpro","7LLF",1,"Protease"),("MPRO","6LU7",1,"Protease"),
    ("ACE2","6M0J",1,"Enzyme"),
    ("cMyc-Max","1NKP",0,"TF-PPI"),("p53-MDM2","1YCR",0,"PPI"),
    ("BCL2","1G5M",0,"PPI"),("XIAP","1TFT",0,"PPI"),
    ("Survivin","1XOX",0,"PPI"),("b-cat","1JPW",0,"PPI"),
    ("PCNA","1AXC",0,"PPI-clamp"),("Ubiq","1UBQ",0,"No-pocket"),
    ("KRAS","4OBE",0,"GTPase"),("NRAS","3CON",0,"GTPase"),
    ("HRAS","4EFL",0,"GTPase"),("Cdc42","2QRZ",0,"GTPase"),
    ("Rac1","2C2H",0,"GTPase"),
    ("MYC","1NKP",0,"TF"),("HIF-1a","1LM8",0,"TF"),
    ("Stat3","1BG1",0,"TF"),("NF-kB","1IKU",0,"TF"),
    ("p65","1SVC",0,"TF"),
    ("CaM","1CLL",0,"Flexible"),
    ("Tuba","1TUB",0,"Structural"),("Actin","1ATN",0,"Structural"),
    ("Grb2","1TZE",0,"Adaptor"),("14-3-3","2BQ0",0,"Adaptor"),
    ("PDZ","1IU0",0,"PDZ"),
    ("RRAS","2FN4",0,"GTPase"),("MRAS","1X1S",0,"GTPase"),
    ("RAL","2BOV",0,"GTPase"),("RAP1","1C1Y",0,"GTPase"),
    ("RHO","1DPF",0,"GTPase"),
    ("FOXO1","2O9J",0,"TF"),("NOTCH1","3ETO",0,"TF-PPI"),
    ("WNT","1IJY",0,"PPI"),("YAP1","3KYS",0,"TF-PPI"),
    ("TEAD1","3KYS",0,"TF"),
    ("FLNA","2K3T",0,"Structural"),("VIM","3SSU",0,"Structural"),
    ("LMNA","1IFR",0,"Structural"),
    ("MAPT","2MZ7",0,"IDP"),
]

# 合併家族分組（用於結果彙整）
COARSE_FAMILY = {
    "Kinase":"Kinase","GPCR":"GPCR","NR":"NR",
    "Protease":"Protease","Enzyme":"Enzyme",
    "Chaperone":"Enzyme","Phosphatase":"Enzyme","Kinesin":"Enzyme",
    "GTPase":"GTPase",
    "PPI":"PPI/TF","TF":"PPI/TF","TF-PPI":"PPI/TF","PPI-clamp":"PPI/TF",
    "No-pocket":"Undrug-Other","Flexible":"Undrug-Other",
    "Structural":"Undrug-Other","Adaptor":"Undrug-Other",
    "PDZ":"Undrug-Other","IDP":"Undrug-Other",
}

def download_pdb(pid, cache='pdb_cache'):
    os.makedirs(cache,exist_ok=True)
    fp=os.path.join(cache,f'{pid.lower()}.pdb')
    if os.path.exists(fp): return fp
    try:
        r=requests.get(f'https://files.rcsb.org/download/{pid}.pdb',timeout=30)
        if r.status_code==200 and 'ATOM' in r.text:
            open(fp,'w').write(r.text); return fp
    except: pass
    return None

def parse_pdb(fp):
    atoms={}; raa={}; fc=None
    with open(fp) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            an=line[12:16].strip(); rn=line[17:20].strip(); ch=line[21]
            try: rnum=int(line[22:26])
            except: continue
            aa=AA3.get(rn)
            if not aa: continue
            if fc is None: fc=ch
            if ch!=fc: continue
            key=(ch,rnum)
            if key not in raa: raa[key]=aa
            if an not in ('N','CA','C'): continue
            try:
                xyz=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                if key not in atoms: atoms[key]={}
                atoms[key][an]=xyz
            except: pass
    def dih(p1,p2,p3,p4):
        b1=p2-p1;b2=p3-p2;b3=p4-p3
        n1=np.cross(b1,b2);n2=np.cross(b2,b3)
        nm=np.linalg.norm(b2)
        if nm<1e-8: return None
        return np.degrees(np.arctan2(np.dot(np.cross(n1,b2/nm),n2),np.dot(n1,n2)))
    keys=sorted(atoms.keys()); out=[]
    for i,key in enumerate(keys):
        cur=atoms[key]; aa=raa.get(key,'X')
        if not all(a in cur for a in ('N','CA','C')): continue
        phi=psi=None
        if i>0 and keys[i-1][0]==key[0] and 'C' in atoms.get(keys[i-1],{}):
            try: phi=dih(atoms[keys[i-1]]['C'],cur['N'],cur['CA'],cur['C'])
            except: pass
        if i<len(keys)-1 and keys[i+1][0]==key[0] and 'N' in atoms.get(keys[i+1],{}):
            try: psi=dih(cur['N'],cur['CA'],cur['C'],atoms[keys[i+1]]['N'])
            except: pass
        if phi and psi:
            out.append((key[1],aa,phi,psi,cur.get('CA')))
    return out

def intrinsic_dim(pts):
    if len(pts)<4: return 2.0
    c=pts-pts.mean(0)
    try:
        _,s,_=np.linalg.svd(c,full_matrices=False)
        lam=s**2; lam=lam[lam>1e-10]
        return float(min((lam.sum())**2/(lam**2).sum(),3.0))
    except: return 2.0

def pocket_torsion(pts):
    if len(pts)<4: return 90.0
    order=np.argsort(np.linalg.norm(pts-pts.mean(0),axis=1))
    p=pts[order]; t=[]
    for i in range(len(p)-3):
        b1=p[i+1]-p[i];b2=p[i+2]-p[i+1];b3=p[i+3]-p[i+2]
        n1=np.cross(b1,b2);n2=np.cross(b2,b3)
        nm=np.linalg.norm(b2)
        if nm<1e-8: continue
        try:
            t.append(abs(np.degrees(np.arctan2(
                np.dot(np.cross(n1,b2/nm),n2),np.dot(n1,n2)))))
        except: pass
    return float(np.mean(t)) if t else 90.0

def compute_features(res):
    valid=[(r[2],r[3],r[4]) for r in res if r[4] is not None]
    if len(valid)<15: return None
    guas=[((1 if abs(p)<=90 else 0),(1 if p>0 else 0),(1 if abs(q)<=60 else 0))
          for p,q,_ in valid]
    ca=np.array([c for _,_,c in valid])
    aa=[r[1] for r in res if r[4] is not None]
    n=len(guas); sc=[]
    for w in [1,3,5,7]:
        s=np.zeros(n)
        for i in range(n):
            tot=cnt=0.0
            for j in range(max(0,i-w),min(n,i+w+1)):
                if j==i: continue
                d=abs(i-j); wt=1/d
                tot+=wt*abs(LUO_NUM[guas[i]]-LUO_NUM[guas[j]]); cnt+=wt
            s[i]=tot/cnt if cnt else 0
        sc.append(s)
    pw=[1/PHI**k for k in range(4)]
    lte=sum(w*s for w,s in zip(pw,sc))/sum(pw)
    lte_n=(lte-lte.min())/(lte.max()-lte.min()+1e-8)
    hidx=np.where(lte_n>=np.percentile(lte_n,70))[0]
    if len(hidx)<4: return None
    try: hull=ConvexHull(ca[hidx]); dl=Delaunay(ca[hidx])
    except: return None
    chc=np.zeros(n)
    for i in range(n):
        if dl.find_simplex(ca[i])>=0:
            chc[i]=min(abs(np.dot(eq[:3],ca[i])+eq[3]) for eq in hull.equations)
    chc_max=float(chc.max())
    vol=float(hull.volume); area=float(hull.area)
    centroid=ca.mean(0)
    prot_r=float(np.sqrt(((ca-centroid)**2).sum(1).mean()))
    sph=(36*math.pi*vol**2)**(1/3)/(area+1e-8)
    pmask=chc>np.percentile(chc,70)
    hf=(sum(1 for i in range(n) if pmask[i] and aa[i] in HYDROPHOBIC)
        /max(pmask.sum(),1))
    pocket_ca=ca[chc>1.0]
    idim=intrinsic_dim(pocket_ca) if len(pocket_ca)>=4 else 2.0
    tort=pocket_torsion(pocket_ca) if len(pocket_ca)>=4 else 90.0
    return dict(
        sphericity=sph, chc_max=chc_max,
        hydro_frac=hf, area_vol=area/(vol+1e-8),
        sph_chc=sph*chc_max, sph_hf=sph*hf,
        sph_chc_hf=sph*chc_max*hf,
        idim=idim, tort_norm=1-tort/180.0,
        idim_chc=idim/3.0*chc_max, tort_sph=(1-tort/180.0)*sph,
        fp_vol=vol, fp_area=area, fp_hydro=hf,
        fp_score=hf*vol**(1/3)/(area/(vol+1e-8)+1e-8),
        n_res=n
    )

def nrm(x):
    r=x.max()-x.min(); return (x-x.min())/(r+1e-8)

def auc_s(y,sc):
    try:
        a=roc_auc_score(y,sc); return a if a>=0.5 else 1-a
    except: return 0.5

def main():
    print("="*70)
    print("Luoshu Druggability 完整審計 v2（LOPO + Family Breakdown）")
    print("="*70)
    t0=time.time()

    seen={}; dedup=[]
    for row in TARGETS:
        k=row[1].upper()
        if k not in seen: seen[k]=True; dedup.append(row)

    records=[]; failed=[]
    for name,pid,label,typ in dedup:
        pf=download_pdb(pid)
        if not pf: failed.append(name); continue
        res=parse_pdb(pf)
        if len(res)<15: failed.append(f"{name}({len(res)})"); continue
        feats=compute_features(res)
        if feats is None: failed.append(name); continue
        feats.update({'name':name,'label':label,'type':typ,'pdb':pid,
                      'family':COARSE_FAMILY.get(typ,'Other')})
        records.append(feats)

    print(f"有效：{len(records)}  失敗：{len(failed)}")
    y=np.array([r['label'] for r in records])
    n_d=int(y.sum()); n_u=int((1-y).sum())
    print(f"Drug={n_d}  Undrug={n_u}\n")

    FEATS=['sphericity','chc_max','sph_chc','sph_hf','sph_chc_hf',
           'idim_chc','tort_sph']
    Xn={f:nrm(np.array([r[f] for r in records])) for f in FEATS}
    Xn['fp']=nrm(0.5*np.array([r['fp_hydro'] for r in records])+
                  0.3*nrm(np.array([r['fp_vol'] for r in records]))+
                  0.2*nrm(np.array([r['fp_score'] for r in records])))

    score_best=(0.60*Xn['sph_chc_hf']+
                0.25*Xn['sphericity']+
                0.15*Xn['sph_hf'])
    score_fp  =Xn['fp']
    score_v3  =0.7*Xn['sphericity']+0.3*Xn['chc_max']

    # ══════════════════════════════════════════════════════════
    # 1. HEAD-TO-HEAD
    # ══════════════════════════════════════════════════════════
    print("═"*60)
    print("1. HEAD-TO-HEAD COMPARISON")
    print("═"*60)
    for nm,sc in [('SiteMap (literature)',None),
                  ('fpocket-style naive', score_fp),
                  ('Luoshu v3 Linear',    score_v3),
                  ('Luoshu Best',         score_best)]:
        if sc is None:
            print(f"  {nm:35} 0.740   --  (文獻)")
        else:
            a=auc_s(y,sc); pa=average_precision_score(y,sc)
            print(f"  {nm:35} ROC={a:.4f}  PR={pa:.4f}  vs SiteMap={a-0.74:+.4f}")

    # ══════════════════════════════════════════════════════════
    # 2. LEAVE-ONE-PROTEIN-OUT → family 彙整
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("2. LEAVE-ONE-PROTEIN-OUT CV → Family Breakdown")
    print("═"*60)
    print("  （每次留一個蛋白為 test，其餘 n-1 個為 train）")
    print("  （train 只做特徵正規化，無參數訓練）\n")

    # LOPO：每次拿掉一個蛋白，重新 normalize，預測留出那個蛋白的排名
    lopo_scores_l=[]; lopo_scores_f=[]
    for i in range(len(records)):
        train_idx=[j for j in range(len(records)) if j!=i]
        # 只重新 normalize（無模型參數）
        def nrm_from_train(vals, ti):
            tv=vals[ti]; mn=tv.min(); mx=tv.max()
            r=mx-mn
            return (vals-mn)/(r+1e-8)
        Xt={}
        for f in FEATS:
            v=np.array([records[j][f] for j in range(len(records))])
            Xt[f]=nrm_from_train(v, train_idx)
        Xt['fp']=nrm_from_train(
            0.5*np.array([r['fp_hydro'] for r in records])+
            0.2*np.array([r['fp_score'] for r in records])+
            0.3*np.array([r['fp_vol'] for r in records]), train_idx)
        s_l=(0.60*Xt['sph_chc_hf'][i]+
             0.25*Xt['sphericity'][i]+
             0.15*Xt['sph_hf'][i])
        s_f=Xt['fp'][i]
        lopo_scores_l.append(float(s_l))
        lopo_scores_f.append(float(s_f))

    lopo_l=np.array(lopo_scores_l)
    lopo_f=np.array(lopo_scores_f)
    auc_lopo_l=auc_s(y,lopo_l)
    auc_lopo_f=auc_s(y,lopo_f)
    pr_lopo_l =average_precision_score(y,lopo_l)
    pr_lopo_f =average_precision_score(y,lopo_f)
    print(f"  LOPO 整體：Luoshu ROC={auc_lopo_l:.4f} PR={pr_lopo_l:.4f}")
    print(f"  LOPO 整體：fp-style ROC={auc_lopo_f:.4f} PR={pr_lopo_f:.4f}")

    # 按 family 彙整：每個蛋白的排名百分位（越高=模型越確信為 Drug）
    print(f"\n  Family 分析（每個蛋白的 LOPO 預測排名）：")
    print(f"  {'Family':15} {'n':3} {'D':3} {'U':3} {'Luoshu rank':12} {'判斷'}")
    fam_results=defaultdict(list)
    for i,r in enumerate(records):
        fam_results[r['family']].append({
            'name':r['name'],'label':r['label'],
            'score_l':lopo_l[i],'score_f':lopo_f[i]
        })

    family_order=['Kinase','GPCR','NR','Protease','Enzyme',
                  'GTPase','PPI/TF','Undrug-Other']
    family_summary={}
    for fam in family_order:
        items=fam_results.get(fam,[])
        if not items: continue
        labs=np.array([x['label'] for x in items])
        scs_l=np.array([x['score_l'] for x in items])
        scs_f=np.array([x['score_f'] for x in items])
        nd=int(labs.sum()); nu=int((1-labs).sum())
        # 用全局排名計算：drug 的平均排名 vs undrug
        # rank_l：這個 family 裡每個蛋白在全局的排名百分位
        global_ranks_l=np.array([np.mean(lopo_l<=s) for s in scs_l])
        drug_rank=global_ranks_l[labs==1].mean() if nd>0 else float('nan')
        undrug_rank=global_ranks_l[labs==0].mean() if nu>0 else float('nan')
        sep=drug_rank-undrug_rank
        if sep>0.3: judge="★★ 良好分離"
        elif sep>0.1: judge="★  輕微分離"
        elif sep>-0.1: judge="◎  難分離"
        else: judge="▼  反向"
        family_summary[fam]={'nd':nd,'nu':nu,'drug_rank':drug_rank,
                             'undrug_rank':undrug_rank,'sep':sep,'judge':judge}
        print(f"  {fam:15} {nd+nu:3} {nd:3} {nu:3} "
              f"D={drug_rank:.2f} U={undrug_rank:.2f} sep={sep:+.2f}  {judge}")

    # ══════════════════════════════════════════════════════════
    # 3. CONFUSION MATRIX（使用已知的 full-sample score）
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("3. CONFUSION MATRIX")
    print("═"*60)
    threshold=np.percentile(score_best,100*(1-n_d/len(y)))
    y_pred=(score_best>=threshold).astype(int)
    cm=confusion_matrix(y,y_pred)
    tn,fp_n,fn_n,tp=cm.ravel()
    prec=precision_score(y,y_pred,zero_division=0)
    rec =recall_score(y,y_pred,zero_division=0)
    f1  =f1_score(y,y_pred,zero_division=0)
    spec=tn/(tn+fp_n) if (tn+fp_n)>0 else 0
    print(f"  TP={tp}  FN={fn_n}  FP={fp_n}  TN={tn}")
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  Specificity={spec:.4f}")

    fps=[(records[i]['name'],records[i]['family'],float(score_best[i]))
         for i in range(len(records)) if y[i]==0 and y_pred[i]==1]
    fns=[(records[i]['name'],records[i]['family'],float(score_best[i]))
         for i in range(len(records)) if y[i]==1 and y_pred[i]==0]
    fps.sort(key=lambda x:-x[2])
    fns.sort(key=lambda x:x[2])
    print(f"\n  FP（undruggable 誤判，{len(fps)} 個）：{[x[0] for x in fps]}")
    print(f"  FN（druggable 誤判，{len(fns)} 個）：{[x[0] for x in fns]}")

    # ══════════════════════════════════════════════════════════
    # 4. PR-AUC 完整表
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("4. 完整評估指標")
    print("═"*60)
    print(f"  {'方法':35} {'ROC-AUC':8} {'PR-AUC':8} {'F1':6}")
    print(f"  {'SiteMap (literature)':35} {'0.740':8} {'~0.720':8} {'--':6}")
    for nm,sc in [('fpocket-style naive',score_fp),
                  ('Luoshu v3 Linear',score_v3),
                  ('Luoshu Best (full-sample)',score_best),
                  ('Luoshu LOPO',lopo_l)]:
        a=auc_s(y,sc)
        pa=average_precision_score(y,sc)
        thr=np.percentile(sc,100*(1-n_d/len(y)))
        yp=(sc>=thr).astype(int)
        f=f1_score(y,yp,zero_division=0)
        print(f"  {nm:35} {a:.4f}   {pa:.4f}   {f:.4f}")

    # ══════════════════════════════════════════════════════════
    # 5. 輸出 CSV
    # ══════════════════════════════════════════════════════════
    csv_fields=['sphericity','chc_max','sph_chc_hf','idim_chc',
                'tort_sph','hydro_frac','fp_vol','fp_hydro']
    with open('luoshu_features.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['name','pdb','label','family','luoshu_score',
                    'lopo_score','fp_score']+csv_fields)
        for r,ls,lp,fps2 in zip(records,score_best,lopo_l,score_fp):
            row=[r['name'],r['pdb'],r['label'],r['family'],
                 f'{ls:.6f}',f'{lp:.6f}',f'{fps2:.6f}']
            row+=[f'{r[fn]:.6f}' for fn in csv_fields]
            w.writerow(row)
    print(f"\n  luoshu_features.csv 儲存完成")

    # ══════════════════════════════════════════════════════════
    # 視覺化
    # ══════════════════════════════════════════════════════════
    fig=plt.figure(figsize=(22,14))

    # 1. ROC
    ax1=fig.add_subplot(2,4,1)
    for nm,sc,c,lw in [
        ('fpocket-style',score_fp,'#FF9800',1.8),
        ('Luoshu v3',score_v3,'#4CAF50',2.0),
        ('Luoshu Best',score_best,'#E91E63',2.5),
        ('Luoshu LOPO',lopo_l,'#9C27B0',2.0),
    ]:
        a=roc_auc_score(y,sc)
        if a<0.5: sc=-sc; a=1-a
        fpr,tpr,_=roc_curve(y,sc)
        ax1.plot(fpr,tpr,color=c,lw=lw,label=f'{nm} {a:.4f}')
    ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
    ax1.set_title('ROC Curves')
    ax1.legend(fontsize=7)

    # 2. PR
    ax2=fig.add_subplot(2,4,2)
    for nm,sc,c,lw in [
        ('fpocket-style',score_fp,'#FF9800',1.8),
        ('Luoshu v3',score_v3,'#4CAF50',2.0),
        ('Luoshu Best',score_best,'#E91E63',2.5),
        ('Luoshu LOPO',lopo_l,'#9C27B0',2.0),
    ]:
        pre,rec2,_=precision_recall_curve(y,sc)
        ap=average_precision_score(y,sc)
        ax2.plot(rec2,pre,color=c,lw=lw,label=f'{nm} {ap:.4f}')
    ax2.axhline(n_d/len(y),color='gray',ls='--',lw=1)
    ax2.set_title('PR Curves')
    ax2.legend(fontsize=7)

    # 3. Family breakdown
    ax3=fig.add_subplot(2,4,3)
    fams=[f for f in family_order if f in family_summary]
    seps=[family_summary[f]['sep'] for f in fams]
    colors3=['#4CAF50' if s>0.3 else '#FF9800' if s>0.1
             else '#9E9E9E' if s>-0.1 else '#F44336' for s in seps]
    ax3.barh(range(len(fams)),seps,color=colors3,alpha=0.85)
    ax3.axvline(0.3,color='green',ls='--',lw=1.5,label='Good sep(0.3)')
    ax3.axvline(0,color='black',ls='-',lw=0.5)
    ax3.set_yticks(range(len(fams)))
    ax3.set_yticklabels(fams,fontsize=9)
    ax3.set_xlabel('Drug rank - Undrug rank (LOPO)')
    ax3.set_title('Family Separation\n(LOPO global rank difference)')
    ax3.legend(fontsize=7)

    # 4. Confusion matrix
    ax4=fig.add_subplot(2,4,4)
    ax4.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax4.text(j,i,str(cm[i,j]),ha='center',va='center',
                    fontsize=18,fontweight='bold',
                    color='white' if cm[i,j]>cm.max()*0.5 else 'black')
    ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
    ax4.set_xticklabels(['Pred+','Pred-'])
    ax4.set_yticklabels(['True+','True-'])
    ax4.set_title(f'Confusion Matrix\nF1={f1:.3f} Prec={prec:.3f}\n'
                  f'Rec={rec:.3f} Spec={spec:.3f}')

    # 5. FP 分析
    ax5=fig.add_subplot(2,4,5)
    if fps:
        fn5=[x[0] for x in fps]; sc5=[x[2] for x in fps]
        c5=['#F44336']*len(fn5)
        ax5.barh(range(len(fn5)),sc5,color=c5,alpha=0.85)
        ax5.axvline(threshold,color='black',ls='--',lw=1.5,label=f'thr={threshold:.3f}')
        ax5.set_yticks(range(len(fn5))); ax5.set_yticklabels(fn5,fontsize=8)
        ax5.set_title(f'False Positives ({len(fps)})\nUndruggable misclassified')
        ax5.legend(fontsize=7)

    # 6. FN 分析
    ax6=fig.add_subplot(2,4,6)
    if fns:
        fn6=[x[0] for x in fns]; sc6=[x[2] for x in fns]
        ax6.barh(range(len(fn6)),sc6,color='#FF9800',alpha=0.85)
        ax6.axvline(threshold,color='black',ls='--',lw=1.5)
        ax6.set_yticks(range(len(fn6))); ax6.set_yticklabels(fn6,fontsize=8)
        ax6.set_title(f'False Negatives ({len(fns)})\nDruggable misclassified')

    # 7. LOPO score 分布
    ax7=fig.add_subplot(2,4,7)
    dm=y==1; um=y==0
    ax7.hist(lopo_l[dm],bins=15,color='#4CAF50',alpha=0.7,
            label=f'Drug n={dm.sum()}',density=True)
    ax7.hist(lopo_l[um],bins=15,color='#F44336',alpha=0.7,
            label=f'Undrug n={um.sum()}',density=True)
    ax7.axvline(threshold,color='black',ls='--',lw=2,label=f'threshold')
    ax7.set_title(f'LOPO Score Distribution\nROC={auc_lopo_l:.4f}')
    ax7.legend(fontsize=8)

    # 8. 完整指標對比
    ax8=fig.add_subplot(2,4,8)
    methods=['fpocket','v3','Best','LOPO']
    roc_vals=[auc_s(y,score_fp),auc_s(y,score_v3),
              auc_s(y,score_best),auc_lopo_l]
    pr_vals=[average_precision_score(y,score_fp),
             average_precision_score(y,score_v3),
             average_precision_score(y,score_best),pr_lopo_l]
    x8=np.arange(4); w8=0.35
    ax8.bar(x8-w8/2,roc_vals,w8,color='#2196F3',alpha=0.85,label='ROC-AUC')
    ax8.bar(x8+w8/2,pr_vals,w8,color='#E91E63',alpha=0.85,label='PR-AUC')
    ax8.axhline(0.74,color='orange',ls='--',lw=1.5,label='SiteMap ROC')
    for i,(r,p) in enumerate(zip(roc_vals,pr_vals)):
        ax8.text(i-w8/2,r+0.01,f'{r:.3f}',ha='center',fontsize=7)
        ax8.text(i+w8/2,p+0.01,f'{p:.3f}',ha='center',fontsize=7)
    ax8.set_xticks(x8); ax8.set_xticklabels(methods)
    ax8.set_ylim(0.5,1.05); ax8.set_title('ROC vs PR-AUC')
    ax8.legend(fontsize=8)

    plt.suptitle(
        f'Luoshu Druggability 完整審計 v2（n={len(records)}, D={n_d}, U={n_u}）\n'
        f'Best: ROC={auc_s(y,score_best):.4f}  PR={average_precision_score(y,score_best):.4f}  '
        f'F1={f1:.4f}  LOPO-ROC={auc_lopo_l:.4f}  LOPO-PR={pr_lopo_l:.4f}',
        fontsize=11,fontweight='bold')
    plt.tight_layout()
    plt.savefig('druggability_audit_v2_result.png',dpi=150,bbox_inches='tight')
    print(f"\n圖表：druggability_audit_v2_result.png")
    print(f"耗時：{time.time()-t0:.1f}s")

if __name__=='__main__':
    main()

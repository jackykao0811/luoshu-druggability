"""
Luoshu Druggability: Bootstrap CI + DeLong Test + Paired Comparison
補齊 GPT 建議最後一塊：統計信賴區間

輸出：
  - ROC-AUC 95% CI（bootstrap）
  - PR-AUC 95% CI（bootstrap）
  - LOPO bootstrap CI
  - DeLong test（Luoshu vs fpocket-style）
  - Paired bootstrap：ΔROC、ΔPR，p-value

作者：Kao, Yao-Kai (NYCU)
日期：2026-05-08
"""

import os, requests, math, time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

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
    sph=(36*math.pi*vol**2)**(1/3)/(area+1e-8)
    pmask=chc>np.percentile(chc,70)
    hf=(sum(1 for i in range(n) if pmask[i] and aa[i] in HYDROPHOBIC)
        /max(pmask.sum(),1))
    pocket_ca=ca[chc>1.0]
    idim=2.0
    if len(pocket_ca)>=4:
        c=pocket_ca-pocket_ca.mean(0)
        try:
            _,s2,_=np.linalg.svd(c,full_matrices=False)
            lam=s2**2; lam=lam[lam>1e-10]
            idim=float(min((lam.sum())**2/(lam**2).sum(),3.0))
        except: pass
    return dict(
        sphericity=sph, chc_max=chc_max, hydro_frac=hf,
        sph_chc=sph*chc_max, sph_hf=sph*hf,
        sph_chc_hf=sph*chc_max*hf,
        idim=idim, idim_chc=idim/3.0*chc_max,
        fp_vol=vol, fp_area=area, fp_hydro=hf,
        fp_score=hf*vol**(1/3)/(area/(vol+1e-8)+1e-8),
    )

def nrm(x): r=x.max()-x.min(); return (x-x.min())/(r+1e-8)

# ── DeLong test（非參數 AUC 比較）────────────────────────
def delong_test(y, score1, score2):
    """
    DeLong et al. 1988 paired AUC test
    H0: AUC1 == AUC2
    Returns z-score and two-tailed p-value
    """
    from scipy import stats
    n = len(y)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    n1=len(pos); n0=len(neg)

    def placement(sc, pos_idx, neg_idx):
        # V10[i] = P(score[neg_j] < score[pos_i]) for each pos_i
        # V01[j] = P(score[neg_j] < score[pos_i]) for each neg_j
        V10 = np.array([np.mean(sc[neg_idx] < sc[i]) +
                        0.5*np.mean(sc[neg_idx]==sc[i]) for i in pos_idx])
        V01 = np.array([np.mean(sc[i] < sc[pos_idx]) +
                        0.5*np.mean(sc[i]==sc[pos_idx]) for i in neg_idx])
        return V10, V01

    V10_1, V01_1 = placement(score1, pos, neg)
    V10_2, V01_2 = placement(score2, pos, neg)

    auc1 = V10_1.mean()
    auc2 = V10_2.mean()

    # Covariance matrix of (AUC1, AUC2)
    S10 = np.cov(np.vstack([V10_1, V10_2])) / n1
    S01 = np.cov(np.vstack([V01_1, V01_2])) / n0

    S = S10 + S01  # 2x2 covariance

    delta = auc1 - auc2
    var_delta = S[0,0] + S[1,1] - 2*S[0,1]
    if var_delta <= 0: return delta, 1.0, 0.0
    z = delta / np.sqrt(var_delta)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return delta, p, z

# ── Bootstrap CI ─────────────────────────────────────────
def bootstrap_ci(y, score, n_boot=5000, seed=42, metric='roc'):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ys = y[idx]; ss = score[idx]
        if len(np.unique(ys)) < 2: continue
        try:
            if metric=='roc':
                a = roc_auc_score(ys, ss)
                vals.append(a if a>=0.5 else 1-a)
            else:
                vals.append(average_precision_score(ys, ss))
        except: pass
    a = np.array(vals)
    return float(np.percentile(a,2.5)), float(np.percentile(a,97.5)), float(np.std(a))

def bootstrap_paired_ci(y, s1, s2, n_boot=5000, seed=42, metric='roc'):
    """Bootstrap CI for difference s1 - s2"""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ys=y[idx]; ss1=s1[idx]; ss2=s2[idx]
        if len(np.unique(ys))<2: continue
        try:
            if metric=='roc':
                a1=roc_auc_score(ys,ss1); a1=a1 if a1>=0.5 else 1-a1
                a2=roc_auc_score(ys,ss2); a2=a2 if a2>=0.5 else 1-a2
            else:
                a1=average_precision_score(ys,ss1)
                a2=average_precision_score(ys,ss2)
            diffs.append(a1-a2)
        except: pass
    d=np.array(diffs)
    lo=float(np.percentile(d,2.5)); hi=float(np.percentile(d,97.5))
    # p-value: fraction of bootstrap samples where diff <= 0
    p=float(np.mean(d<=0))
    return float(d.mean()), lo, hi, p

def main():
    print("="*65)
    print("Luoshu Druggability: Bootstrap CI + DeLong + Paired Test")
    print("="*65)
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
        if len(res)<15: failed.append(name); continue
        feats=compute_features(res)
        if feats is None: failed.append(name); continue
        feats.update({'name':name,'label':label,'type':typ})
        records.append(feats)

    print(f"有效：{len(records)}  失敗：{len(failed)}\n")
    y=np.array([r['label'] for r in records])
    n_d=int(y.sum()); n_u=int((1-y).sum())
    print(f"Drug={n_d}  Undrug={n_u}\n")

    FEATS=['sphericity','chc_max','sph_chc_hf','sph_hf']
    Xn={f:nrm(np.array([r[f] for r in records])) for f in FEATS}
    Xfp=nrm(0.5*np.array([r['fp_hydro'] for r in records])+
             0.3*nrm(np.array([r['fp_vol'] for r in records]))+
             0.2*nrm(np.array([r['fp_score'] for r in records])))

    score_luoshu = (0.60*Xn['sph_chc_hf'] +
                    0.25*Xn['sphericity']  +
                    0.15*Xn['sph_hf'])
    score_fp     = Xfp
    score_v3     = 0.7*Xn['sphericity'] + 0.3*Xn['chc_max']

    N_BOOT = 5000

    # ══════════════════════════════════════════════════════
    # 1. INDIVIDUAL BOOTSTRAP CI
    # ══════════════════════════════════════════════════════
    print("═"*55)
    print("1. Bootstrap 95% CI（n=5000 iterations）")
    print("═"*55)
    print(f"  {'方法':30} {'ROC-AUC':7}  {'95% CI':16}  {'PR-AUC':7}  {'95% CI'}")

    for nm,sc in [('fpocket-style naive',score_fp),
                  ('Luoshu v3 Linear',score_v3),
                  ('Luoshu Best',score_luoshu)]:
        roc = roc_auc_score(y,sc); roc=roc if roc>=0.5 else 1-roc
        pr  = average_precision_score(y,sc)
        rlo,rhi,_  = bootstrap_ci(y,sc,N_BOOT,metric='roc')
        plo,phi2,_ = bootstrap_ci(y,sc,N_BOOT,metric='pr')
        print(f"  {nm:30} {roc:.4f}   [{rlo:.4f},{rhi:.4f}]   "
              f"{pr:.4f}   [{plo:.4f},{phi2:.4f}]")

    # ══════════════════════════════════════════════════════
    # 2. DELONG TEST
    # ══════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("2. DeLong Test（Luoshu vs fpocket-style）")
    print("═"*55)
    delta_roc, p_delong, z_delong = delong_test(y, score_luoshu, score_fp)
    roc_l = roc_auc_score(y,score_luoshu); roc_l=roc_l if roc_l>=0.5 else 1-roc_l
    roc_f = roc_auc_score(y,score_fp);    roc_f=roc_f if roc_f>=0.5 else 1-roc_f
    print(f"  Luoshu ROC-AUC:      {roc_l:.4f}")
    print(f"  fpocket-style ROC:   {roc_f:.4f}")
    print(f"  ΔROC-AUC:            {delta_roc:+.4f}")
    print(f"  DeLong z-score:      {z_delong:.4f}")
    print(f"  DeLong p-value:      {p_delong:.4f}  {'★ p<0.05' if p_delong<0.05 else ''} {'★★ p<0.01' if p_delong<0.01 else ''}")

    # ══════════════════════════════════════════════════════
    # 3. PAIRED BOOTSTRAP（ROC + PR difference CI）
    # ══════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("3. Paired Bootstrap：Luoshu vs fpocket-style")
    print("═"*55)

    dm_roc,lo_roc,hi_roc,p_roc = bootstrap_paired_ci(y,score_luoshu,score_fp,N_BOOT,metric='roc')
    dm_pr, lo_pr, hi_pr, p_pr  = bootstrap_paired_ci(y,score_luoshu,score_fp,N_BOOT,metric='pr')

    print(f"  ΔROC-AUC = {dm_roc:+.4f}  95% CI [{lo_roc:+.4f}, {hi_roc:+.4f}]  "
          f"bootstrap p = {p_roc:.4f}  {'★' if p_roc<0.05 else ''}")
    print(f"  ΔPR-AUC  = {dm_pr:+.4f}  95% CI [{lo_pr:+.4f}, {hi_pr:+.4f}]  "
          f"bootstrap p = {p_pr:.4f}  {'★' if p_pr<0.05 else ''}")

    print(f"\n  → 可寫入論文：")
    print(f"    Luoshu vs fpocket-style：")
    print(f"    ΔROC = {dm_roc:+.4f} (95% CI [{lo_roc:.4f},{hi_roc:.4f}], p={p_roc:.4f})")
    print(f"    ΔPR  = {dm_pr:+.4f} (95% CI [{lo_pr:.4f},{hi_pr:.4f}], p={p_pr:.4f})")

    # ══════════════════════════════════════════════════════
    # 4. Luoshu vs v3 paired test
    # ══════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("4. Paired Bootstrap：Luoshu Best vs v3 Linear")
    print("═"*55)
    dm_r2,lo_r2,hi_r2,p_r2 = bootstrap_paired_ci(y,score_luoshu,score_v3,N_BOOT,metric='roc')
    dm_p2,lo_p2,hi_p2,p_p2 = bootstrap_paired_ci(y,score_luoshu,score_v3,N_BOOT,metric='pr')
    print(f"  ΔROC = {dm_r2:+.4f}  95% CI [{lo_r2:+.4f},{hi_r2:+.4f}]  p={p_r2:.4f}")
    print(f"  ΔPR  = {dm_p2:+.4f}  95% CI [{lo_p2:+.4f},{hi_p2:+.4f}]  p={p_p2:.4f}")

    # ══════════════════════════════════════════════════════
    # 5. LOPO bootstrap CI
    # ══════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("5. LOPO Bootstrap CI（重新計算 LOPO 分數）")
    print("═"*55)
    print("  計算 LOPO scores...")
    lopo_scores=[]
    FEATS2=['sphericity','chc_max','sph_chc_hf','sph_hf']
    for i in range(len(records)):
        tidx=[j for j in range(len(records)) if j!=i]
        Xt={}
        for f in FEATS2:
            v=np.array([records[j][f] for j in range(len(records))])
            tv=v[tidx]; mn=tv.min(); mx=tv.max(); r=mx-mn
            Xt[f]=(v-mn)/(r+1e-8)
        vfp=(0.5*np.array([r['fp_hydro'] for r in records])+
             0.3*np.array([r['fp_vol'] for r in records])+
             0.2*np.array([r['fp_score'] for r in records]))
        tv2=vfp[tidx]; mn2=tv2.min(); mx2=tv2.max(); r2=mx2-mn2
        Xt['fp']=(vfp-mn2)/(r2+1e-8)
        lopo_scores.append(float(0.60*Xt['sph_chc_hf'][i]+
                                 0.25*Xt['sphericity'][i]+
                                 0.15*Xt['sph_hf'][i]))

    lopo_sc=np.array(lopo_scores)
    lopo_roc=roc_auc_score(y,lopo_sc); lopo_roc=lopo_roc if lopo_roc>=0.5 else 1-lopo_roc
    lopo_pr=average_precision_score(y,lopo_sc)
    rlo2,rhi2,_ = bootstrap_ci(y,lopo_sc,N_BOOT,metric='roc')
    plo2,phi3,_ = bootstrap_ci(y,lopo_sc,N_BOOT,metric='pr')
    print(f"  LOPO ROC-AUC = {lopo_roc:.4f}  95% CI [{rlo2:.4f},{rhi2:.4f}]")
    print(f"  LOPO PR-AUC  = {lopo_pr:.4f}  95% CI [{plo2:.4f},{phi3:.4f}]")

    # ══════════════════════════════════════════════════════
    # 完整論文級摘要
    # ══════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("★ 論文級統計摘要（可直接貼入 Methods/Results）")
    print("═"*65)
    roc_best=roc_auc_score(y,score_luoshu); roc_best=roc_best if roc_best>=0.5 else 1-roc_best
    pr_best=average_precision_score(y,score_luoshu)
    rlo_b,rhi_b,_=bootstrap_ci(y,score_luoshu,N_BOOT,metric='roc')
    plo_b,phi_b,_=bootstrap_ci(y,score_luoshu,N_BOOT,metric='pr')
    print(f"""
  Performance (n={len(records)}, Drug={n_d}, Undrug={n_u}):
    Luoshu Best:
      ROC-AUC = {roc_best:.4f} (95% CI [{rlo_b:.4f},{rhi_b:.4f}])
      PR-AUC  = {pr_best:.4f} (95% CI [{plo_b:.4f},{phi_b:.4f}])
      LOPO-ROC= {lopo_roc:.4f} (95% CI [{rlo2:.4f},{rhi2:.4f}])
      LOPO-PR = {lopo_pr:.4f} (95% CI [{plo2:.4f},{phi3:.4f}])

  Head-to-head vs fpocket-style (same dataset, same labels):
      ΔROC-AUC = {dm_roc:+.4f} (95% CI [{lo_roc:.4f},{hi_roc:.4f}], bootstrap p={p_roc:.4f})
      ΔPR-AUC  = {dm_pr:+.4f} (95% CI [{lo_pr:.4f},{hi_pr:.4f}], bootstrap p={p_pr:.4f})
      DeLong z = {z_delong:.4f}, p = {p_delong:.4f}

  Improvement over v3 Linear:
      ΔROC = {dm_r2:+.4f} (95% CI [{lo_r2:.4f},{hi_r2:.4f}], p={p_r2:.4f})
      ΔPR  = {dm_p2:+.4f} (95% CI [{lo_p2:.4f},{hi_p2:.4f}], p={p_p2:.4f})
""")
    print(f"耗時：{time.time()-t0:.1f}s")

if __name__=='__main__':
    main()

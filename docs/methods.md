# Detailed Methods

## 1. Dataset Construction

### Druggable Proteins (n=87)
Sourced from DUD-E (Mysinger et al. 2012, J. Med. Chem.) and extended with additional well-characterized targets:
- **Kinases (26):** CDK2, EGFR, BRAF, Aurora-A, BCR-ABL, VEGFR2, SRC, p38-MAPK, CHK1, GSK3B, ROCK1, PLK1, MET, IGF1R, FGFR1, LCK, JAK2, AKT1, AKT2, CDK1, PIK3CA, mTOR, BTK, PDK1, and others
- **Proteases (11):** HIV-PR, Trypsin, MMP3, MMP2, FXa, Renin, BACE1, CathB, FURIN, PLpro (SARS-CoV-2), MPRO (SARS-CoV-2)
- **Nuclear Receptors (11):** PPARg, PPARd, AR, ERa, ERb, GR, PR, THRa, RXRa, FXR, LXRb
- **GPCRs (5):** ADRB1, ADRB2, CXCR4, DRD3, HRH1
- **Enzymes and other (34):** AChE, PARP1, PDE5, COX-2, HMGR, HDAC2, EZH2, IDH1, ACE2, BRD4, DOT1L, WDR5, MCL1, and others

### Undruggable Proteins (n=36)
Classified based on established pharmacological literature (Hopkins & Groom 2002, Nat Rev Drug Discov):
- **GTPases (10):** KRAS, NRAS, HRAS, Cdc42, Rac1, RRAS, MRAS, RAL, RAP1, RHO
- **TF/PPI surfaces (15):** MYC, HIF-1a, STAT3, NF-kB, p65, FOXO1, YAP1, TEAD1, NOTCH1, cMyc-Max, p53-MDM2, BCL2*, XIAP, Survivin, b-catenin
- **Structural/IDP (11):** Tubulin, Actin, Vimentin, Lamin-A, Tau/MAPT, Grb2, 14-3-3, PDZ domain, CaM, FLNA, PCNA

*BCL2 is included as undruggable per DUD-E classification, though it is clinically targeted by venetoclax — treated as a label-ambiguous case.

### Exclusion Criteria
- Fewer than 15 resolved backbone N/CA/C atoms in the first chain
- 5 structures excluded on this basis

## 2. Feature Computation

### Step 1: Luoshu Trigram Assignment
For each residue i with backbone dihedral angles (φ_i, ψ_i):
```
b1(i) = 1 if |φ_i| ≤ 90°  else 0
b2(i) = 1 if φ_i > 0°     else 0
b3(i) = 1 if |ψ_i| ≤ 60°  else 0
```
The trigram (b1, b2, b3) maps to Luoshu number L via the 3×3 Luoshu magic square encoding:
```
(1,1,1)→9, (1,1,0)→7, (1,0,1)→3, (1,0,0)→1
(0,1,1)→2, (0,1,0)→6, (0,0,1)→8, (0,0,0)→4
```

### Step 2: Local Transition Energy (LTE)
```
LTE_w(i) = Σ_{j≠i, |j-i|≤w} |L(i) - L(j)| / |i-j|  /  Σ_{j≠i, |j-i|≤w} 1/|i-j|
```
computed for windows w ∈ {1, 3, 5, 7}.

Golden-ratio-weighted combination:
```
LTE(i) = Σ_k φ^(-k) × LTE_{w_k}(i)  /  Σ_k φ^(-k)
```
where φ = 1.6180339887 (golden ratio), k = 0,1,2,3.

### Step 3: Pocket Definition
Normalize LTE to [0,1]. Residues in the top 30th percentile define the putative binding pocket.

### Step 4: Convex Hull Features
Compute convex hull of pocket CA coordinates:
- **Volume** V and **surface area** A
- **Sphericity**: S = (36π V²)^(1/3) / A  (range 0–1; sphere = 1)
- **CHC_max**: For each non-pocket CA, if inside the convex hull, compute its minimum perpendicular distance to any hull facet. CHC_max = maximum over all such distances.

### Step 5: Additional Features
- **f_h**: Fraction of pocket-region residues (CHC > 70th percentile) that are hydrophobic {A,C,F,G,H,I,L,M,P,V,W,Y}
- **Intrinsic dimensionality (idim)**: Participation ratio of PCA singular values of pocket CA coordinates = (Σλ)² / Σλ², clipped to [0,3]
- **Polynomial features**: sph×chc = S×CHC, sph×hf = S×f_h, sph×chc×hf = S×CHC×f_h

### Step 6: fpocket-Inspired Baseline
To enable transparent same-data comparison:
```
score_fp = 0.5 × f_h + 0.3 × norm(V) + 0.2 × norm(f_h × V^(1/3) / AV)
```
where AV = area-to-volume ratio. This reflects fpocket's volume-hydrophobicity logic using identical CA input.

## 3. Scoring Model

**Luoshu Best:**
```
score = 0.60 × norm(sph×chc×hf) + 0.25 × norm(sphericity) + 0.15 × norm(sph×hf)
```
where norm() = min-max normalization over the dataset. No parameters are fitted from data.

## 4. Validation Protocol

### Full-Sample Evaluation
ROC-AUC and PR-AUC computed on all 123 proteins. Permutation test: 2000 random label shuffles; p = fraction where permuted AUC ≥ observed AUC.

### Bootstrap 95% CI
5000-iteration stratified bootstrap resampling (random seed = 42). CI = [2.5th, 97.5th] percentile of bootstrap distribution.

### Leave-One-Protein-Out (LOPO)
123 iterations. In each iteration:
1. Remove protein i
2. Recompute feature min-max normalization from proteins {1,...,n}\{i}
3. Apply normalization to protein i
4. Compute score for protein i using fixed weight vector

Aggregate LOPO ROC-AUC and PR-AUC computed from all 123 held-out predictions.

### DeLong Test
Paired AUC comparison (DeLong et al. 1988, Biometrics 44:837-845) for ROC-AUC difference. Computes V10 and V01 placement matrices, derives covariance of AUC estimates, and reports z-score and two-tailed p-value.

### Paired Bootstrap (PR-AUC)
5000-iteration paired bootstrap for PR-AUC difference. p-value = fraction of bootstrap samples where Δ ≤ 0.

## 5. Confusion Matrix Threshold
Threshold set at the (1 - n_drug/n_total) quantile of predicted scores, ensuring the fraction of proteins predicted as druggable equals the true prevalence.

## 6. Computational Requirements
- Input: PDB file (backbone N/CA/C only)
- Runtime: < 1 second per protein (Python, no GPU)
- Dependencies: numpy, scipy, scikit-learn, matplotlib, requests
- PDB files: downloaded automatically from RCSB (https://files.rcsb.org/)

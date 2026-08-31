import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json

t  = pd.read_csv("results/twin_results.csv")
TW = json.load(open("results/twin_analysis.json"))
TX = json.load(open("results/twin_extra.json"))
EX = json.load(open("results/extra_statistics.json"))["detector_specific"]
CC = json.load(open("results/concentration_analysis.json"))

RED,ORANGE,TEAL,BLUE,BROWN = "#C94F4A","#E8943A","#4AACB0","#5B8DB8","#9C6B4A"
INK,GREY = "#33373B","#6E6E6E"
FS_LAB,FS_AX,FS_TK,FS_LG,FS_AN,FS_BAR = 28,16,15,15,14,12
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"]})

fig,ax = plt.subplots(2,2,figsize=(13,6.5),dpi=200)
plt.subplots_adjust(hspace=0.90,wspace=0.44,top=0.855,bottom=0.10,left=0.085,right=0.99)

def style(a):
    a.set_facecolor("white"); a.grid(True,color="#DDDDDD",ls="--",lw=0.6,zorder=0)
    a.set_axisbelow(True)
    for s in ("top","right"): a.spines[s].set_visible(False)
    a.tick_params(labelsize=FS_TK)
def lab(a,s):
    a.text(0.0,1.40,s,transform=a.transAxes,fontsize=FS_LAB,fontweight="bold",
           ha="left",va="bottom",clip_on=False,color=INK)

# ---------------- (A) twin scatter ----------------
a=ax[0,0]; style(a)
for g,c,m in [("CO",RED,"o"),("Ethanol",ORANGE,"s"),("Ethylene",TEAL,"^"),("Methane",BLUE,"D")]:
    s=t[t.unknown_name==g]
    a.scatter(s.DNR_maha,s.AUROC_maha,s=85,color=c,marker=m,alpha=0.85,
              edgecolor="white",lw=0.8,label=f"{g} withheld",zorder=3)
a.set_xscale("log"); a.set_xlim(0.25,4.4); a.set_ylim(-0.03,1.85)
a.axvline(1,color=INK,ls="--",lw=1.8,zorder=2,label="magnitude crossover")
a.axvline(TX["breakpoint_DNR"],color=ORANGE,ls="-.",lw=1.8,zorder=2,
          label=f"fitted breakpoint ({TX['breakpoint_DNR']:.2f})")
a.axhline(0.5,color=BROWN,ls=":",lw=1.8,zorder=2,label="chance")
a.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
a.set_xlabel("DNR  (Mahalanobis, log scale)",fontsize=FS_AX)
a.set_ylabel("Open-set AUROC",fontsize=FS_AX)
a.set_title("Twin arrays: the association reappears\nunder unit-to-unit shift",fontsize=FS_AX,pad=6)
a.text(0.02,0.02,f"Spearman $\\rho$ = {TW['correlations']['DNR_maha|maha']['rho']:.3f}"
                 f"   (n = {TW['n_splits']})",transform=a.transAxes,
       fontsize=FS_AN-1,ha="left",va="bottom",
       bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#333333",lw=1.0))
a.legend(fontsize=FS_LG-4,loc="upper left",frameon=False,ncol=2,columnspacing=1.0,
         handlelength=1.8,labelspacing=0.25)

# ---------------- (B) descriptors, both datasets ----------------
a=ax[0,1]; style(a)
rows=[("DNR (Mahalanobis)","DNR_maha","DNR_maha"),
      ("DNR (energy)","DNR_energy","DNR_energy"),
      ("DNR (Euclidean)","DNR_euclid","DNR_euclid"),
      ("$d_{drift}$ alone","d_drift_maha","d_drift"),
      ("$d_{novel}$ alone","d_novel_maha","d_novel")]
y=np.arange(len(rows)); h=0.36
series={}
for off,col,src,lb in [(-h/2,RED,"drift","Drift dataset (n = 95)"),
                       (+h/2,TEAL,"twin","Twin dataset (n = 80)")]:
    vs,lo_,hi_=[],[],[]
    for _,kd,kt in rows:
        r = EX["Mahalanobis"][kd] if src=="drift" else TW["correlations"][f"{kt}|maha"]
        lo,hi = r["ci_cluster"] if src=="drift" else r["cluster_ci"]
        vs.append(r["rho"]); lo_.append(r["rho"]-lo); hi_.append(hi-r["rho"])
    a.barh(y+off,vs,h,color=col,zorder=3,label=lb,
           xerr=np.array([lo_,hi_]),error_kw=dict(ecolor=INK,elinewidth=1.2,capsize=3,alpha=0.9))
    series[src]=(vs,lo_,hi_)
# one label per row: drift / twin, placed clear of every whisker
LBLX = -1.92
for i in range(len(rows)):
    a.text(LBLX,y[i],f"{series['drift'][0][i]:+.2f} / {series['twin'][0][i]:+.2f}",
           va="center",ha="left",fontsize=FS_BAR,color=INK)
a.axvline(0,color=INK,lw=1.2,zorder=2)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows],fontsize=FS_TK-1)
a.set_xlim(-2.00,1.45); a.set_ylim(5.05,-0.85)
a.set_xlabel("Spearman $\\rho$ with open-set AUROC\n(cluster bootstrap CI)",fontsize=FS_AX)
a.set_title("The ratio outranks the magnitudes\non both datasets",fontsize=FS_AX,pad=6)
a.legend(fontsize=FS_LG-3,loc="upper right",frameon=False)

# ---------------- (C) detector specificity ----------------
a=ax[1,0]; style(a)
dets=[("Mahalanobis\nscore","Mahalanobis","maha"),("5-NN\ndistance","kNN","knn"),
      ("softmax\nconfidence","MSP","msp")]
x=np.arange(len(dets)); w=0.36
for off,col,src,lb in [(-w/2,RED,"drift","Drift dataset"),(+w/2,TEAL,"twin","Twin dataset")]:
    vs,lo_,hi_=[],[],[]
    for _,dk,tk in dets:
        r = EX[dk]["DNR_maha"] if src=="drift" else TW["correlations"][f"DNR_maha|{tk}"]
        lo,hi = r["ci_cluster"] if src=="drift" else r["cluster_ci"]
        vs.append(r["rho"]); lo_.append(r["rho"]-lo); hi_.append(hi-r["rho"])
    a.bar(x+off,vs,w,color=col,zorder=3,label=lb,
          yerr=np.array([lo_,hi_]),error_kw=dict(ecolor=INK,elinewidth=1.2,capsize=3,alpha=0.9))
    for i,v in enumerate(vs):
        yy = v-lo_[i]-0.075 if v<0 else v+hi_[i]+0.045
        a.text(x[i]+off,yy,f"{v:+.2f}",ha="center",va="center",fontsize=FS_BAR,color=INK)
a.axhline(0,color=INK,lw=1.2,zorder=2)
a.set_xticks(x); a.set_xticklabels([d[0] for d in dets])
a.set_ylim(-1.12,1.10)
a.set_ylabel("Spearman $\\rho$  (DNR vs AUROC)",fontsize=FS_AX)
a.set_xlabel("Novelty score",fontsize=FS_AX)
a.set_title("DNR predicts distance-based scores;\nconfidence moves the other way",
            fontsize=FS_AX,pad=6)
a.legend(fontsize=FS_LG-3,loc="upper left",frameon=False)

# ---------------- (D) concentration control ----------------
a=ax[1,1]; style(a)
bars=[("unconditional  (n = 95)",CC["Q2"]["unconditional"],GREY),
      ("| unknown-known gap",CC["Q2"]["ctrl_gap"],BLUE),
      ("| gap + batch shift",CC["Q2"]["ctrl_gap_shift"],BLUE),
      ("| gap + shift + spread",CC["Q2"]["ctrl_all"],BLUE),
      ("full range  (n = 68)",CC["Q3"]["rho_same_splits_full"],ORANGE),
      ("50-250 ppmv only  (n = 68)",CC["Q3"]["rho_within_window"],TEAL)]
yy=np.arange(len(bars))
a.barh(yy,[abs(b[1]) for b in bars],color=[b[2] for b in bars],height=0.62,zorder=3)
for i,b in enumerate(bars):
    a.text(abs(b[1])+0.018,i,f"{b[1]:.3f}",va="center",fontsize=FS_BAR,color=INK)
a.set_yticks(yy); a.set_yticklabels([b[0] for b in bars],fontsize=FS_TK-1)
a.set_ylim(5.7,-0.7); a.set_xlim(0,1.15)
a.set_xlabel("|Spearman $\\rho$| between DNR\nand open-set AUROC",fontsize=FS_AX)
a.set_title("Concentration does not account\nfor the association",fontsize=FS_AX,pad=6)

for s,a_ in zip(["(a)","(b)","(c)","(d)"],[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]): lab(a_,s)
# Figure 3 in the submitted manuscript
fig.savefig("results/fig_03_external.png",dpi=200,bbox_inches="tight",
            pad_inches=0.25,facecolor="white")
print("twin rho",round(TW["correlations"]["DNR_maha|maha"]["rho"],3),
      "bp",round(TX["breakpoint_DNR"],3))

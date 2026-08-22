import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv("results/raw_results.csv")
d["pair"] = d.src.astype(str) + "->" + d.tgt.astype(str)
u = d.drop_duplicates(subset=["pair", "unknown"]).copy()   # 1->2 belongs to both settings

RED,ORANGE,TEAL,BLUE,BROWN = "#C94F4A","#E8943A","#4AACB0","#5B8DB8","#9C6B4A"
INK,GREY = "#33373B","#6E6E6E"
FS_LAB,FS_AX,FS_TK,FS_LG,FS_AN,FS_BAR = 28,16,15,13,12,10
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"]})

fig,ax = plt.subplots(2,2,figsize=(20,10),dpi=200)
plt.subplots_adjust(hspace=0.55,wspace=0.38,top=0.88,bottom=0.09,left=0.06,right=0.97)

def style(a):
    a.set_facecolor("white"); a.grid(True,color="#DDDDDD",ls="--",lw=0.6,zorder=0)
    a.set_axisbelow(True)
    for s in ("top","right"): a.spines[s].set_visible(False)
    a.tick_params(labelsize=FS_TK)
def lab(a,t):
    a.text(0.0,1.18,t,transform=a.transAxes,fontsize=FS_LAB,fontweight="bold",
           ha="left",va="bottom",clip_on=False,color=INK)

# (A) DNR vs AUROC
a=ax[0,0]; style(a)
for s,c,m in [("Setting 1",RED,"o"),("Setting 2",TEAL,"s")]:
    g=u[u.setting==s]
    a.scatter(g.DNR_maha,g.AUROC_maha,s=80,color=c,marker=m,alpha=0.85,
              edgecolor="white",lw=0.8,label=s,zorder=3)
a.set_xscale("log"); a.set_xlim(0.02,20); a.set_ylim(-0.03,1.05)
a.axvline(1,color=INK,ls="--",lw=1.8,zorder=2)
a.axhline(0.5,color=BROWN,ls=":",lw=1.8,zorder=2)
rho,p = spearmanr(u.DNR_maha,u.AUROC_maha)
a.set_xlabel("DNR  (Mahalanobis, log scale)",fontsize=FS_AX)
a.set_ylabel("Open-set AUROC",fontsize=FS_AX)
a.set_title("Novelty detection collapses, then inverts, as DNR passes 1",fontsize=FS_AX,pad=6)
a.text(1.15,0.02,"DNR = 1",fontsize=FS_AN,color=INK,rotation=90,va="bottom")
a.text(0.024,0.52,"chance",fontsize=FS_AN,color=BROWN,va="bottom")
a.text(0.98,0.97,f"Spearman $\\rho$ = {rho:.3f}  (n = {len(u)})",transform=a.transAxes,
       fontsize=FS_AN,ha="right",va="top",
       bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#333333",lw=1.0))
a.legend(fontsize=FS_LG,loc="lower left",frameon=False)

# (B) predictor comparison
a=ax[0,1]; style(a)
preds=[("DNR (Mahalanobis)","DNR_maha",RED),("DNR (energy)","DNR_energy",RED),
       ("DNR (Euclidean)","DNR_euclid",RED),("MMD ratio","MMD_ratio",ORANGE),
       ("$d_{drift}$ alone","d_drift_maha",BLUE),("$d_{novel}$ alone","d_novel_maha",BLUE),
       ("MMD (shift only)","MMD_drift",BLUE),("proxy A-distance","PAD_drift",BLUE),
       ("batch interval","interval",BLUE)]
vals=[abs(spearmanr(u[k],u.AUROC_maha)[0]) for _,k,_ in preds]
order=np.argsort(vals)
names=[preds[i][0] for i in order]; cols=[preds[i][2] for i in order]
v=[vals[i] for i in order]
a.barh(range(len(v)),v,color=cols,height=0.68,zorder=3)
for i,x in enumerate(v): a.text(x+0.012,i,f"{x:.2f}",va="center",fontsize=FS_BAR,color=INK)
a.set_yticks(range(len(v))); a.set_yticklabels(names,fontsize=FS_TK-1)
a.set_xlim(0,0.95); a.set_xlabel("|Spearman $\\rho$| with open-set AUROC",fontsize=FS_AX)
a.axvspan(0,0.40,color=GREY,alpha=0.07,zorder=1)
a.text(0.395,-0.62,"magnitude-only measures",fontsize=FS_AN,color=GREY,ha="right",va="center")
a.set_title("Ratios predict; domain-shift magnitudes do not",fontsize=FS_AX,pad=6)

# (C) DNR band vs AUROC
a=ax[1,0]; style(a)
bands=pd.cut(u.DNR_maha,[0,0.5,1,2,100],labels=["< 0.5","0.5 – 1","1 – 2","> 2"])
grp=[u.AUROC_maha[bands==b].values for b in ["< 0.5","0.5 – 1","1 – 2","> 2"]]
bp=a.boxplot(grp,patch_artist=True,widths=0.55,medianprops=dict(color=INK,lw=2.2),
             flierprops=dict(marker="o",ms=5,mfc=GREY,mec="none",alpha=0.6))
for patch,c in zip(bp["boxes"],[TEAL,TEAL,ORANGE,RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(INK); patch.set_lw(1.1)
a.axhline(0.5,color=BROWN,ls=":",lw=1.8,zorder=2)
a.set_xticklabels([f"{b}\n(n={len(g)})" for b,g in
                   zip(["< 0.5","0.5 – 1","1 – 2","> 2"],grp)])
a.set_xlabel("DNR band",fontsize=FS_AX); a.set_ylabel("Open-set AUROC",fontsize=FS_AX)
a.set_ylim(-0.05,1.08)
a.set_title("Median AUROC falls below chance once DNR exceeds 1",fontsize=FS_AX,pad=6)

# (D) RQ4 compensation
a=ax[1,1]; style(a)
w=0.34; xs=np.arange(2)
dr=[ (u[f"d_drift_maha_{m}"]/u.d_drift_maha).median() for m in ("mean","coral")]
dn=[ (u[f"d_novel_maha_{m}"]/u.d_novel_maha).median() for m in ("mean","coral")]
a.bar(xs-w/2,dr,w,color=ORANGE,label="$d_{drift}$ retained",zorder=3)
a.bar(xs+w/2,dn,w,color=TEAL,label="$d_{novel}$ retained",zorder=3)
for x,y in zip(xs-w/2,dr): a.text(x,y+0.02,f"{y:.2f}",ha="center",fontsize=FS_BAR,color=INK)
for x,y in zip(xs+w/2,dn): a.text(x,y+0.02,f"{y:.2f}",ha="center",fontsize=FS_BAR,color=INK)
a.axhline(1.0,color=GREY,ls="--",lw=1.2,zorder=2)
a.set_xticks(xs); a.set_xticklabels(["Mean-shift","CORAL"])
a.set_ylim(0,1.22); a.set_ylabel("Fraction of distance retained\nafter compensation",fontsize=FS_AX)
a.set_xlabel("Drift-compensation method",fontsize=FS_AX)
a.set_title("Compensation shrinks the novelty gap as well",fontsize=FS_AX,pad=6)
a.legend(fontsize=FS_LG,loc="upper center",frameon=False,ncol=2,bbox_to_anchor=(0.5,1.0))

for t,a_ in zip(["(A)","(B)","(C)","(D)"],[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]): lab(a_,t)
fig.savefig("results/fig_02_dnr_core.png",dpi=200,bbox_inches="tight",
            pad_inches=0.25,facecolor="white")
print("n_unique",len(u),"| rho",round(rho,3),"| p",f"{p:.2e}")
print("bands:",[f"{b}:{len(g)}:{np.median(g):.3f}" for b,g in zip(['<0.5','0.5-1','1-2','>2'],grp)])

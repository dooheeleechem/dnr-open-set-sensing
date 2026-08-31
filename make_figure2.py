import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr

d = pd.read_csv("results/raw_results.csv")
d["pair"] = d.src.astype(str) + "->" + d.tgt.astype(str)
u = d.drop_duplicates(subset=["pair", "unknown"]).copy()
EX = json.load(open("results/extra_statistics.json"))
CI = EX["detector_specific"]["Mahalanobis"]
SEG = EX["segmented_regression"]
C2 = json.load(open("results/extra_statistics2.json"))["compensation_unknown_contamination"]

RED,ORANGE,TEAL,BLUE,BROWN = "#C94F4A","#E8943A","#4AACB0","#5B8DB8","#9C6B4A"
INK,GREY = "#33373B","#6E6E6E"
FS_LAB,FS_AX,FS_TK,FS_LG,FS_AN,FS_BAR = 28,16,15,15,14,12
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"]})

fig,ax = plt.subplots(2,2,figsize=(13,6.5),dpi=200)
plt.subplots_adjust(hspace=0.72,wspace=0.44,top=0.87,bottom=0.10,left=0.075,right=0.99)

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
a.axvline(1,color=INK,ls="--",lw=1.8,zorder=2,label="magnitude crossover")
a.axvline(SEG["breakpoint_DNR"],color=ORANGE,ls="-.",lw=1.8,zorder=2,
          label=f"fitted breakpoint ({SEG['breakpoint_DNR']:.2f})")
a.axhline(0.5,color=BROWN,ls=":",lw=1.8,zorder=2,label="chance")
rho,p = spearmanr(u.DNR_maha,u.AUROC_maha)
a.set_xlabel("DNR  (Mahalanobis, log scale)",fontsize=FS_AX)
a.set_ylabel("Open-set AUROC",fontsize=FS_AX)
a.set_title("Performance falls, then inverts, as DNR grows",fontsize=FS_AX,pad=6)
a.text(0.03,0.97,f"Spearman $\\rho$ = {rho:.3f}   (n = {len(u)})",transform=a.transAxes,
       fontsize=FS_AN-1,ha="left",va="top",
       bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#333333",lw=1.0))
a.legend(fontsize=FS_LG-4,loc="lower left",frameon=False,ncol=1,
         handlelength=1.8,labelspacing=0.25)

# (B) predictor comparison with cluster CIs
a=ax[0,1]; style(a)
preds=[("DNR (Mahalanobis)","DNR_maha",RED),("DNR (energy)","DNR_energy",RED),
       ("DNR (Euclidean)","DNR_euclid",RED),("MMD ratio","MMD_ratio",ORANGE),
       ("$d_{drift}$ alone","d_drift_maha",BLUE),("$d_{novel}$ alone","d_novel_maha",BLUE),
       ("proxy A-distance","PAD_drift",BLUE),("MMD (shift only)","MMD_drift",BLUE),
       ("batch interval","interval",BLUE)]
vals=[abs(CI[k]["rho"]) for _,k,_ in preds]
errs=[[abs(CI[k]["rho"])-min(abs(v) for v in CI[k]["ci_cluster"]),
       max(abs(v) for v in CI[k]["ci_cluster"])-abs(CI[k]["rho"])] for _,k,_ in preds]
order=np.argsort(vals)
names=[preds[i][0] for i in order]; cols=[preds[i][2] for i in order]
v=[vals[i] for i in order]; e=np.array([errs[i] for i in order]).T
a.barh(range(len(v)),v,color=cols,height=0.68,zorder=3,
       xerr=e,error_kw=dict(ecolor=INK,elinewidth=1.3,capsize=3.5,alpha=0.85))
for i,x in enumerate(v): a.text(x+e[1][i]+0.02,i,f"{x:.2f}",va="center",fontsize=FS_BAR,color=INK)
a.set_yticks(range(len(v))); a.set_yticklabels(names,fontsize=FS_TK-1)
a.set_ylim(-0.65,9.55)
a.set_xlim(0,1.30); a.set_xlabel("|Spearman $\\rho$| with open-set AUROC\n(cluster bootstrap CI)",
                                 fontsize=FS_AX)
a.axvspan(0,0.40,color=GREY,alpha=0.07,zorder=1)
a.text(0.02,9.25,"magnitude-only measures shaded",fontsize=FS_AN-1,color=GREY,ha="left",va="center")
a.set_title("Ratios predict; domain-shift magnitudes predict less",fontsize=FS_AX,pad=6)

# (C) DNR band vs AUROC
a=ax[1,0]; style(a)
labs=["< 0.5","0.5 - 1","1 - 2","> 2"]
bands=pd.cut(u.DNR_maha,[0,0.5,1,2,100],labels=labs)
grp=[u.AUROC_maha[bands==b].values for b in labs]
bp=a.boxplot(grp,patch_artist=True,widths=0.55,medianprops=dict(color=INK,lw=2.2),
             flierprops=dict(marker="o",ms=5,mfc=GREY,mec="none",alpha=0.6))
for patch,c in zip(bp["boxes"],[TEAL,TEAL,ORANGE,RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(INK); patch.set_lw(1.1)
a.axhline(0.5,color=BROWN,ls=":",lw=1.8,zorder=2)
a.set_xticklabels([f"{b}\n(n={len(g)})" for b,g in zip(labs,grp)])
a.set_xlabel("DNR band",fontsize=FS_AX); a.set_ylabel("Open-set AUROC",fontsize=FS_AX)
a.set_ylim(-0.05,1.08)
a.set_title("Median AUROC falls below chance once DNR exceeds 1",fontsize=FS_AX,pad=6)

# (D) five compensation conditions
a=ax[1,1]; style(a)
cond=[("no compensation",u.DNR_maha.median(),u.AUROC_maha.median(),GREY,"o"),
      ("mean-shift",C2["mean"]["median_DNR_all_target"],C2["mean"]["median_AUROC_all_target"],ORANGE,"o"),
      ("mean-shift, oracle",C2["mean"]["median_DNR_known_only"],C2["mean"]["median_AUROC_known_only"],ORANGE,"D"),
      ("CORAL",C2["coral"]["median_DNR_all_target"],C2["coral"]["median_AUROC_all_target"],TEAL,"o"),
      ("CORAL, oracle",C2["coral"]["median_DNR_known_only"],C2["coral"]["median_AUROC_known_only"],TEAL,"D")]
xs=[c[1] for c in cond]; ys=[c[2] for c in cond]
o=np.argsort(xs)
a.plot(np.array(xs)[o],np.array(ys)[o],color=GREY,lw=1.4,ls="--",zorder=2)
for n,x,y,c,m in cond:
    a.scatter(x,y,s=200,color=c,marker=m,edgecolor="white",lw=1.2,zorder=4)
a.set_xscale("log"); a.set_xlim(0.070,3.60); a.set_ylim(0.545,1.115)
a.axhline(0.5,color=BROWN,ls=":",lw=1.5,zorder=1)
off={"no compensation":(1.10,-0.004,"left"),
     "mean-shift":(1.00,-0.058,"center"),
     "mean-shift, oracle":(1.10,-0.004,"left"),
     "CORAL":(1.00,+0.035,"center"),
     "CORAL, oracle":(1.00,-0.060,"center")}
for n,x,y,c,m in cond:
    fx,dy,ha=off[n]
    a.annotate(n,(x,y),xytext=(x*fx,y+dy),fontsize=FS_AN-1,color=INK,ha=ha)
a.set_xlabel("Median DNR after compensation  (log scale)",fontsize=FS_AX)
a.set_ylabel("Median open-set AUROC",fontsize=FS_AX)
a.set_title("Changing the geometry moves performance with it",fontsize=FS_AX,pad=6)
a.text(0.985,0.975,"circles: alignment from entire target\ndiamonds: alignment from known target only",
       transform=a.transAxes,fontsize=FS_AN-3,ha="right",va="top",color=INK,
       bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="#333333",lw=1.0))

for t,a_ in zip(["(a)","(b)","(c)","(d)"],[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]): lab(a_,t)
fig.savefig("results/fig_02_dnr_core.png",dpi=200,bbox_inches="tight",
            pad_inches=0.25,facecolor="white")
print("n",len(u),"rho",round(rho,3),"bp",SEG["breakpoint_DNR"])
print("cond:",[(n,round(x,3),round(y,3)) for n,x,y,_,_ in cond])

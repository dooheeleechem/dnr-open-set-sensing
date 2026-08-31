import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Data: Yao et al. 2023, Chemom. Intell. Lab. Syst. 242, 105003
# Table 3 (Setting 1) and Table 4 (Setting 2). OSR performance (%), openness = 10.56% for ALL entries.
S1 = {
 "CCSA_EVM":     [59.10,75.90,85.18,60.61,86.81,72.56,69.43,50.39,76.58],
 "CCSA_MMF":     [75.95,71.03,80.54,90.47,72.75,63.36,77.69,76.69,60.79],
 "CCSA_MMF_CAC": [86.79,80.67,76.46,84.84,67.12,73.10,75.50,85.99,63.12],
 "CACSA":        [85.69,83.23,79.72,88.94,74.20,74.76,84.44,92.41,66.72],
}
S2 = {
 "CCSA_EVM":     [59.10,60.83,70.55,71.23,84.00,81.55,68.65,76.30,72.11],
 "CCSA_MMF":     [75.95,90.06,84.01,95.81,76.93,87.90,83.02,82.62,62.60],
 "CCSA_MMF_CAC": [86.79,86.75,87.03,97.56,84.93,90.31,88.80,81.81,71.58],
 "CACSA":        [85.69,93.54,80.61,97.07,80.79,90.70,87.65,84.27,70.02],
}
OPENNESS = 10.56
batches = np.arange(2,11)

RED,ORANGE,TEAL,BLUE = "#C94F4A","#E8943A","#4AACB0","#5B8DB8"
BROWN, INK, GREY = "#9C6B4A", "#33373B", "#6E6E6E"
COL = {"CCSA_EVM":BLUE,"CCSA_MMF":TEAL,"CCSA_MMF_CAC":ORANGE,"CACSA":RED}
MRK = {"CCSA_EVM":"o","CCSA_MMF":"s","CCSA_MMF_CAC":"^","CACSA":"D"}

FS_LABEL,FS_AXIS,FS_TICK,FS_LEG,FS_ANN = 28,16,15,15,14
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42})

fig,axes = plt.subplots(1,2,figsize=(13,6.5),dpi=200)
plt.subplots_adjust(wspace=0.42,top=0.80,bottom=0.12,left=0.075,right=0.945)

def style(ax):
    ax.set_facecolor("white")
    ax.grid(True,color="#DDDDDD",ls="--",lw=0.6,zorder=0)
    ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)

def panel(ax,txt):
    ax.text(0.0,1.18,txt,transform=ax.transAxes,fontsize=FS_LABEL,
            fontweight="bold",ha="left",va="bottom",clip_on=False,color=INK)

# ---------------- (A) Setting 1: performance falls, openness does not move
ax=axes[0]; style(ax)
for k,v in S1.items():
    ax.plot(batches,v,marker=MRK[k],ms=9,lw=2.4,color=COL[k],
            label=k.replace("_"," "),zorder=3)
ax.set_xlabel("Target batch (source = batch 1)",fontsize=FS_AXIS)
ax.set_ylabel("Open-set recognition performance (%)",fontsize=FS_AXIS)
ax.set_xticks(batches); ax.set_ylim(40,132)
ax.set_title("Setting 1: time interval grows with batch index",fontsize=FS_AXIS,pad=6)

ax2=ax.twinx()
ax2.plot(batches,[OPENNESS]*9,lw=3.2,color=BROWN,ls=(0,(6,3)),zorder=3)
ax2.set_ylabel("Openness (%)",fontsize=FS_AXIS,color=BROWN)
ax2.set_ylim(0,92); ax2.tick_params(labelsize=FS_TICK,colors=BROWN)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(BROWN)
ax2.text(2.15,OPENNESS+1.6,"openness = 10.56 %, constant across every split",
    fontsize=FS_ANN-1,color=BROWN,va="bottom",ha="left")
ax.legend(fontsize=FS_LEG-3,loc="upper left",frameon=False,ncol=2,
          columnspacing=1.2,handlelength=2.0)

# ---------------- (B) all 72 reported values at one openness
ax=axes[1]; style(ax)
allv=[]; rng=np.random.default_rng(7)
for i,(name,D) in enumerate([("Setting 1",S1),("Setting 2",S2)]):
    for k,v in D.items():
        x=i+1+rng.uniform(-0.16,0.16,len(v))
        ax.scatter(x,v,s=95,color=COL[k],edgecolor="white",lw=1.0,
                   marker=MRK[k],zorder=3,label=k.replace("_"," ") if i==0 else None)
        allv+=v
    m=np.mean([x for k,v in D.items() for x in v])
    ax.hlines(m,i+0.76,i+1.24,color=INK,lw=2.6,zorder=4)
    ax.text(i+1.29,m,f"mean {m:.1f}",fontsize=FS_ANN,va="center",
            ha="left",color=INK)
lo,hi=min(allv),max(allv)
ax.set_xlim(0.50,2.95); ax.set_ylim(40,124)
ax.set_xticks([1,2]); ax.set_xticklabels(["Setting 1\n(batch 1 → K)","Setting 2\n(batch K → K+1)"])
ax.set_ylabel("Open-set recognition performance (%)",fontsize=FS_AXIS)
ax.set_title("Every point below has the same openness",fontsize=FS_AXIS,pad=6)
ax.annotate("",xy=(0.64,lo),xytext=(0.64,hi),
            arrowprops=dict(arrowstyle="<->",color=RED,lw=2.0))
ax.text(0.58,(lo+hi)/2,f"{hi-lo:.1f} points",rotation=90,fontsize=FS_ANN+1,
        color=RED,ha="right",va="center",fontweight="bold")
box=dict(boxstyle="round,pad=0.45",facecolor="white",edgecolor="#333333",lw=1.1)
ax.text(0.5,0.975,
    "72 reported values, openness = 10.56 % for all of them\n"
    f"observed range {lo:.2f} to {hi:.2f} %",
    transform=ax.transAxes,fontsize=FS_ANN-2,ha="center",va="top",bbox=box)
ax.legend(fontsize=FS_LEG-3,loc="lower right",frameon=False,ncol=2,columnspacing=1.0)

panel(axes[0],"(a)"); panel(axes[1],"(b)")  # ACS: lowercase panel labels
fig.savefig("results/fig_01_openness_blind.png",
            dpi=200,bbox_inches="tight",pad_inches=0.25,facecolor="white")
print("saved; range",min(allv),max(allv),"spread",max(allv)-min(allv))

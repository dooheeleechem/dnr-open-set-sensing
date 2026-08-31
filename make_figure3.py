import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np, pandas as pd

d=pd.read_csv("results/raw_results.csv")
d["pair"]=d.src.astype(str)+r"$\rightarrow$"+d.tgt.astype(str)
u=d.drop_duplicates(subset=["src","tgt","unknown"])
gases=["Ethanol","Ethylene","Ammonia","Acetaldehyde","Acetone","Toluene"]
order=[(1,k) for k in range(2,11)]+[(k,k+1) for k in range(2,10)]
labels=[f"{a}"+r"$\rightarrow$"+f"{b}" for a,b in order]
M=np.full((len(order),6),np.nan)
for i,(a,b) in enumerate(order):
    for j,g in enumerate(gases):
        r=u[(u.src==a)&(u.tgt==b)&(u.unknown_name==g)]
        if len(r): M[i,j]=r.DNR_maha.values[0]

RED,TEAL,INK,GREY="#C94F4A","#4AACB0","#33373B","#6E6E6E"
cmap=LinearSegmentedColormap.from_list("dnr",[TEAL,"#EFF4F3","#FBEFE8",RED])
cmap.set_bad("#E8E8E8")
norm=TwoSlopeNorm(vmin=np.log10(0.03),vcenter=0.0,vmax=np.log10(12))

FS_LAB,FS_AX,FS_TK,FS_AN=28,16,15,11
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"]})
fig,ax=plt.subplots(figsize=(20,10),dpi=200)
plt.subplots_adjust(left=0.11,right=0.88,top=0.84,bottom=0.12)
im=ax.imshow(np.log10(M).T,aspect="auto",cmap=cmap,norm=norm)
ax.set_facecolor("white")
ax.set_xticks(range(len(order))); ax.set_xticklabels(labels,fontsize=FS_TK,rotation=45,ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(gases,fontsize=FS_TK)
ax.set_xlabel("Source $\\rightarrow$ target batch",fontsize=FS_AX)
ax.set_ylabel("Withheld (unknown) analyte",fontsize=FS_AX)
ax.set_title("DNR varies by two orders of magnitude across splits that share one openness value",
             fontsize=FS_AX,pad=42)
for i in range(len(order)):
    for j in range(6):
        v=M[i,j]
        if np.isnan(v): ax.text(i,j,"n/a",ha="center",va="center",fontsize=FS_AN-1,color=GREY)
        else:
            ax.text(i,j,f"{v:.2f}",ha="center",va="center",fontsize=FS_AN,
                    color="white" if (v>3 or v<0.12) else INK,
                    fontweight="bold" if v>1 else "normal")
ax.axvline(8.5,color=INK,lw=2.4)
ax.text(4.0,-0.72,"Setting 1  (batch 1 $\\rightarrow$ K)",fontsize=FS_AX-1,ha="center",color=INK)
ax.text(12.5,-0.72,"Setting 2  (batch K $\\rightarrow$ K+1)",fontsize=FS_AX-1,ha="center",color=INK)
for s in ("top","right"): ax.spines[s].set_visible(False)
cb=fig.colorbar(im,ax=ax,pad=0.02,fraction=0.035,
                ticks=[np.log10(x) for x in (0.05,0.2,1,3,10)])
cb.ax.set_yticklabels(["0.05","0.2","1","3","10"],fontsize=FS_TK)
cb.set_label("DNR (log scale)",fontsize=FS_AX)
cb.ax.axhline(0.0,color=INK,lw=2.0)
# Supporting Information Figure S1 in the submitted manuscript
fig.savefig("results/figS1_dnr_landscape.png",dpi=200,bbox_inches="tight",pad_inches=0.25,facecolor="white")
print("cells:",int(np.isfinite(M).sum()),"| >1:",int((M>1).sum()),"| n/a:",int(np.isnan(M).sum()))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
RED,TEAL,INK,GREY="#C94F4A","#4AACB0","#33373B","#8A8A8A"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"]})
fig,axes=plt.subplots(1,2,figsize=(3.25,1.75),dpi=600)
plt.subplots_adjust(left=0.02,right=0.98,top=0.80,bottom=0.13,wspace=0.14)

def panel(ax,drift,title,col,verdict):
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off"); ax.set_facecolor("white")
    ax.add_patch(Circle((2.2,4.0),0.55,facecolor="white",edgecolor=INK,lw=0.9,zorder=3))
    ax.text(2.2,4.0,"K",ha="center",va="center",fontsize=6,color=INK,zorder=4)
    ax.add_patch(Circle((2.2+drift,4.0),0.55,facecolor=col,edgecolor=col,lw=0.9,alpha=0.85,zorder=3))
    ax.text(2.2+drift,4.0,"K'",ha="center",va="center",fontsize=6,color="white",zorder=4)
    ax.add_patch(FancyArrowPatch((2.75,4.0),(1.65+drift,4.0),arrowstyle="-|>",
        mutation_scale=6,lw=1.1,color=col,zorder=2))
    ax.text(2.2+drift/2,4.75,"$d_{drift}$",ha="center",fontsize=6,color=col)
    ax.add_patch(Circle((7.4,1.5),0.55,facecolor="white",edgecolor=GREY,lw=0.9,ls="--",zorder=3))
    ax.text(7.4,1.5,"U",ha="center",va="center",fontsize=6,color=GREY,zorder=4)
    ax.add_patch(FancyArrowPatch((2.6,3.55),(6.95,1.75),arrowstyle="-|>",
        mutation_scale=6,lw=1.1,color=GREY,ls=(0,(3,2)),zorder=2))
    ax.text(4.4,2.05,"$d_{novel}$",ha="center",fontsize=6,color=GREY)
    ax.text(5,5.62,title,ha="center",fontsize=6.6,color=col,fontweight="bold")
    ax.text(5,0.10,verdict,ha="center",fontsize=5.6,color=INK)

panel(axes[0],2.2,"DNR < 1",TEAL,"low drift-novelty confounding")
panel(axes[1],5.0,"DNR > 1",RED,"high drift-novelty confounding")
fig.text(0.5,0.955,"A difficulty axis for open-set chemical sensing",
         ha="center",fontsize=7.4,color=INK,fontweight="bold")
fig.savefig("toc_graphic.png",dpi=600,facecolor="white",
            bbox_inches="tight",pad_inches=0.02)
print("saved 3.25 x 1.75 in @600 dpi")

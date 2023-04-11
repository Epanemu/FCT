import colorsys
import matplotlib
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_style("darkgrid")
matplotlib.rcParams.update({'errorbar.capsize': 2})

from copy import deepcopy

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb[:3])
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(0.7, l * scale_l), s)

def plot_in_axis(ax, vals, name, color, mode="minmax", show_train=True, show_test=True, bar_width=0.002, show_bounds=True, fixed_lims=False):
    # ax.scatter(vals["TrainAcc"], vals["TrainLeafAcc"], label=f"{name} - train", color=color)
    # ax.scatter(vals["TestAcc"], vals["TestLeafAcc"], label=f"{name} - test", marker="v", color=color)

    # ax.scatter([vals["TrainAcc"].mean()], [vals["TrainLeafAcc"].mean()], label=f"{name} - train", markersize=15, color=color)
    # ax.scatter([vals["TestAcc"].mean()], [vals["TestLeafAcc"].mean()], label=f"{name} - test", marker="v", markersize=15, color=color)

    # ax.errorbar([vals["TrainAcc"].mean()], [vals["TrainLeafAcc"].mean()],
    #             xerr=vals["TrainAcc"].std(), yerr=vals["TrainLeafAcc"].std(), capthick=20,
    #             label=f"{name} - train",marker="o", markersize=8, color=color)
    # ax.errorbar([vals["TestAcc"].mean()], [vals["TestLeafAcc"].mean()],
    #             xerr=vals["TestAcc"].std(), yerr=vals["TestLeafAcc"].std(),
    #             label=f"{name} - test", marker="v", markersize=8, color=color)

    if mode == "std":
        if show_train:
            ax.errorbar([vals["TrainAcc"].mean()], [vals["TrainLeafAcc"].mean()],
                        xerr=vals["TrainAcc"].std(), yerr=vals["TrainLeafAcc"].std(), capsize=2,
                        label=f"{name} - train", marker="o", markersize=8,
                        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
        if show_test:
            ax.errorbar([vals["TestAcc"].mean()], [vals["TestLeafAcc"].mean()],
                        xerr=vals["TestAcc"].std(), yerr=vals["TestLeafAcc"].std(), capsize=2,
                        label=f"{name} - test", marker="v", markersize=8,
                        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
        bar_y = vals["TrainLeafAcc"].mean()
        bar_x = vals["TrainAcc"].mean() - bar_width/2
        u_bound = vals["ObjBound"].mean()
    elif mode == "minmax":
        if show_train:
            x = vals["TrainAcc"].median()
            y = vals["TrainLeafAcc"].median()
            max_x, min_x = vals["TrainAcc"].max(), vals["TrainAcc"].min()
            max_y, min_y = vals["TrainLeafAcc"].max(), vals["TrainLeafAcc"].min()
            ax.errorbar([x], [y], xerr=[[x-min_x], [max_x-x]], yerr=[[y-min_y],[max_y-y]],
                        capsize=2, label=f"{name} - train", marker="o", markersize=8,
                        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
        if show_test:
            x = vals["TestAcc"].median()
            y = vals["TestLeafAcc"].median()
            max_x, min_x = vals["TestAcc"].max(), vals["TestAcc"].min()
            max_y, min_y = vals["TestLeafAcc"].max(), vals["TestLeafAcc"].min()
            ax.errorbar([x], [y], xerr=[[x-min_x], [max_x-x]], yerr=[[y-min_y],[max_y-y]],
                        capsize=2, label=f"{name} - test", marker="v", markersize=8,
                        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
        bar_y = vals["TrainLeafAcc"].max()
        bar_x = vals["TrainAcc"].median() - bar_width/2
        u_bound = vals["ObjBound"].median()

    if show_train and show_bounds:
        ax.add_patch(Rectangle((bar_x, bar_y), bar_width, u_bound-bar_y,
             edgecolor = scale_lightness(color, 1.5),
             linestyle = 'solid',
             hatch='///',
             # facecolor = scale_lightness(color, 2),
             fill=False,
             lw=1))

    if fixed_lims:
        ax.set_xlim((0.45, 1.05))
        ax.set_ylim((-0.05, 1.05))

def plot_xgb_bar(ax, vals, show_train=True, show_test=True, mode="std"):
    if mode == "std":
        train = vals["TrainAcc"].mean()
        test = vals["TestAcc"].mean()
    elif mode == "minmax":
        train = vals["TrainAcc"].median()
        test = vals["TestAcc"].median()

    y = ax.get_ylim()
    if show_train:
        ax.plot([train, train], y, linestyle='dotted', color="k")
    if show_test:
        ax.plot([test, test], y, linestyle='dashed', color="k")
    ax.set_ylim(y)


def add_labels(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Accruacy of model")
    ax.set_ylabel("Lowest (soft) accuracy in a leaf")


def add_legend(ax, labels, colors, mode="std"):
    err_label = "Standard deviation" if mode=="std" else "Min - Max range"
    acc_measure = "Mean" if mode=="std" else "Median"
    gap_label = "Mean objective gap" if mode=="std" else "Best objective gap"
    handles, _ = ax.get_legend_handles_labels()
    errorbar_handle = ax.errorbar([], [], xerr=1, yerr=1, capsize=2, linestyle="",
                label=err_label, marker="", color="k")

    legend_elements = [
        Line2D([0], [0], marker='o', color=(1,1,1,0), label=f'{acc_measure} Train accuracy',
                          markeredgecolor="k", markerfacecolor=None, markersize=10),
        Line2D([0], [0], marker='v', color=(1,1,1,0), label=f'{acc_measure} Test accuracy',
                          markeredgecolor="k", markerfacecolor=None, markersize=10),
        errorbar_handle,
        Patch(hatch='///', edgecolor="k", facecolor=(1,1,1,0), label=gap_label),
        Line2D([0], [0], linestyle='dotted', color="k", label=f'XGB Train accuracy'),
        Line2D([0], [0], linestyle='dashed', color="k", label=f'XGB Test accuracy'),
        Patch(visible=False),  # spacer
    ] + [Patch(facecolor=c, edgecolor=(1,1,1,0), label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements)

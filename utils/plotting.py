import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_style("darkgrid")
matplotlib.rcParams.update({'errorbar.capsize': 2})


def set_difficulty_legend(ax, labels, colors, anchor):
    errorbar_handle = ax.errorbar([], [], xerr=1, yerr=1, capsize=2, uplims=True, lolims=True, xuplims=True, xlolims=True, linestyle="",
                label="Standard deviation", marker="", color="k")
    errorbar_dotted = ax.errorbar([], [], xerr=1, yerr=1, capsize=2, uplims=True, lolims=True, xuplims=True, xlolims=True, linestyle="",
                label="Standard deviation", marker="", color="k")
    for b in errorbar_dotted[2]:
        b.set_linestyle("dotted")

    legend_elements = []
    legend_elements.append(Patch(visible=False, label='Hybrid trees'))
    legend_elements.append(
        Line2D([0], [0], marker="D", color=(1,1,1,0), label=f'Mean Proposed',
                        markeredgecolor="k", markerfacecolor=None, markersize=10))
    legend_elements.append(
        Line2D([0], [0], marker="x", color=(1,1,1,0), label=f'Mean CART',
                        markeredgecolor="k", markerfacecolor=None, markersize=10))
    legend_elements.append(errorbar_handle)
    # legend_elements.append(Line2D([0], [0], linestyle='dotted', color="k", label=f'non-extended model'))
    legend_elements.append(Patch(visible=False)),  # spacer

    legend_elements.append(Line2D([0], [0], linestyle='dashed', color="k", label=f'Mean XGBoost'))
    legend_elements.append(Patch(visible=False)),  # spacer

    legend_elements += [Patch(facecolor=c, edgecolor=(1,1,1,0), label=l) for c, l in zip(colors, labels)]

    # ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(1.1, -0.15), ncol=len(legend_elements))
    leg = ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=anchor)
    # leg = ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=anchor, ncol=2)
    style_legend_titles_by_setting_position(leg, bold=True)


def set_legend(ax, labels, colors, anchor, ncol=1):
    errorbar_handle = ax.errorbar([], [], xerr=1, yerr=1, capsize=2, uplims=True, lolims=True, xuplims=True, xlolims=True, linestyle="",
                label="Standard deviation", marker="", color="k")
    errorbar_dotted = ax.errorbar([], [], xerr=1, yerr=1, capsize=2, uplims=True, lolims=True, xuplims=True, xlolims=True, linestyle="",
                label="Standard deviation", marker="", color="k")
    for b in errorbar_dotted[2]:
        b.set_linestyle("dotted")

    legend_elements = []
    legend_elements.append(Patch(visible=False, label='Hybrid tree'))
    legend_elements.append(
        Line2D([0], [0], marker="D", color=(1,1,1,0), label=f'Mean OOS Accuracy',
                        markeredgecolor="k", markerfacecolor=None, markersize=10))
    legend_elements.append(errorbar_handle)
    legend_elements.append(Patch(visible=False, label='Low-depth tree'))
    legend_elements.append(
        Line2D([0], [0], marker="s", color=(1,1,1,0), label=f'Mean OOS Accuracy',
                        markeredgecolor="k", markerfacecolor=None, markersize=10))
    legend_elements.append(errorbar_dotted)
    # legend_elements.append(Line2D([0], [0], linestyle='dotted', color="k", label=f'non-extended model'))
    legend_elements.append(Patch(visible=False)),  # spacer

    legend_elements.append(Line2D([0], [0], linestyle='dashed', color="k", label=f'Mean XGBoost'))

    legend_elements += [Patch(facecolor=c, edgecolor=(1,1,1,0), label=l) for c, l in zip(colors, labels)]

    # ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(1.1, -0.15), ncol=len(legend_elements))
    leg = ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=anchor, ncol=ncol)
    style_legend_titles_by_setting_position(leg, bold=True)

# source: https://stackoverflow.com/questions/24787041/multiple-titles-in-legend-in-matplotlib
def style_legend_titles_by_setting_position(leg, bold=False):
    """ Style legend "titles"

    A legend entry can be marked as a title by setting visible=False. Titles
    get left-aligned and optionally bolded.
    """
    # matplotlib.offsetbox.HPacker unconditionally adds a pixel of padding
    # around each child.
    hpacker_padding = 2

    for handle, label in zip(leg.legendHandles, leg.texts):
        if not handle.get_visible():
            # See matplotlib.legend.Legend._init_legend_box()
            widths = [leg.handlelength, leg.handletextpad]
            offset_points = sum(leg._fontsize * w for w in widths)
            offset_pixels = leg.figure.canvas.get_renderer().points_to_pixels(offset_points) + hpacker_padding
            label.set_position((-offset_pixels, 0))
            if bold:
                label.set_fontweight('bold')

def set_labels(ax, title, xlabel="Accruacy of model", ylabel="Leaf accuracy (Minimal accuracy in a single leaf)"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def show_xgb(ax, val, color="k"):
    ylims = ax.get_ylim()
    ax.plot([val, val], ylims, linestyle='dashed', color=color)
    ax.set_ylim(ylims)

def plot_low_depth(ax, vals, leaf_vals, color):
    _, _, bars = ax.errorbar([vals.mean()], [leaf_vals.mean()],
        xerr=vals.std(), yerr=leaf_vals.std(), capsize=2, marker="s", markersize=8,
        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
    for b in bars:
        b.set_linestyle("dotted")

def plot_hybrid(ax,vals, leaf_vals, color):
    ax.errorbar([vals.mean()], [leaf_vals.mean()],
        xerr=vals.std(), yerr=leaf_vals.std(), capsize=4, marker="D", markersize=8,
        markerfacecolor=(1,1,1,0), markeredgecolor=color, color=color)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op

OUT_DIR = op.realpath(op.join(__file__ , '../../results'))

def get_fname(file):
    return op.splitext(op.basename(file))[0]

def outpath_prefix_for_script(script_file):
    return f"{OUT_DIR}/{get_fname(script_file)}"

A4_PAPER_CONTENT_WIDTH = 7.1
DEFAULT_HEIGHT = 2.16 # default figure height, in inches

COLORS_BLUE = [ # Different shades of Blue, from Refactoring UI Color Palette #8
    "#002159", "#01337D", "#03449E", "#0552B5", "#0967D2",
    "#2186EB", "#47A3F3", "#7CC4FA", "#BAE3FF", "#E6F6FF"]
COLORS_RED = [ # Different shades of Red, from Refactoring UI Color Palette #1
    "#610404", "#780A0A", "#911111", "#A61B1B", "#BA2525",
    "#D64545", "#E66A6A", "#F29B9B", "#FACDCD", "#FFEEEE"]
COLORS_GRAY = [ # Different shades of Grey, from Refactoring UI Colors
    "#222222", "#3B3B3B", "#515151", "#626262", "#7E7E7E",
    "#9E9E9E", "#B1B1B1", "#CFCFCF", "#E1E1E1", "#F7F7F7"]
COLORS_COOL_GRAY = [ # Different shades of Cool Grey, from Refactoring UI Colors
    "#1F2933", "#323F4B", "#3E4C59", "#52606D", "#616E7C",
    "#7B8794", "#9AA5B1", "#CBD2D9", "#E4E7EB", "#F5F7FA"]

GRAY_COLOR = "#666666"
BLACK_COLOR = "#000000"
OBSERVATION_COLOR = COLORS_RED[4]
# HIDDEN_STATE_COLOR = COLORS_GRAY[4]
HIDDEN_STATE_COLOR = "#4CB24C"
# POST_MEAN_COLOR = "#FFFF0099"
POST_MEAN_COLOR = "#002BFF"
POST_MODE_COLOR = ("#00AAAA"
    # + "CC" # 80% alpha
    ) 
# POST_DIST_CMAP = "cividis"
POST_DIST_CMAP = "hot"

def setup_mpl_style(fontsize=8):
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['figure.titlesize'] = fontsize
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize-1
    mpl.rcParams['axes.labelpad'] = 4.0
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams['figure.constrained_layout.use'] = True

LABELS_DICT = {
    "obs": "Observations",
    "hid": "True hidden state",
    "post_dist": "Posterior probability",
    "post_mean": "Posterior mean",
    "post_mode": "Posterior mode",
}

def plot_sequence(t, obs, hid, hidden_state_space, post_dists, post_means,
    figsize=(A4_PAPER_CONTENT_WIDTH,
             DEFAULT_HEIGHT * 1.5),
    xlabel="Time step",
    labels_dict=LABELS_DICT):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    plot_sequence_on_ax(ax, t, obs, hid, hidden_state_space, post_dists, post_means,
    xlabel=xlabel, labels_dict=labels_dict)
    return fig

def plot_sequence_on_ax(ax, t, obs, hid, hidden_state_space, post_dists,
    post_means=None,
    post_modes=None,
    xlabel="Time step",
    labels_dict=LABELS_DICT,
    yextent=None,
    labelcolor=COLORS_GRAY[8],
    cmap=None, markersize=1., lw=1.,
    labelspacing=0.5,
    cbaraspect=60,
    cbarpad=0.025,
    cticklabelsize=6,
    vmin=0):
    ax.set_xlabel(xlabel)
    # Plot posterior probability density
    if post_dists is not None:
        if yextent is None:
            step_fst = np.abs(hidden_state_space.values[1] - hidden_state_space.values[0])
            step_last = np.abs(hidden_state_space.values[-1] - hidden_state_space.values[-2])
            yextent = hidden_state_space.values[0]-step_fst/2, hidden_state_space.values[-1]+step_last/2
        im = ax.imshow(post_dists.T,
            aspect="auto",
            origin="lower",
            extent=(t[0]-0.5, t[-1]+0.5, yextent[0], yextent[1]),
            cmap=(POST_DIST_CMAP if cmap is None else cmap),
            vmin=vmin)
    # Plot true hidden state
    if hid is not None:
        ax.plot(t, hid, '--', lw=lw, color=HIDDEN_STATE_COLOR,
            label=labels_dict["hid"])
    # Plot observations
    if obs is not None:
        ax.plot(t, obs, 
            '.', ms=markersize, color=OBSERVATION_COLOR,
            label=labels_dict["obs"], zorder=10)
    # Plot posterior mean
    if post_means is not None:
        ax.plot(t, post_means, '-', lw=lw, color=POST_MEAN_COLOR,
            label=labels_dict["post_mean"])
    # Plot posterior mode
    if post_modes is not None:
        ax.plot(t, post_modes, '-', lw=lw, color=POST_MODE_COLOR,
            label=labels_dict["post_mode"])
    # Legend
    if len(ax.lines) > 0:
        ax.legend(labelcolor=labelcolor, labelspacing=labelspacing)
    # Color bar
    if post_dists is not None:
        cbar = ax.figure.colorbar(im, ax=ax, location='right', aspect=cbaraspect,
            shrink=0.9, pad=cbarpad)
        cbar.ax.tick_params(labelsize=cticklabelsize)
        cbar.set_label(LABELS_DICT["post_dist"], rotation=-90, labelpad=10)

def save_figure(fig, figpath, verbose=True, do_create_dir_if_needed=True):
    if do_create_dir_if_needed:
        d = op.dirname(figpath)
        create_dir_if_needed(d)
    fig.savefig(figpath)
    plt.close(fig)
    if verbose:
        print(f"Figure saved at {figpath}")

def create_dir_if_needed(d):
    if not op.exists(d):
        os.makedirs(d, exist_ok=True)

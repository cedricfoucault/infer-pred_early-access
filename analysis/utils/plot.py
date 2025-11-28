# from . import data
import matplotlib as mpl
import matplotlib.pyplot as plt

CM = 1/2.54 # cm in inches

LETTER_WIDTH = 21.6*CM
LETTER_HEIGHT = 27.9*CM
PAPER_MARGINS = 1.8*CM
PAPER_CONTENT_WIDTH = (LETTER_WIDTH - 2*PAPER_MARGINS) # == 18*CM
PAPER_CONTENT_HEIGHT = (LETTER_HEIGHT - 2*PAPER_MARGINS) # == 14.3*CM

DEFAULT_HEIGHT = 2.16 

COLORS_BLUE = [ # Different shades of Blue, from Refactoring UI Color Palette #8
    "#002159", "#01337D", "#03449E", "#0552B5", "#0967D2",
    "#2186EB", "#47A3F3", "#7CC4FA", "#BAE3FF", "#E6F6FF"]
COLORS_RED = [ # Different shades of Red, from Refactoring UI Color Palette #1
    "#610404", "#780A0A", "#911111", "#A61B1B", "#BA2525",
    "#D64545", "#E66A6A", "#F29B9B", "#FACDCD", "#FFEEEE"]
COLORS_ORANGE = [ # Orange (Vivid) - Refactoctoring UI Color Swatches
    "#841003", "#AD1D07", "#C52707", "#DE3A11", "#F35627",
    "#F9703E", "#FF9466", "#FFB088", "#FFD0B5", "#FFE8D9"]
COLORS_GRAY = [ # Different shades of Grey, from Refactoring UI Colors
    "#222222", "#3B3B3B", "#515151", "#626262", "#7E7E7E",
    "#9E9E9E", "#B1B1B1", "#CFCFCF", "#E1E1E1", "#F7F7F7"]
COLORS_COOL_GRAY = [ # Different shades of Cool Grey, from Refactoring UI Colors
    "#1F2933", "#323F4B", "#3E4C59", "#52606D", "#616E7C",
    "#7B8794", "#9AA5B1", "#CBD2D9", "#E4E7EB", "#F5F7FA"]

GRAY_COLOR = "#666666"
BLACK_COLOR = "#000000"
OBS_COLOR = COLORS_RED[6]
HMEAN_COLOR = "#8259A3"
HSD_COLOR = "#8259A3"
SUBJECT_BET_CENTER_COLOR = "#EFCA8F"
SUBJECT_BET_EDGES_COLOR = "#EFCA8F"
SUBJECT_BET_FILL_COLOR = SUBJECT_BET_EDGES_COLOR + "60" # 40% alpha
# BET_WIDTH_LEVEL_LABELS = [f"{sml} ({val}°)" for sml, val in zip(
#         ["Small", "Medium", "Large"], data.COINS_MEG_BET_WIDTH_LEVELS)]
# BET_WIDTH_LEVEL_COLORS = [mpl.colormaps["Accent"](i)
#     for i in range(data.COINS_MEG_N_BET_WIDTH_LEVELS)]

# STABLE_COLOR = mpl.colormaps["Paired"](3)
# VOLATILE_COLOR = mpl.colormaps["Paired"](5)
# STOCH_LEVEL_COLORS = [mpl.colormaps["plasma"]((i+1)/4)
#     for i in range(data.COINS_MEG_N_STOCHASTICITY_LEVELS)]

ERROR_SHADING_ALPHA = 0.2

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

def stat_label(p, criteria=(0.05, 0.01, 0.001)):
    if p <= criteria[2]:
        return "***"
    elif p <= criteria[1]:
        return "**"
    elif p <= criteria[0]:
        return "*"
    else:
        return "ns"

def stat_t_text(ttest):
    return f"t({ttest.df})={ttest.statistic:.1f}, p={ttest.pvalue:.0E}"

def get_grayscale_color(color):
    rgb = mpl.colors.to_rgb(color)
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (gray, gray, gray)

COLORDICT = dict(
    obs=OBS_COLOR,
    betloc=SUBJECT_BET_CENTER_COLOR,
    hmean=HMEAN_COLOR,
    hsd=HSD_COLOR)

FILLDICT = dict(
    betwid=True,
    hsd=True)

EDGECOLORDICT = dict(
    betwid=SUBJECT_BET_EDGES_COLOR,
    hsd=HSD_COLOR)

FACECOLORDICT = dict(
    betwid=SUBJECT_BET_FILL_COLOR,
    hsd=HSD_COLOR + "60")

LABELDICT = dict(
    obs="Beam locations",
    betloc="Paddle location",
    betwid="Paddle width",
    hmean="Generative mean",
    hsd="Generative s.d.")

LINESTYLEDICT = dict(
    obs='',
    betloc="-",
    betwid='-', 
    hmean="--",
    hsd="--")

LINEWIDTHDICT = dict(
    betloc=1.,
    hmean=1.,)

MARKERDICT = dict(
    obs='.')

MARKERSIZEDICT = dict(
    obs=1.)

ZORDERDICT = dict(
    obs=0,
    betloc=1,
    betwid=2,
    hmean=-2,
    hsd=-1)

def plot_seq_common(ax, t, items):
    """
    Helper function to plot the common elements of the sequence plots.
    """
    for key, vals in items:
        y = vals['y']
        vals.setdefault('fill', FILLDICT.get(key, False))
        params = vals.get('params', {}).copy()
        params.setdefault('label', LABELDICT.get(key, None))
        params.setdefault('color', COLORDICT.get(key, None))
        params.setdefault('linestyle', LINESTYLEDICT.get(key, None))
        params.setdefault('linewidth', LINEWIDTHDICT.get(key, None))
        params.setdefault('zorder', ZORDERDICT.get(key, None))
        if vals['fill']:
            params.setdefault('facecolor', FACECOLORDICT.get(key, None))
            params.setdefault('edgecolor', EDGECOLORDICT.get(key, None))
            ax.fill_between(t, y[0], y[1], **params)
        else:
            params.setdefault('marker', MARKERDICT.get(key, None))
            params.setdefault('markersize', MARKERSIZEDICT.get(key, None))
            ax.plot(t, y, **params)

def add_text(ax, text, fontsize=None, autoshrink=False, autostretch=True, **kwargs):
    if fontsize is None:
        fontsize = mpl.rcParams['font.size']
    x = -0.15
    stretch = None
    if len(text) > 20:
        x = -0.2
        if autoshrink:
            fontsize -= 1
        elif autostretch:
            stretch = 'extra-condensed'
    if len(text) > 40:
        x = -0.25
        if autoshrink:
            fontsize -= 1
        elif autostretch:
            stretch = 'ultra-condensed'
    ax.text(x, 1.05, text,
        fontsize=fontsize, stretch=stretch,
        va="bottom", ha="left", transform=ax.transAxes,
        **kwargs)

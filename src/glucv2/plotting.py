import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Validated categorical order (dataviz reference palette, light surface) and a blue<->red diverging map
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e4e3df'
DIVERGING = LinearSegmentedColormap.from_list('bluegrayred', ['#184f95', '#6da7ec', '#f0efec', '#ec8a89', '#a72b2b'])
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f0efec', '#9ec5f4', '#3987e5', '#1c5cab', '#0d366b'])


def style():
    """Publication style: editable SVG text, embedded TrueType in PDF, seaborn 'ticks' look."""
    try:
        import seaborn as sns
        sns.set_theme(style='ticks', context='paper')
    except ImportError:
        pass
    mpl.rcParams.update({
        'svg.fonttype': 'none',            # text stays <text> in SVG (editable in Illustrator/Inkscape)
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5, 'legend.frameon': False, 'axes.titleweight': 'bold',
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def save_multi(fig, stem, formats=('pdf', 'svg', 'png'), dpi=600, demo=False):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    if demo:
        fig.text(0.5, 0.5, 'SYNTHETIC DEMO — NOT FOR PUBLICATION', ha='center', va='center', rotation=30, fontsize=26, alpha=.12)
    for ext in formats:
        fig.savefig(stem.with_suffix('.' + ext), bbox_inches='tight', dpi=dpi if ext == 'png' else None)
    plt.close(fig)


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontweight='bold', fontsize=10, va='top', ha='left', color=INK)

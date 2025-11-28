import matplotlib.pyplot as plt
import os
import os.path as op
import pandas as pd

DIR_REL = "../../../results/"
DIR = op.realpath(op.join(__file__ , DIR_REL))

def get_path(dir, basename, ext="png"):
    fname = f"{basename}.{ext}"
    return op.join(dir, fname)

def get_fname(file):
    return op.splitext(op.basename(file))[0]

def get_sub_dir(subid, create_if_needed=True):
    sub_dir = op.join(DIR, f"sub-{subid:02d}")
    if create_if_needed:
        create_dir_if_needed(sub_dir)
    return sub_dir

def create_dir_if_needed(d):
    if not op.exists(d):
        os.makedirs(d, exist_ok=True)

def save_figure(fig, figpath, verbose=True, **kwargs):
    fig.savefig(figpath, **kwargs)
    plt.close(fig)
    if verbose:
        print(f"Figure saved at {figpath}")

def save_stats(stat_data_pd, fpath, verbose=True, **kwargs):
    stat_data_pd.to_csv(fpath, **kwargs)
    if verbose:
        print(f"Stats saved at {fpath}")

def save_text(text, fpath, verbose=True):
    with open(fpath, "w+") as f:
        print(text, file=f)
    if verbose:
        print(f"Text saved at {fpath}")

def name_with_params(prefix, keys, vals):
    name = prefix
    for key, val in zip(keys, vals):
        if ((val is not None)
            and (val is not False)):
            if type(val) == bool:
                name += f"_{key}"
            else:
                name += f"_{key}-{val}"
    return name

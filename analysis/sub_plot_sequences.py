"""
Plot subject sequences
"""

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import sub_data
import utils

def main(args):
    dataset = args.dataset
    sub_list_path, task_config_path = utils.datasets.get_dataset_paths(dataset)
    data_files = sub_data.get_subject_files(sub_list_path)

    task_config = utils.config.load_config_at_path_with_default_task_config(task_config_path)
    config = utils.config.create_joint_config(task_config)

    output_dir = op.join(utils.output.DIR, dataset, "examples", "sequence")
    utils.output.create_dir_if_needed(output_dir)

    # Concatenate individual subjects' data into a single dataframe
    sub_dfs = []
    for i_file, data_file in enumerate(data_files):
        data = sub_data.load_subject_data(data_file)
        df = sub_data.make_subject_dataframe(data, config)
        sub_dfs += [df]
    df = pd.concat(sub_dfs)

    fontsize = 10
    labelspacing = 0.25
    markersize = 4
    utils.plot.setup_mpl_style(fontsize=fontsize)

    subs = df[sub_data.SUB_KEY].unique()
    blocks = df["blockIdx"].unique()
    for sub, block in itertools.product(subs, blocks):
        if args.sub is not None and sub != args.sub:
            continue
        if args.block is not None and block != args.block:
            continue
        df_seq = df[(df[sub_data.SUB_KEY] == sub) & (df["blockIdx"] == block)]
        if df_seq.empty:
            continue

        title = f"Subject {sub}, Block {block}"
        fig = plot_seq(df_seq, title=title,
            labelspacing=labelspacing,
            markersize=markersize)
        ext = "png" if args.ext is None else args.ext
        name = f"seq_sub-{sub}_block-{block}"
        path = utils.output.get_path(output_dir, name, ext)
        utils.output.save_figure(fig, path)

def plot_seq(data,
    title=None,
    figsize=(utils.plot.PAPER_CONTENT_WIDTH,
             utils.plot.PAPER_CONTENT_WIDTH * 1/2),
    ax=None,
    paramsdict={"hsd": {"label": "Generative variance (±2sd)"}},
    labelspacing=0.5,
    markersize=1.5):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    xlabel = "Trial"
    ylabel = "Angle (°)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    obs = data["obs"].to_numpy()
    hmean = data["hmean"].to_numpy()
    hsd = data["hsd"].to_numpy()
    betloc = data["betloc"].to_numpy()
    betwid = data["betwid"].to_numpy()
    nobs = obs.shape[0]
    hmean = np.unwrap(hmean, period=360)
    betloc = utils.calc.align_angle_series_to(betloc, hmean)
    obs = utils.calc.align_angle_series_to(obs, hmean)
    t = np.arange(nobs)+1

    # Plot observations
    params_obs = paramsdict.get("obs", {})
    params_obs["markersize"] = markersize
    utils.plot.plot_seq_common(ax, t, [
        ("obs", dict(y=obs, params=params_obs)),])
    
    # Plot hidden state (generative mean and s.d.)
    utils.plot.plot_seq_common(ax, t, [
        ("hmean", dict(y=hmean)),
        ("hsd", dict(y=(hmean+2*hsd, hmean-2*hsd),
            params=paramsdict.get("hsd", {})))])
    # Plot subject's paddle location and width
    utils.plot.plot_seq_common(ax, t, [
            ('betloc', dict(y=betloc,
                           params=paramsdict.get("betloc", {}))),
            ('betwid', dict(y=(betloc + betwid/2,
                              betloc - betwid/2),
                           params=paramsdict.get("betwid", {})))])

    ax.legend(labelspacing=labelspacing)

    if title:
        ax.set_title(title)

    return fig

if __name__ == '__main__':
    # Examples used in the paper:
    # - change-point-env: -sub 5c4f5967aac8be0001716a65 -block 2
    # - random-walk-env: -sub 6575c7da000442e5a54f5e24 -block 14
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["change-point-env", "random-walk-env"],
        help="Which experimental dataset / environment to analyze.")
    parser.add_argument("-ext", default="pdf")
    parser.add_argument("-sub", default=None)
    parser.add_argument("-block", default=None, type=int)
    args = parser.parse_args()
    main(args)



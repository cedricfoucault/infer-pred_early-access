"""
Run the agent model on the sequences generated with the task config
and plot the behavior of the model

Note: These plots are outdated 
"""

# Add the framework directory to sys.path so the bayeslearner package can be imported
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DIR = REPO_ROOT / "modeling_framework"
import sys
sys.path.append(str(FRAMEWORK_DIR))

import argparse
import bayeslearner
import sim_data
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import infer_pred_model
import utils

#
# Plot function
#
def plot_block_seq(obs, hmean, hsd, betloc, betwid, optout=None, post_mean_hmean=None,
    post_mode_hmean=None, title=None,
    figsize=(utils.plot.PAPER_CONTENT_WIDTH,
             utils.plot.PAPER_CONTENT_WIDTH/2),
    ax=None, paramsdict={},
    betwid_levels=[60, 120], hsd_levels=[10, 30]):
    hmean = np.unwrap(hmean, period=360)
    betloc = utils.calc.align_angle_series_to(betloc, hmean)
    obs = utils.calc.align_angle_series_to(obs, hmean) # needed?
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    xlabel = "Observation number"
    ylabel = "Angle (°)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    nobs = obs.shape[0]
    t = np.arange(nobs)+1
    # Plot observations
    utils.plot.plot_seq_common(ax, t, [
        ("obs", dict(y=obs, params=paramsdict.get("obs", {}))),])
    # utils.plot.plot_seq_common(ax, t, [
    #     ("obs", dict(y=obs, params=paramsdict.get("obs", {}))),
    #     ("betloc", dict(y=betloc, params=paramsdict.get("betloc", {}))),
    #     ("betwid", dict(y=(betloc+betwid, betloc-betwid),
    #                    params=paramsdict.get("betwid", {})))
    #     ,])
    # Plot bet (location and edges)
    did_precise_bet = np.isclose(betwid, betwid_levels[0])
    did_imprecise_bet = np.isclose(betwid, betwid_levels[1])
    if optout:
        did_precise_bet &= ~optout
        did_imprecise_bet &= ~optout
    precisecolor = "#023497"
    imprecisecolor = "#FCBA03"
    colors = [precisecolor, imprecisecolor]
    labels = ["Precise bet", "Imprecise bet"]
    has_labeled = [False, False]
    for i in range(nobs):
        if optout and optout[i]:
            continue
        for i_level in range(2):
            if np.isclose(betwid[i], betwid_levels[i_level]):
                color = colors[i_level]
                if has_labeled[i_level]:
                    label = None
                else:
                    label = labels[i_level]
                    has_labeled[i_level] = True
        x = np.array([t[i]+0.5, t[i]+1.5])
        ymid = np.array([betloc[i], betloc[i]])
        ywid = np.array([betwid[i], betwid[i]])
        locparams = paramsdict.get("betloc", {}).copy()
        locparams["label"] = label
        locparams["color"] = color
        widparams = paramsdict.get("betwid", {}).copy()
        widparams["label"] = None
        widparams["color"] = color
        widparams["edgecolor"] = color
        widparams["facecolor"] = color + "60"
        utils.plot.plot_seq_common(ax, x, [
            ("betloc", dict(y=ymid,
                           params=locparams)),
            ("betwid", dict(y=(ymid + ywid/2,
                              ymid - ywid/2),
                           params=widparams))])
    # Plot posterior stats if provided
    if post_mean_hmean is not None:
        post_mean_hmean = utils.calc.align_angle_series_to(post_mean_hmean, hmean)
        ax.plot(t, post_mean_hmean,
                '-', lw=1.,
                color=bayeslearner.plotutils.POST_MEAN_COLOR,
                label="Posterior mean")
    if post_mode_hmean is not None:
        post_mode_hmean = utils.calc.align_angle_series_to(post_mode_hmean, hmean)
        ax.plot(t, post_mode_hmean, 
                '-', lw=1.,
                color=bayeslearner.plotutils.POST_MODE_COLOR,
                label="Posterior mode")
    # Plot hidden state (hidden mean and s.d.)
    if hsd_levels is not None:
        is_lowsd = np.isclose(hsd, hsd_levels[0])
        is_highsd = np.isclose(hsd, hsd_levels[1])
        for (mask, hmeancolor, hsdcolor) in [
            (is_lowsd, utils.plot.COLORS_GRAY[2], utils.plot.COLORS_GRAY[4]),
            (is_highsd, utils.plot.COLORS_GRAY[5], utils.plot.COLORS_GRAY[7])]:
            t_masked = np.where(mask, t, np.nan)
            hmean_masked = np.where(mask, hmean, np.nan)
            hsd_masked = np.where(mask, hsd, np.nan)
            hmean_params = paramsdict.get("hmean", {}).copy()
            hmean_params["color"] = hmeancolor
            hsd_params = paramsdict.get("hsd", {}).copy()
            hsd_params["color"] = hsdcolor
            hsd_params["edgecolor"] = hsdcolor
            hsd_params["facecolor"] = hsdcolor + "60"
            utils.plot.plot_seq_common(ax, t, [
                ("hmean", dict(y=hmean_masked, params=hmean_params)),
                ("hsd", dict(y=(hmean_masked+hsd_masked, hmean_masked-hsd_masked), params=hsd_params))])
    else:
        utils.plot.plot_seq_common(ax, t, [
                ("hmean", dict(y=hmean)),
                ("hsd", dict(y=(hmean+hsd, hmean-hsd)))])
    ax.legend()
    if title:
        ax.set_title(title)
    return fig

def main(args):
    output_dir = op.join(utils.output.DIR, "sequences")
    utils.output.create_dir_if_needed(output_dir)
    utils.plot.setup_mpl_style()
    #
    # Run the model on each block, make several plots of the block sequence showing
    # different components of the model (inference, behavioral response), and
    # compute various performance measures at the block level
    #
    task_config = utils.config.load_config_at_path_with_default_task_config(args.task_config_path)
    model_config = utils.config.load_config_at_path_with_default_model_config(args.model_config_path)
    config = utils.config.create_joint_config(task_config, model_config)
    model = infer_pred_model.get_model(config)
    model_name = model.name
    do_post_dist = "bayes" in config["model"]["type"]
    nobs = task_config["nobs"]
    betwid_levels = task_config["responses"]["widths"]
    hsd_levels = task_config["state_space"]["hsd"].get("values", None)
    true_gen_model = infer_pred_model.InferPredGenerativeModel(task_config)
    for seed in args.seeds:
        data = sim_data.generate_experiment(true_gen_model,
            1, 1, nobs, seed=seed)
        obs = data["obs"].to_numpy()

        if do_post_dist:
            post_dist, bet = model.compute_post_dist_and_bet_given_obs(
                obs, post_dist_keys=['mean', 'mode', 'margprob'])
        else:
            bet = model.compute_bet_given_obs(obs)

        #
        # Plot the sequence and the behavior of different components of the model
        # over the sequence.
        #

        # Plot model's responses
        betloc = bet[:, 0]
        betwid = bet[:, 1]
        post_mean_hmean = post_dist['mean'][:, model.hmean_dim] if do_post_dist else None
        post_mode_hmean = post_dist['mode'][:, model.hmean_dim] if do_post_dist else None
        title = f"Model: {model_name} (seq. seed {seed})"
        hmean = data["hmean"].to_numpy()
        hsd = data["hsd"].to_numpy()

        # for with_post_stats in [False, True]:
        for with_post_stats in [False]:
            fig = plot_block_seq(obs, hmean, hsd, betloc, betwid,
                title=title,
                post_mean_hmean=(post_mean_hmean if with_post_stats else None),
                post_mode_hmean=(post_mode_hmean if with_post_stats else None),
                betwid_levels=betwid_levels,
                hsd_levels=hsd_levels,
                )
            ax = fig.gca()
            prefix = utils.output.get_fname(__file__)
            prefix += "_sequence"
            figname = utils.output.name_with_params(prefix,
                ["config", "seed"], [config["id"], f"{seed:02d}"])
            ext = "png"
            figpath = utils.output.get_path(output_dir, figname, ext)
            utils.output.save_figure(fig, figpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-tcp", "--task_config_path", default=None)
    parser.add_argument("-mcp", "--model_config_path", default=None)
    parser.add_argument("-s", "--seeds", nargs="+", default=[0], type=(int))
    args = parser.parse_args()
    main(args)



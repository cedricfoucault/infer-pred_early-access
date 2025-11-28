"""
Module for behavioral analyses.
"""

import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import scipy.stats
from sub_data import SUB_KEY
import utils

def save_analysis_output(out, kind, out_params, analysis_name, prefix=None,
    outdirname="analyses", ext=None):
    output_dir = op.join(utils.output.DIR, outdirname, analysis_name)
    utils.output.create_dir_if_needed(output_dir)
    if prefix is None:
        prefix = analysis_name
    name = utils.output.name_with_params(prefix, *out_params)
    if kind == "figure":
        ext = "png" if ext is None else ext
        save_fun = utils.output.save_figure
    elif kind == "stats":
        ext = "csv" if ext is None else ext
        save_fun = utils.output.save_stats
    elif kind == "text":
        ext = "txt" if ext is None else ext
        save_fun = utils.output.save_text
    path = utils.output.get_path(output_dir, name, ext)
    save_fun(out, path)

def save_analysis_figure(fig, out_params, analysis_name, prefix=None,
    outdirname="analyses", ext=None):
    save_analysis_output(fig, "figure", out_params, analysis_name, prefix=prefix,
    outdirname=outdirname, ext=ext)

def save_analysis_stats(stats_data, out_params, analysis_name, prefix=None,
    outdirname="analyses"):
    save_analysis_output(stats_data, "stats", out_params, analysis_name, prefix=prefix,
        outdirname=outdirname)

def save_analysis_text(text, out_params, analysis_name, prefix=None,
    outdirname="analyses"):
    save_analysis_output(text, "text", out_params, analysis_name, prefix=prefix,
        outdirname=outdirname)

def run_lr_analysis(data, config, out_params, pltkwargs={}, text=None,
    analysis_level="group", outdirname="analyses", do_sem=False, do_samples=False,
    do_count=False, lr_measure="lr", do_stats=False, do_by_var_alone=False,
    ext=None):
    """
    Analyze learning rate as a function of the absolute prediction error.
    This analysis is also done considering as a third factor,
    the true stochasticity level, or the inferred stochasticity level, as
    reflected by the bet width.
    """
    is_hsd_discrete = is_hsd_discrete_with_config(config)
    hsd_key = "prevhsd" if is_hsd_discrete else "prevhsd_bin"
    if (not is_hsd_discrete) and ("prevhsd_bin" not in data):
        add_hsd_bin_to_data(data, config)

    outputs = ["mean", "sem"] if do_sem else ["mean"]
    ape_bins = [0, 10, 20, 30, 50, 70, 90, 110, 180]
    figsize = (utils.plot.PAPER_CONTENT_WIDTH/3,
                utils.plot.PAPER_CONTENT_WIDTH/3)
    figsize_count = (2.3 * utils.plot.CM * 3/2, 2 * utils.plot.CM * 3/2)
    data['ape_bin'], bin_edges = pd.cut(data["ape"], bins=ape_bins,
        retbins=True, labels=False)

    # Analyze learning rate bin absolute prediction error and variance
    analysis_name = "learning_rate/by_prediction_error_and_variance"
    for do_hsd in [False, True]:
        prefix = "lr_by_pe"
        if lr_measure != "lr":
            analysis_name = "learning_rate/control_analyses"
            if lr_measure == "lr_exclude-no-update":
                analysis_name += "/exclude_no_movement_trials"
                prefix = "lr_excl_no_mov_by_pe"
            elif lr_measure == "p_updt_loc":
                analysis_name += "/proportion_movement"
                prefix = "prop_mov_by_pe"
        if do_hsd:
            prefix += f"_and_var"
        third_factor = hsd_key if do_hsd else None
        factors = ["ape_bin"] + ([third_factor] if third_factor else [])
        grouped_data = group_by_factors(data, factors,
            analysis_level=analysis_level, outputs=outputs)
        fig = plot_lr_analysis(grouped_data, third_factor, pltkwargs,
            figsize, text=text, do_sem=do_sem, lr_key=lr_measure)
        save_analysis_figure(fig, out_params, analysis_name, prefix,
            outdirname, ext=ext)
        if do_count:
            # Analyze count of data points based on which each learning rate point
            # was computed
            count_data = compute_bin_counts(data, ape_bins, third_factor, analysis_level=analysis_level)
            fig = plot_bin_counts(count_data, third_factor, pltkwargs,
                figsize_count, key=lr_measure, bins=ape_bins)
            save_analysis_figure(fig, out_params, analysis_name, "hist_" + prefix,
                outdirname, ext=ext)

    # Analyze learning rate by variance alone
    if do_by_var_alone:
        figsize = (utils.plot.PAPER_CONTENT_WIDTH/4,
                utils.plot.PAPER_CONTENT_WIDTH/4)
        analysis_name = "learning_rate/by_variance"
        prefix = "lr_by_var"
        grouped_data = group_by_factors(data, [hsd_key], analysis_level=analysis_level,
            outputs=outputs)
        fig = plot_mean_by_hsd(grouped_data, pltkwargs, figsize,
            y_key=lr_measure, text=text, do_sem=do_sem, do_samples=do_samples,
            ttest=None, ylim=(0., 0.8))
        save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname, ext=ext)

def group_by_factors(data, factors, analysis_level="group", outputs=["mean"]):
    if analysis_level == "group":
        # Compute the mean within subject first for each factor level,
        # then group by factor levels across subjects
        grouping_columns = [SUB_KEY] + factors
        grouped_data = data.groupby(grouping_columns).mean().groupby(grouping_columns[1:])
    else:
        # Group by factor levels
        grouped_data = data.groupby(factors)
    out = {"groupby": grouped_data}
    if "mean" in outputs:
        out["mean"] = grouped_data.mean(numeric_only=True)
    if "sem" in outputs:
        out["sem"] = grouped_data.sem(numeric_only=True)
    return out

def do_paired_stats(grouped_data, y_key):
    x_vals, y_vals = zip(*[(xval, d[y_key].values) for xval, d in grouped_data["groupby"]])
    assert len(x_vals) == len(y_vals)
    assert len(x_vals) == 2
    ttest = scipy.stats.ttest_rel(y_vals[0], y_vals[1])
    stats_data = pd.Series({
        "mean - condition 1": y_vals[0].mean(),
        "sem - condition 1": scipy.stats.sem(y_vals[0]),
        "mean - condition 2": y_vals[1].mean(),
        "sem - condition 2": scipy.stats.sem(y_vals[1]),
        "mean - paired diff 1-2": (y_vals[0] - y_vals[1]).mean(),
        "sem - paired diff 1-2": scipy.stats.sem(y_vals[0] - y_vals[1]),
        "degrees of freedom": ttest.df,
        "t statistic": ttest.statistic,
        "p value": ttest.pvalue,})
    return stats_data, ttest

def do_paired_ttest(grouped_data, y_key):
    x_vals, y_vals = zip(*[(xval, d[y_key].values) for xval, d in grouped_data["groupby"]])
    assert len(x_vals) == len(y_vals)
    assert len(x_vals) == 2
    return scipy.stats.ttest_rel(y_vals[0], y_vals[1])

def compute_bin_counts(data, ape_bins, third_factor=None, analysis_level="group"):
    if analysis_level == "group":
        # Compute the count of (non-nan) data points by ape bin and third factor
        # within subject,  then average the counts across subjects
        grouping_columns = [SUB_KEY, 'ape_bin'] + ([third_factor] if third_factor else [])
        count_data = data.groupby(grouping_columns).count().groupby(grouping_columns[1:]).mean()
    else:
        # Compute the count of (non-nan) data points by ape bin and third factor
        grouping_columns = ['ape_bin'] + ([third_factor] if third_factor else [])
        count_data = data.groupby(grouping_columns).count()
    return count_data

def plot_lr_analysis(grouped_data, third_factor, pltkwargs,
    figsize, text=None, do_sem=False,
    lr_key="lr"):
    # Plot mean learning rate as a function of absolute prediction error
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    if third_factor:
        mean_groups = grouped_data["mean"].groupby(third_factor)
        if do_sem:
            sem_groups = grouped_data["sem"].groupby(third_factor)
        for (third_factor_val, mean_group) in mean_groups:
            x = mean_group['ape'].values
            y = mean_group[lr_key].values
            kwargs = pltkwargs[third_factor][third_factor_val]["plot"]
            ax.plot(x, y, **kwargs)
            if do_sem:
                yerr = sem_groups.get_group(third_factor_val)[lr_key].values
                ax.fill_between(x, y + yerr, y - yerr, color=kwargs["color"],
                    alpha=utils.plot.ERROR_SHADING_ALPHA)
        ax.legend(title=utils.CPT_NAME_FOR_KEY[third_factor])
    else:
        x = grouped_data["mean"]["ape"].values
        y = grouped_data["mean"][lr_key].values
        kwargs = pltkwargs['plot']
        if do_sem:
            yerr = grouped_data["sem"][lr_key].values
            ax.plot(x, y, **kwargs)
            ax.fill_between(x, y + yerr, y - yerr, color=kwargs["color"],
                alpha=utils.plot.ERROR_SHADING_ALPHA)
        else:
            ax.plot(x, y, **pltkwargs['plot'])
    ax.set_xlabel(utils.CPT_NAME_FOR_KEY["ape"])
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[lr_key])
    ax.set_ylim(min(ax.get_ylim()[0], 0.), max(ax.get_ylim()[1], 1.05))
    return fig

def plot_bin_counts(count_data, third_factor, pltkwargs, figsize,
    key, bins=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if third_factor:
        i_group = 0
        for third_factor_val, group in count_data.groupby(third_factor):
            kwargs = pltkwargs[third_factor][third_factor_val]["bar"]
            bar_w = 0.4
            kwargs.pop("width", None)
            xticks = group.index.to_frame()["ape_bin"].to_numpy()
            ax.bar(xticks + bar_w * i_group,
                group[key],
                width=bar_w, align='edge',
                **pltkwargs[third_factor][third_factor_val]["bar"])
            i_group += 1
        ax.legend(title=utils.CPT_NAME_FOR_KEY[third_factor])
    else:
        xticks = count_data.index.to_frame()["ape_bin"].to_numpy()
        ax.bar(xticks, count_data[key], align='edge',
            **pltkwargs['bar'])
    labelpad = 2
    ax.set_xlabel(utils.CPT_NAME_FOR_KEY['ape'], labelpad=labelpad)
    ax.set_ylabel("Count", labelpad=labelpad)
    ax.set_ylim(min(ax.get_ylim()[0], 1), None)
    if bins is not None:
        ax.set_xticks(np.append(xticks, xticks[-1]+1),
            [f"{bins[i]}–" if i < (len(bins)-1) else f"{bins[i]}" for i in range(len(bins))]
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    return fig

def run_betwid_analysis(data, config, out_params, pltkwargs={}, text=None,
    analysis_level="group", outdirname="analyses",
    do_sem=False, do_samples=False, do_individual=False, do_count=False,
    do_stats=False,
    ext=None):
    """
    Analyze the bet width and the bet width increases/decreases:
    - as a function of the true stochasticity level
    - as a function of the absolute prediction error
    """
    is_hsd_discrete = is_hsd_discrete_with_config(config)
    hsd_key = "hsd" if is_hsd_discrete else "hsd_bin"
    if (not is_hsd_discrete) and ("hsd_bin" not in data):
        add_hsd_bin_to_data(data, config)

    outputs = ["mean", "sem"] if do_sem else ["mean"]

    # Analyze p(large width) per stochasticity level (hsd).
    # grouped_data = compute_mean_by_hsd(data, hsd_key=hsd_key, analysis_level=analysis_level,
    #     outputs=outputs)
    grouped_data = group_by_factors(data, [hsd_key], analysis_level=analysis_level,
        outputs=outputs)
    betwid_key = "nextplrgwid"
    analysis_name = "paddle_width/by_variance"
    prefix = f"pdlwid_by_var"
    if do_stats:
        stats_data, ttest = do_paired_stats(grouped_data, betwid_key)
        save_analysis_stats(stats_data, out_params, analysis_name, prefix, outdirname)
    else:
        ttest = None
    figsize = (utils.plot.PAPER_CONTENT_WIDTH/4,
               utils.plot.PAPER_CONTENT_WIDTH/4)
    fig = plot_mean_by_hsd(grouped_data, pltkwargs, figsize,
        betwid_key, text=text, do_sem=do_sem, do_samples=do_samples, ttest=ttest,
        ylim=(-0.05, 1.05))
    save_analysis_figure(fig, out_params, analysis_name, prefix,
        outdirname, ext=ext)

def plot_bar_means(grouped_data, pltkwargs, figsize, y_key, text=None,
    do_sem=False, ttest=None):
    # TBD: DRY/merge with analyses.plot_mean_by_hsd
    grouping_keys = grouped_data["groupby"].keys
    x_key = grouping_keys[0]
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    x = np.arange(len(grouped_data["mean"].index))
    y = grouped_data["mean"][y_key].values
    kwargs = pltkwargs['bar']
    if do_sem:
        yerr = grouped_data["sem"][y_key].values
        ax.bar(x, y, yerr=yerr, ecolor=utils.plot.GRAY_COLOR, **kwargs)
    else:
        ax.bar(x, grouped_data["mean"][y_key].values, **kwargs)
    ax.set_xlabel(utils.CPT_NAME_FOR_KEY[x_key])
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[y_key])
    xticks = grouped_data["mean"].index.values
    if isinstance(xticks[0], str):
        xticks = [s.capitalize() for s in xticks]
    ax.set_xticks(x, xticks)
    ax.set_ylim(0, 1.05)
    if ttest is not None:
        stars = utils.plot.stat_label(ttest.pvalue)
        ylim = ax.get_ylim()
        text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.95
        ax.text((x[0]+x[1])/2, text_y, stars, va="bottom", ha="center")
        text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.9
        t_text = utils.plot.stat_t_text(ttest)
        ax.text((x[0]+x[1])/2, text_y, t_text, va="bottom", ha="center",
            fontsize=5)
    return fig

def plot_bar_means_2factor(grouped_data, pltkwargs, figsize, y_key, text=None,
    do_sem=False):
    grouping_keys = grouped_data["groupby"].keys
    x_key = grouping_keys[0]
    label_key = grouping_keys[1]
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    mean_groups = grouped_data["mean"].groupby(label_key)
    if do_sem:
        sem_groups = grouped_data["sem"].groupby(label_key)
    i_group = 0
    bar_width = 0.25
    xticks = None
    for (label_val, mean_group) in mean_groups:
        y = mean_group[y_key].values
        x = np.arange(len(y))
        x_offset = i_group * bar_width
        kwargs = copy.deepcopy(pltkwargs[label_key][label_val]["bar"])
        kwargs["width"] = bar_width
        if do_sem:
            yerr = sem_groups.get_group(label_val)[y_key].values
            ax.bar(x + x_offset, y, yerr=yerr, ecolor=utils.plot.GRAY_COLOR,
                **kwargs)
        else:
            ax.bar(x + x_offset, y, **kwargs)
        if xticks is None:
            xticks = [v[0] for v in mean_group.index.values]
            if isinstance(xticks[0], str):
                xticks = [s.capitalize() for s in xticks]
        i_group += 1
    ax.legend(title=utils.CPT_NAME_FOR_KEY[label_key])
    ax.set_xlabel(utils.CPT_NAME_FOR_KEY[x_key])
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[y_key])    
    x_offset = (i_group - 1) * bar_width / 2
    ax.set_xticks(x + x_offset, xticks)
    ax.set_ylim(0, 1.05)
    return fig

def plot_mean_by_hsd(grouped_data, pltkwargs, figsize, y_key, text=None,
    do_sem=False, do_samples=False, ttest=None, ylim=(0, 1.05)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text, wrap=True)
    x = np.arange(len(grouped_data["mean"].index))
    xvals = grouped_data["mean"].index.values
    y = grouped_data["mean"][y_key].values
    color = [pltkwargs["hsd"][v]["bar"]["color"] for v in xvals]
    kwargs = pltkwargs['bar']
    kwargs.pop("color", None)
    if do_sem:
        yerr = grouped_data["sem"][y_key].values
        ax.bar(x, y, yerr=yerr, color=color, ecolor=utils.plot.GRAY_COLOR,
            linewidth=1., edgecolor="black", **kwargs)
    else:
        ax.bar(x, grouped_data["mean"][y_key].values, color=color,
            linewidth=1., edgecolor="black", **kwargs)
    if do_samples:
        # Plot individual data points with some x jitter
        num_participants = len(grouped_data["groupby"].get_group((xvals[0],))[y_key].values)
        xvals_per_participant = [[] for _ in range(num_participants)]
        yvals_per_participant = [[] for _ in range(num_participants)]
        for (i, xval) in enumerate(xvals):
            y_samples = grouped_data["groupby"].get_group((xval,))[y_key].values
            x_jitter = (np.random.rand(len(y_samples)) - 0.5) * 0.1
            x_samples = np.ones(len(y_samples)) * x[i] + x_jitter 
            ax.plot(x_samples, y_samples,
                marker='.', ls='None', color=utils.plot.GRAY_COLOR, ms=1,
                alpha=0.5,
                )
            for i_sub in range(len(y_samples)):
                xvals_per_participant[i_sub] += [x_samples[i_sub]]
                yvals_per_participant[i_sub] += [y_samples[i_sub]]
        # Join the dots corresponding to the same participant with a thin line
        for i_sub in range(len(xvals_per_participant)):
            ax.plot(xvals_per_participant[i_sub], yvals_per_participant[i_sub],
                color=utils.plot.GRAY_COLOR, lw=0.5, alpha=0.2)

    ax.set_xlabel(utils.CPT_NAME_FOR_KEY["hsd"])
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[y_key])
    ax.set_xticks(x, ["low", "high"])
    ax.set_ylim(ylim)
    if ttest is not None:
        stars = utils.plot.stat_label(ttest.pvalue)
        ylim = ax.get_ylim()
        text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.95
        ax.text((x[0]+x[1])/2, text_y, stars, va="bottom", ha="center")
        text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.9
        t_text = utils.plot.stat_t_text(ttest)
        ax.text((x[0]+x[1])/2, text_y, t_text, va="bottom", ha="center",
            fontsize=5)
    return fig

def plot_betwid_by_ape(grouped_data, ape_bins, pltkwargs, figsize,
    betwid_key, text=None, do_sem=False, do_individual=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    x = np.arange(len(ape_bins)-1)
    xticklabels = [f"[{ape_bins[i]}–{ape_bins[i+1]}]" for i in x]
    ax.set_xticks(x, xticklabels)
    if do_individual:
        Y = np.array([grouped_data["groupby"].get_group((i,))[betwid_key].values for i in x])
        for i_sub in range(Y.shape[1]):
            kwargs = copy.deepcopy(pltkwargs['plot'])
            kwargs["lw"] = 0.5
            kwargs["ms"] = 0.5
            ax.plot(x, Y[:, i_sub], **kwargs, alpha=0.5)
    else:
        y = grouped_data["mean"][betwid_key].values
        kwargs = pltkwargs['plot']
        if do_sem:
            yerr = grouped_data["sem"][betwid_key].values
            ax.errorbar(x, y, yerr=yerr, ecolor=utils.plot.GRAY_COLOR, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
    ax.set_xlabel(utils.CPT_NAME_FOR_KEY["ape"])
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[betwid_key])
    ax.set_ylim(0, 1.05)
    return fig

def run_around_event_analysis(data, config, out_params, pltkwargs={}, text=None,
    analysis_level="group", outdirname="analyses", do_sem=False,
    event_type="chghmean_and_chghsd", ext=None):
    outputs = ["mean", "sem"] if do_sem else ["mean"]
    win_around_cp = utils.era.get_window_around_cp(w_max=7)
    figsize = (utils.plot.PAPER_CONTENT_WIDTH / 2,
               utils.plot.DEFAULT_HEIGHT)
    hsd_levels = config["task"]["state_space"]["hsd"]["values"]
    lr_measure = "lr"

    # Analyze change points in mean (hmean).
    if event_type == "chghmean":
        # Analyze learning rate
        analysis_name = "learning_rate/event_related"
        prefix = f"lr_around_mean_cp"
        grouped_lrs = compute_var_around_event_hmean(data, win_around_cp, None,
            var_key=lr_measure, analysis_level=analysis_level, outputs=outputs)
        xlabel = "# Obs. after change point in mean"
        fig = plot_var_around_cp_hmean(grouped_lrs, win_around_cp, pltkwargs,
            figsize, var_key=lr_measure, text=text, do_sem=do_sem, xlabel=xlabel)
        save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname, ext=ext)
        # Same analysis, split by stochasticity level (hsd).
        analysis_name = "learning_rate/event_related"
        prefix = f"lr_around_mean_cp_by_var"
        grouped_lrs = compute_var_around_event_hmean(data, win_around_cp, hsd_levels,
            var_key=lr_measure, analysis_level=analysis_level, outputs=outputs)
        fig = plot_var_around_cp_hmean(grouped_lrs, win_around_cp, pltkwargs,
            figsize, var_key=lr_measure, factor="hsd", text=text, do_sem=do_sem, xlabel=xlabel)
        save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname, ext=ext)

    # Analyze change points in variance (hsd).
    if event_type == "chghsd": 
        # Analyze p(large width) around a change point in stochasticity (hsd),
        # distinguishing between an increase and a decrease in hsd.
        analysis_name = "paddle_width/event_related"
        prefix = "padlwid_around_var_cp"
        var_key = "nextplrgwid" # use *next*plrgwid, the response *after* having
                                # received the observation
        group_plrgwid = compute_var_around_cp_hsd(data, win_around_cp, var_key=var_key,
            analysis_level=analysis_level, outputs=outputs)
        fig = plot_var_around_cp_hsd(group_plrgwid, win_around_cp,
            pltkwargs, figsize,  var_key=var_key, text=text, do_sem=do_sem)
        save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname,
            ext=ext)
        # Analyze the apparent learning rate around a change point in stochasticity (hsd),
        # distinguishing between an increase and a decrease in hsd.
        analysis_name = "learning_rate/event_related"
        prefix = "lr_around_var_cp"
        grouped_data = compute_var_around_cp_hsd(data, win_around_cp, var_key=lr_measure,
            analysis_level=analysis_level, outputs=outputs)
        fig = plot_var_around_cp_hsd(grouped_data, win_around_cp,
            pltkwargs, figsize, var_key=lr_measure, text=text, do_sem=do_sem)
        save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname,
            ext=ext)
        
    # Jointly analyze change points in mean (hmean) and change points in variance (hsd).
    if event_type == "chghmean_and_chghsd":
        for meastype in ["lr", "padlwid"]:
            var_key = lr_measure if (meastype == "lr") else "nextplrgwid"
            analysis_name = "learning_rate/event_related" if (meastype == "lr") else "paddle_width/event_related"
            prefix = "lr_around_mean_and_var_cp" if (meastype == "lr") else "padlwid_around_mean_and_var_cp"
            grouped_vals_cp_hmean = compute_var_around_event_hmean(data, win_around_cp, None,
                var_key=var_key, analysis_level=analysis_level, outputs=outputs)
            grouped_vals_cp_hsd = compute_var_around_cp_hsd(data, win_around_cp, var_key=var_key,
                analysis_level=analysis_level, outputs=outputs, signs=[+1, -1])
            grouped_vals_inc_hsd = {k: grouped_vals_cp_hsd[k][+1] for k in grouped_vals_cp_hsd}
            grouped_vals_dec_hsd = {k: grouped_vals_cp_hsd[k][-1] for k in grouped_vals_cp_hsd}
            fig = plot_var_around_cp_hmean_and_inc_dec_hsd(
                grouped_vals_cp_hmean, grouped_vals_inc_hsd, grouped_vals_dec_hsd,
                win_around_cp, pltkwargs, figsize,
                    var_key=var_key, xlabel="# Obs. after event",
                    text=None,
                    do_sem=do_sem)
            save_analysis_figure(fig, out_params, analysis_name, prefix, outdirname,
                ext=ext)
        

def analyze_padlwid_increase_duration(data, config, out_params,
                              outdirname="analyses", var_key="nextplrgwid",
                              n_pre_event_lags=2, max_post_event_lags=5,
                              return_fraction=0.5):
    """
    Analyze and compare the duration of paddle-width increases following
    change points in mean vs. increases in variance.
    """
    win_around_cp = utils.era.get_window_around_cp(w_max=14)
    
    # Compute individual curves for change point in mean
    grouped_cp_hmean = compute_var_around_event_hmean(
        data, win_around_cp, None, var_key=var_key, 
        analysis_level="group", outputs=["individual"], event="chghmean")
    curves_cp_hmean = grouped_cp_hmean["individual"]  # shape: (n_time_lags, n_participants)
    
    # Compute individual curves for increase in variance
    grouped_cp_hsd = compute_var_around_cp_hsd(
        data, win_around_cp, var_key=var_key,
        analysis_level="group", signs=[+1], outputs=["individual"])
    curves_inc_hsd = grouped_cp_hsd["individual"][+1]  # shape: (n_time_lags, n_participants)
    
    # Compute durations for each participant
    durations_cp_hmean = compute_increase_durations_by_participant(
        curves_cp_hmean, win_around_cp, n_pre_event_lags, 
        max_post_event_lags, return_fraction)
    durations_inc_hsd = compute_increase_durations_by_participant(
        curves_inc_hsd, win_around_cp, n_pre_event_lags, 
        max_post_event_lags, return_fraction)
    
    # Remove NaN values (but keep paired structure)
    valid_mask = ~(np.isnan(durations_cp_hmean) | np.isnan(durations_inc_hsd))
    durations_cp_hmean_valid = durations_cp_hmean[valid_mask]
    durations_inc_hsd_valid = durations_inc_hsd[valid_mask]
    n_valid = np.sum(valid_mask)
    
    # Compute descriptive statistics
    mean_cp_hmean = np.mean(durations_cp_hmean_valid)
    sem_cp_hmean = scipy.stats.sem(durations_cp_hmean_valid)
    median_cp_hmean = np.median(durations_cp_hmean_valid)
    mean_inc_hsd = np.mean(durations_inc_hsd_valid)
    sem_inc_hsd = scipy.stats.sem(durations_inc_hsd_valid)
    median_inc_hsd = np.median(durations_inc_hsd_valid)
    
    # Perform a Wilcoxon signed-rank test comparing the two duration distributions
    wilcoxon_result = scipy.stats.wilcoxon(durations_inc_hsd_valid, durations_cp_hmean_valid)
    # Compute the proportion of participants with longer durations following an increase in variance
    n_inc_longer = int(np.sum(durations_inc_hsd_valid > durations_cp_hmean_valid))
    assert n_valid > 0
    proportion_inc_longer = n_inc_longer / n_valid
    binomtest_result = scipy.stats.binomtest(n_inc_longer, n_valid)
    ci = binomtest_result.proportion_ci(confidence_level=0.95, method="wilson")
    proportion_ci_low = ci.low
    proportion_ci_high = ci.high
    
    # Save results to CSV
    analysis_name = "paddle_width/event_related/increase_duration_analysis"
    prefix = f"padlwid_incr_dur_event_comparison"
    results = {
        "event_type": ["change_point_mean", "increase_variance"],
        "n_participants": [n_valid, n_valid],
        "median_duration": [median_cp_hmean, median_inc_hsd],
        "mean_duration": [mean_cp_hmean, mean_inc_hsd],
        "sem_duration": [sem_cp_hmean, sem_inc_hsd],
        "wilcoxon_statistic": [wilcoxon_result.statistic, wilcoxon_result.statistic],
        "wilcoxon_p_value": [wilcoxon_result.pvalue, wilcoxon_result.pvalue],
        "proportion_inc_gt_cp": [proportion_inc_longer, proportion_inc_longer],
        "proportion_inc_gt_cp_ci_low": [proportion_ci_low, proportion_ci_low],
        "proportion_inc_gt_cp_ci_high": [proportion_ci_high, proportion_ci_high],
        "binomial_p_value": [binomtest_result.pvalue, binomtest_result.pvalue],
    }
    df_results = pd.DataFrame(results)
    save_analysis_stats(df_results, out_params, analysis_name, prefix, outdirname)

def compute_var_around_event_hmean(data, win_around_cp, hsd_levels, var_key="lr",
    analysis_level="group", outputs=["mean"], event="chghmean", ape_thresh=90):
    do_split_by_hsd = (hsd_levels is not None) and (len(hsd_levels) > 1)
    if do_split_by_hsd:
        result = {k: [[] for t in win_around_cp] for k in hsd_levels}
    else:
        result = [[] for t in win_around_cp]
    
    subids = data[SUB_KEY].unique() if analysis_level == "group" else ["dummy"]
    for sub_id in subids:
        if analysis_level == "group":
            sub_data = data[data[SUB_KEY] == sub_id]
        else:
            sub_data = data
        if do_split_by_hsd:
            sub_result = {k: [[] for t in win_around_cp] for k in hsd_levels}
        else:
            sub_result = [[] for t in win_around_cp]
        
        for i_block in sub_data['blockIdx'].unique():
            seqdata = sub_data[(sub_data['blockIdx'] == i_block)]
            lrs = seqdata[var_key].values
            if event == "high-ape":
                mean_change_vals = np.where(seqdata["ape"].values >= ape_thresh,
                    1, 0)
            else:
                mean_change_vals = seqdata["hmeanchg"].values
            if do_split_by_hsd:
                # Segment the sequence into periods where the stochasticity level remains
                # the same and iterate over those periods to assign learning rate values to
                # a (time point, stochasticity level).
                # TBD: Encapsulate this segmentation in a reusable function, called
                # e.g. utils.era.find_periods() such that the code becomes 'for
                # start, end in utils.era.find_periods(changevals)'.
                stoch_change_vals = seqdata["hsdchg"].values
                stoch_did_change = ~np.isclose(stoch_change_vals, 0)
                i_stoch_changes = np.argwhere(stoch_did_change)[:, 0]
                stoch_vals = seqdata["hsd"].values
                for start, end in zip(np.insert(i_stoch_changes, 0, 0),
                    np.insert(i_stoch_changes, i_stoch_changes.shape[0], stoch_did_change.shape[0])):
                    assert np.allclose(stoch_vals[start:end], stoch_vals[start])
                    lrs_around_cp = utils.era.aggregate_values_in_window_around_cp(
                        lrs[start:end], win_around_cp, mean_change_vals[start:end])
                    stoch = stoch_vals[start]
                    for t_win in range(len(win_around_cp)):
                        sub_result[stoch][t_win] += lrs_around_cp[t_win]
            else:
                lrs_around_cp = utils.era.aggregate_values_in_window_around_cp(
                    lrs, win_around_cp, mean_change_vals)
                for t_win in range(len(win_around_cp)):
                    sub_result[t_win] += lrs_around_cp[t_win]

        if do_split_by_hsd:
            for k in hsd_levels:
                for t_win in range(len(win_around_cp)):
                    vals = sub_result[k][t_win]
                    result[k][t_win] += [np.nanmean(vals) if len(vals) > 0 else np.nan]
        else:
            for t_win in range(len(win_around_cp)):
                vals = sub_result[t_win]
                result[t_win] += [np.nanmean(vals) if len(vals) > 0 else np.nan]

    out = {}
    if do_split_by_hsd:
        if "mean" in outputs:
            out["mean"] = {k: np.nanmean(np.array(result[k]), axis=-1) for k in hsd_levels}
        if "sem" in outputs:
            out["sem"] = {k: scipy.stats.sem(np.array(result[k]), axis=-1,
                nan_policy="omit") for k in hsd_levels}
        if "individual" in outputs:
            out["individual"] = {k: np.array(result[k]) for k in hsd_levels}
    else:
        a = np.array(result) # shape: (n_time_lags, n_participants)
        if "mean" in outputs:
            out["mean"] = np.nanmean(a, axis=-1)
        if "sem" in outputs:
            out["sem"] = scipy.stats.sem(a, axis=-1, nan_policy="omit")
        if "individual" in outputs:
            out["individual"] = a
    return out

def compute_var_around_cp_hsd(data, win_around_cp, var_key="nextplrgwid",
    analysis_level="group", signs=[+1, -1], outputs=["mean"]):
    result = {sgn: [[] for t in win_around_cp] for sgn in signs}
    
    subids = data[SUB_KEY].unique() if analysis_level == "group" else ["dummy"]
    for sub_id in subids:
        if analysis_level == "group":
            sub_data = data[data[SUB_KEY] == sub_id]
        else:
            sub_data = data
        sub_result = {sgn: [[] for t in win_around_cp] for sgn in signs}
        
        for i_block in sub_data['blockIdx'].unique():
            seq_data = sub_data[(sub_data['blockIdx'] == i_block)]
            plrgwid = seq_data[var_key].values
            hsd_change_vals = seq_data["hsdchg"].values

            for sgn in signs:
                win_plrgwid = utils.era.aggregate_values_in_window_around_cp(
                    plrgwid, win_around_cp, hsd_change_vals, change_sign=sgn)
                for t_win in range(len(win_around_cp)):
                    sub_result[sgn][t_win] += win_plrgwid[t_win]

        for sgn in signs:
            for t_win in range(len(win_around_cp)):
                result[sgn][t_win] += [np.nanmean(sub_result[sgn][t_win])]
                
    out = {}
    if "mean" in outputs:
        out["mean"] = {sgn: np.mean(np.array(result[sgn]), axis=-1) for sgn in signs}
    if "sem" in outputs:
        out["sem"] = {sgn: scipy.stats.sem(np.array(result[sgn]), axis=-1, nan_policy="omit") for sgn in signs}
    if "individual" in outputs:
        # shape: (n_time_lags, n_participants)
        out["individual"] = {sgn: np.array(result[sgn]) for sgn in signs}
    return out

def compute_increase_duration(curve, win_around_cp, n_pre_event_lags=2, 
                             max_post_event_lags=4, return_fraction=0.5):
    """
    Compute the duration of an increase in a curve relative to baseline.
    
    Parameters:
    -----------
    curve : array-like (shape: n_time_lags)
        Values across time lags 
    win_around_cp : array-like (shape: n_time_lags)
        Time lag values (e.g., [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
        The event onset corresponds to time lag 0.
        Time lags >= 0 are considered post-event.
    n_pre_event_lags : int
        Number of pre-event lags to use for baseline (default: 2)
    max_post_event_lags : int
        Maximum number of post-event lags to consider for peak (default: 4)
    return_fraction : float
        Fraction of the amplitude to return to (0.5 = halfway back to baseline)
    
    Returns:
    --------
    duration : float or np.nan
        Number of lags from peak to return threshold, or np.nan if cannot be computed
    """
    # Find indices corresponding to pre-event and post-event periods
    event_onset = 0
    pre_event_mask = np.array(win_around_cp) < event_onset
    post_event_mask = np.array(win_around_cp) >= event_onset
    
    if np.sum(pre_event_mask) < n_pre_event_lags:
        return np.nan
    
    # 1. Compute baseline (average of pre-event lags)
    baseline = np.nanmean(curve[pre_event_mask][-n_pre_event_lags:])
    
    # 2. Find peak in post-event period (up to max_post_event_lags)
    post_event_indices = np.where(post_event_mask)[0]
    search_indices = post_event_indices[:min(max_post_event_lags, len(post_event_indices))]
    
    if len(search_indices) == 0:
        return np.nan
    
    peak_idx_rel = np.nanargmax(curve[search_indices])
    peak_idx = search_indices[peak_idx_rel]
    peak_value = curve[peak_idx]
    
    # 3. Compute amplitude
    amplitude = peak_value - baseline
    
    if amplitude <= 0 or np.isnan(amplitude):
        return np.nan
    
    # 4. Compute threshold (return_fraction of the way back from peak to baseline)
    threshold = peak_value - (return_fraction * amplitude)
    
    # 5. Find duration: number of consecutive lags from peak to cross threshold
    duration = 1
    for idx in range(peak_idx + 1, len(curve)):
        if curve[idx] <= threshold:
            break
        duration += 1

    return duration


def compute_increase_durations_by_participant(individual_curves, win_around_cp,
                                     n_pre_event_lags=2, max_post_event_lags=5, 
                                     return_fraction=0.5):
    """
    Compute increase durations for each participant from individual curves.
    
    Parameters:
    -----------
    individual_curves : np.ndarray
        Shape: (n_time_lags, n_participants)
    win_around_cp : array-like
        Time lag values
    
    Returns:
    --------
    durations : np.ndarray
        Duration for each participant (shape: n_participants)
    """
    n_time_lags, n_participants = individual_curves.shape
    durations = np.zeros(n_participants)
    
    for i_sub in range(n_participants):
        curve = individual_curves[:, i_sub]
        durations[i_sub] = compute_increase_duration(
            curve, win_around_cp, n_pre_event_lags, 
            max_post_event_lags, return_fraction)
    
    return durations

def plot_var_around_cp(grouped_vals, win_around_cp, pltkwargs, figsize,
    var_key, factor, xlabel, legendtitle, text=None, do_sem=False,
    x_offset=1, fig=None):
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    xvals = win_around_cp + x_offset
    xticks = xvals[2::2]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[var_key])
    if factor is not None:
        for i, (k, y) in enumerate(grouped_vals["mean"].items()):
            kwargs = pltkwargs[factor][k]["plot"]
            ax.plot(xvals, y, **kwargs)
            if do_sem:
                yerr = grouped_vals["sem"][k]
                ax.fill_between(xvals, y + yerr, y - yerr, color=kwargs["color"],
                    alpha=utils.plot.ERROR_SHADING_ALPHA)
    else:
        y = grouped_vals["mean"]
        kwargs = pltkwargs["plot"]
        ax.plot(xvals, y, **kwargs)
        if do_sem:
            yerr = grouped_vals["sem"]
            ax.fill_between(xvals, y + yerr, y - yerr, color=kwargs["color"],
                alpha=utils.plot.ERROR_SHADING_ALPHA)

    ax.set_xticks(xticks)
    if factor is not None:
        ax.legend(title=legendtitle)
    ax.set_ylim(0, 1.05)
    return fig

def plot_var_around_cp_hmean(grouped_lrs, win_around_cp, pltkwargs, figsize,
    var_key="lr", factor=None, text=None, do_sem=False,
    xlabel="# Obs. after change point in mean", fig=None):
    return plot_var_around_cp(grouped_lrs, win_around_cp, pltkwargs,
        figsize, var_key, factor, xlabel,
        utils.CPT_NAME_FOR_KEY[factor] if factor else None,
        text=text, do_sem=do_sem, fig=fig)

def plot_var_around_cp_hsd(grouped_plrgwid, win_around_cp, pltkwargs, figsize,
    var_key="nextplrgwid", text=None, do_sem=False):
    return plot_var_around_cp(grouped_plrgwid, win_around_cp, pltkwargs,
        figsize, var_key, "chgsgn", "# Obs. after increase/decrease in stochasticity",
        None, text=text, do_sem=do_sem)

def plot_var_around_cp_hmean_and_inc_hsd(grouped_vals_cp_hmean,
    grouped_vals_inc_hsd, win_around_cp, pltkwargs, figsize,
    var_key, xlabel="# Obs. after event", text=None, do_sem=False, x_offset=1,
    label_cp_hmean="Change point in mean"):
    return plot_var_around_cp_hmean_and_inc_dec_hsd(grouped_vals_cp_hmean,
        grouped_vals_inc_hsd, None, win_around_cp, pltkwargs, figsize=figsize,
        var_key=var_key, xlabel=xlabel, text=text, do_sem=do_sem, x_offset=x_offset,
        label_cp_hmean=label_cp_hmean)

def plot_var_around_cp_hmean_and_inc_dec_hsd(grouped_vals_cp_hmean,
    grouped_vals_inc_hsd, grouped_vals_dec_hsd, win_around_cp, pltkwargs, figsize,
    var_key, xlabel="# Obs. after event", text=None, do_sem=False, x_offset=1,
    label_cp_hmean="Change point in mean", ylim=(-0.1, 1.1)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if text is not None:
        utils.plot.add_text(ax, text)
    xvals = win_around_cp + x_offset
    xticks = xvals[2::2]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(utils.CPT_NAME_FOR_KEY[var_key])
    triplets = [
        (grouped_vals_cp_hmean, label_cp_hmean, pltkwargs["chghmean"]["plot"]["color"], "o"),
        (grouped_vals_inc_hsd, "Increase in variance", pltkwargs["chghsd"]["plot"]["color"], "^")]
    if grouped_vals_dec_hsd is not None:
        triplets += [(grouped_vals_dec_hsd, "Decrease in variance", pltkwargs["chghsd"]["plot"]["color"], "v")]
    for grouped_vals, label, color, marker in triplets:
        y = grouped_vals["mean"]
        kwargs = pltkwargs["plot"].copy()
        kwargs["color"] = color
        kwargs["marker"] = marker
        ax.plot(xvals, y, label=label, **kwargs)
        if do_sem:
            yerr = grouped_vals["sem"]
            ax.fill_between(xvals, y + yerr, y - yerr, color=kwargs["color"],
                alpha=utils.plot.ERROR_SHADING_ALPHA)
    ax.set_xticks(xticks)
    ax.legend(title="Event type", alignment="left", title_fontsize=7)
    ax.set_ylim(ylim)
    # Add vertical dashed line to indicate event onset
    ax.axvline(x=(xvals[1] + xvals[2]) / 2, ymax=0.95, color="black", ls=":", lw=1.,
        zorder=-1)
    ax.text((xvals[1] + xvals[2]) / 2, ylim[0] + (ylim[1]-ylim[0])*0.96, "Event",
        ha="center", va="bottom",
        fontsize=7,
        color="black")
    return fig

def run_performance_analysis(data, config, out_params,
    analysis_level="group",
    outdirname="analyses",
    do_sem=False, do_score=True):
    """
    Analyze task performance, through several measures of performance
    """
    if analysis_level == "group":
        # Compute mean within subject
        data = data.groupby([SUB_KEY]).mean()
    res = {}
    all_keys = ["did_catch", "ape", "mae", "plrgwid", "pincwid", "pdecwid", "reward"]
    if do_score:
        all_keys += ["score"]
    keys = [k for k in all_keys if k in data]
    for k in keys:
        vals = data[k].values
        res[k] = {
            "mean": np.mean(vals),
            "sem": scipy.stats.sem(vals),
        }
    
    table_rows = []
    for k, v in res.items():
        if k == "did_catch":
            l = f"Proportion of beams caught"
        elif k == "ape":
            l = f"Absolute prediction error"
        elif k == "mae":
            l = f"Mean absolute error"
        elif k == "plrgwid":
            l = f"Proportion of trials using large paddle"
        elif k == "pincwid":
            l = f"Proportion of trials switching from small to large paddle"
        elif k == "pdecwid":
            l = f"Proportion of trials switching from large to small paddle"
        elif k == "reward":
            l = "Reward per trial (pts)"
        elif k == "score":
            l = "Score per block (pts)"
        row = {
            "Metric": l,
            "Mean": v['mean']
        }
        if do_sem:
            row["Sem"] = v['sem']
        table_rows += [row]
    
    table = pd.DataFrame(table_rows)
    analysis_name = "performance"
    prefix = "perf_summary"
    save_analysis_stats(table, out_params, analysis_name, prefix=prefix,
        outdirname=outdirname)

def add_hsd_bin_to_data(data, config, hsd_cutoff=20.):
    config_hsd_state_space = config["task"]["state_space"]["hsd"]
    hsd_min = config_hsd_state_space.get("min", 0.)
    hsd_max = config_hsd_state_space.get("max", np.inf)
    hsd_cutoff = 20.
    hsd_bins = [hsd_min, hsd_cutoff, hsd_max]
    data["prevhsd_bin"] = pd.cut(data["prevhsd"],
        bins=hsd_bins, include_lowest=True,
        labels=False)
    data["hsd_bin"], retbins = pd.cut(data["hsd"],
        bins=hsd_bins, include_lowest=True,
        labels=False, retbins=True)

def is_hsd_discrete_with_config(config):
    return config["task"]["state_space"]["hsd"]["type"] == "discrete"

"""
Perform group-level behavioral analyses.
"""

from analyses import (run_lr_analysis, run_betwid_analysis, run_around_event_analysis,
    run_performance_analysis, save_analysis_text, analyze_padlwid_increase_duration
    )
import argparse
import numpy as np
import os.path as op
import pandas as pd
import scipy
from sub_analyses import prepare_plot_kwargs
import sub_data
from sub_summary_table import str_from_ms
import utils

def main(args):
    np.random.seed(0)
    dataset = args.dataset
    sub_list_path, task_config_path = utils.datasets.get_dataset_paths(dataset)
    data_files = sub_data.get_subject_files(sub_list_path)

    task_config = utils.config.load_config_at_path_with_default_task_config(task_config_path)
    config = utils.config.create_joint_config(task_config)

    # Concatenate individual subjects' data into a single dataframe
    sub_dfs = []
    for i_file, data_file in enumerate(data_files):
        data = sub_data.load_subject_data(data_file)
        df = sub_data.make_subject_dataframe(data, config)
        sub_dfs += [df]
    df = pd.concat(sub_dfs)

    utils.plot.setup_mpl_style(fontsize=7)
    pltkwargs = prepare_plot_kwargs(config)

    out_params = ["human"], [True]
    text = f"Human behavior"
    outdirname = dataset
    is_cp_task = config["task"]["dynamics"]["hmean"]["type"] == "change_point"
    for lr_measure in ["lr", "p_updt_loc", "lr_exclude-no-update"]:
        run_lr_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
            analysis_level="group", outdirname=outdirname, do_sem=True, do_samples=True,
            lr_measure=lr_measure, do_stats=True, ext=args.ext,
            do_count=(True if lr_measure=="lr" else False),
            do_by_var_alone=(True if lr_measure=="lr" else False))
    run_betwid_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
        analysis_level="group", outdirname=outdirname,
        do_sem=True, do_samples=True, do_individual=True, do_stats=True, ext=args.ext)
    if is_cp_task:
        run_around_event_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
                analysis_level="group", outdirname=outdirname, do_sem=True, ext=args.ext)
        analyze_padlwid_increase_duration(data=df,
            config=config,
            out_params=out_params,
            outdirname=outdirname)
    run_performance_analysis(df, config, out_params, outdirname=outdirname,
        do_sem=True)
    run_completion_time_analysis(data_files, outdirname, out_params)


def run_completion_time_analysis(data_files, outdirname, out_params):
    analysis_name = "performance"
    prefix = "completion_times"
    n_sub = len(data_files)
    completion_times = {k: np.empty(n_sub) for k in ["total", "instructions", "task"]}
    for i, data_file in enumerate(data_files):
        data = sub_data.load_subject_data(data_file)
        t_start_instructions = data["expEvents"]["startInstructions"][0][0]
        t_start_fstblock = data["expEvents"]["startBlock"][0][0]
        t_end_lstblock = data["expEvents"]["finishBlock"][-1][0]
        completion_times["total"][i] = t_end_lstblock - t_start_instructions
        completion_times["instructions"][i] = t_start_fstblock - t_start_instructions
        completion_times["task"][i] = t_end_lstblock - t_start_fstblock

    completion_time_summaries = {}
    for k in completion_times:
        completion_time_summaries[k] = {
            "median": np.median(completion_times[k]),
            "iqr_low": np.quantile(completion_times[k], 1/4),
            "iqr_high": np.quantile(completion_times[k], 3/4),
            "mean": np.mean(completion_times[k]),
            "sem": scipy.stats.sem(completion_times[k]),
        }

    time_strings = {k1: {k2: str_from_ms(completion_time_summaries[k1][k2], sep="")
        for k2 in completion_time_summaries[k1]}
        for k1 in completion_time_summaries}
    lines = ["Completion times — median [IQR] | mean ± sem"]
    for k1 in time_strings:
        times = time_strings[k1]
        lines += [f"{k1.capitalize()} — " +
                f" {times['median']} [{times['iqr_low']}, {times['iqr_high']}] | {times['mean']} ± {times['sem']}"""]

    text = "\n".join(lines)
    save_analysis_text(text, out_params, analysis_name, prefix=prefix,
        outdirname=outdirname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["change-point-env", "random-walk-env"],
        help="Which experimental dataset / environment to analyze.")
    parser.add_argument("-ext", default="pdf")
    args = parser.parse_args()
    main(args)

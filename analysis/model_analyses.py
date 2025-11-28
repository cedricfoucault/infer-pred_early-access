"""
Perform group-level behavioral analyses using the model's behavior instead of the
participants' behavior (this script mirrors group_analyses.py).
"""

from analyses import (run_lr_analysis, run_betwid_analysis, run_around_event_analysis,
    run_performance_analysis
    )
import argparse
import infer_pred_model
import numpy as np
import os.path as op
import pandas as pd
from sim_analyses import prepare_plot_kwargs
import sub_data
import utils

def main(args):
    np.random.seed(0)
    dataset = args.dataset
    sub_list_path, task_config_path = utils.datasets.get_dataset_paths(dataset)
    data_files = sub_data.get_subject_files(sub_list_path)
    task_config = utils.config.load_config_at_path_with_default_task_config(task_config_path)
    model_config = utils.config.load_config_at_path_with_default_model_config(args.model_config_path)
    config = utils.config.create_joint_config(task_config, model_config)
    model = infer_pred_model.get_model(config)

    sub_dfs = []
    for i_file, data_file in enumerate(data_files):
        data = sub_data.load_subject_data(data_file)
        df = sub_data.make_dataframe_for_subject_data_and_model(data, config, model=model)
        sub_dfs += [df]
    df = pd.concat(sub_dfs)

    utils.plot.setup_mpl_style(fontsize=7)
    pltkwargs = prepare_plot_kwargs(config)

    out_params = ["model"], [model_config['id']]
    text = model.name
    outdirname = dataset
    is_cp_task = config["task"]["dynamics"]["hmean"]["type"] == "change_point"
    if not args.skip_lr:
        lr_measures = ["lr", "p_updt_loc", "lr_exclude-no-update"] if not args.skip_lr_control else ["lr"]
        for lr_measure in lr_measures:
            run_lr_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
                analysis_level="group", outdirname=outdirname, do_sem=False, ext=args.ext,
                lr_measure=lr_measure, do_stats=args.do_stats,
                do_count=(True if lr_measure=="lr" else False),
                do_by_var_alone=(True if lr_measure=="lr" else False))
    if not args.skip_padlwid:
        run_betwid_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
            analysis_level="group", outdirname=outdirname,
            do_sem=False, do_individual=False, ext=args.ext,
            do_stats=args.do_stats)
    if is_cp_task and not args.skip_padlwid:
        run_around_event_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
            analysis_level="group", outdirname=outdirname, do_sem=False, ext=args.ext)
    if not args.skip_perf:
        run_performance_analysis(df, config, out_params, outdirname=outdirname, do_score=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["change-point-env", "random-walk-env"],
        help="Which experimental dataset / environment to analyze.")
    parser.add_argument("-mcp", "--model_config_path", default=None)
    parser.add_argument("-ext", default="pdf")
    parser.add_argument("--skip_lr", action='store_true', default=False)
    parser.add_argument("--skip_lr_control", action='store_true', default=False)
    parser.add_argument("--skip_padlwid", action='store_true', default=False)
    parser.add_argument("--skip_perf", action='store_true', default=False)
    parser.add_argument("--do_stats", action='store_true', default=False)
    args = parser.parse_args()
    main(args)

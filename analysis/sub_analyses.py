"""
Perform single-subject behavioral analyses.
"""

from analyses import run_lr_analysis, run_betwid_analysis, run_around_event_analysis
import argparse
import copy
import numpy as np
import os.path as op
import pandas as pd
import sub_data
from sub_data import SUB_KEY
import utils

def main(args):
    dataset = args.dataset
    sub_list_path, task_config_path = utils.datasets.get_dataset_paths(dataset)
    task_config = utils.config.load_config_at_path_with_default_task_config(task_config_path)
    config = utils.config.create_joint_config(task_config)
    # Subject's data
    data = sub_data.load_subject_data(args.data_file)
    df = sub_data.make_subject_dataframe(data, config)
    sub_id = data[SUB_KEY]

    utils.plot.setup_mpl_style(fontsize=7)
    pltkwargs = prepare_plot_kwargs(config)

    out_params = ['sub'], [sub_id]
    text = f"Subject: {sub_id}"
    outdirname = op.join(dataset, "single_subject", f"sub-{sub_id}")
    run_lr_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
        analysis_level="subject", outdirname=outdirname, do_count=True)
    run_betwid_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
        analysis_level="subject", outdirname=outdirname)
    is_cp_task = config["task"]["dynamics"]["hmean"]["type"] == "change_point"
    if is_cp_task:
        run_around_event_analysis(df, config, out_params, pltkwargs=pltkwargs, text=text,
            analysis_level="subject", outdirname=outdirname)

def prepare_plot_kwargs(config):
    # Default plot parameters
    plotkwargs = dict(marker='o', linestyle='-', ms=2, lw=1.5,
            color=utils.plot.COLORS_BLUE[5])
    barplotkwargs = dict(width=0.5, color=utils.plot.COLORS_BLUE[6])
    pltkwargs_dflt = {"plot": plotkwargs, "bar": barplotkwargs}
    pltkwargs = copy.deepcopy(pltkwargs_dflt)
    # Plot parameters for different levels of stochasticity or bet width
    hsd_levels = config["task"]["state_space"]["hsd"]["values"]
    betwid_levels = config["task"]["responses"]["widths"]
    hsd_labels = ["low", "high"]
    betwid_labels = ["small", "large"]
    for factor, values, labels in [("hsd", hsd_levels, hsd_labels), ("prevhsd", hsd_levels, hsd_labels),
        ("betwid", betwid_levels, betwid_labels), ("prevbetwid", betwid_levels, betwid_labels)]:
        pltkwargs[factor] = {v: copy.deepcopy(pltkwargs_dflt) for v in values}
        for i, v in enumerate(values):
            pltkwargs[factor][v]['plot']["color"] = (
                utils.plot.COLORS_BLUE[5] if i == 0
                else utils.plot.COLORS_ORANGE[5])
            pltkwargs[factor][v]['plot']["label"] = labels[i]
            pltkwargs[factor][v]['bar']["color"] = (
                utils.plot.COLORS_BLUE[5] if i == 0
                else utils.plot.COLORS_ORANGE[5])
            pltkwargs[factor][v]['bar']["label"] = labels[i]
    # Plot parameters for different signs of change
    signs = [+1, -1]
    pltkwargs["chgsgn"] = {v: copy.deepcopy(pltkwargs_dflt) for v in signs}
    pltkwargs["chgsgn"][+1]["plot"]["linestyle"] = "-"
    pltkwargs["chgsgn"][-1]["plot"]["linestyle"] = ":"
    pltkwargs["chgsgn"][+1]["plot"]["label"] = "increase"
    pltkwargs["chgsgn"][-1]["plot"]["label"] = "decrease"
    # Plot parameters for change-point in mean vs in hsd
    pltkwargs["chghmean"] = copy.deepcopy(pltkwargs_dflt)
    pltkwargs["chghmean"]["plot"]["color"] = "#517DC0"
    pltkwargs["chghsd"] = copy.deepcopy(pltkwargs_dflt)
    pltkwargs["chghsd"]["plot"]["color"] = "#CE4A7F"
    return pltkwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["change-point-env", "random-walk-env"],
        help="Which experimental dataset / environment the subject belongs to.")
    parser.add_argument("data_file", help="Path to the subject's data file.")
    args = parser.parse_args()
    main(args)

"""
Simulate behavioral analyses using a given behavioral model and task configuration.
"""

from analyses import (run_lr_analysis, run_betwid_analysis, run_around_event_analysis,
    run_performance_analysis
    )
import argparse
import copy
import infer_pred_model
import os.path as op
import sim_data
from sub_data import SUB_KEY
import utils

def main(args):
    utils.plot.setup_mpl_style(fontsize=7)
    seed = args.seed
    task_config = utils.config.load_config_at_path_with_default_task_config(args.task_config_path)
    model_config = utils.config.load_config_at_path_with_default_model_config(args.model_config_path)
    config = utils.config.create_joint_config(task_config, model_config)
    model = infer_pred_model.get_model(config)
    if args.seq_data_path is not None:
        import json
        with open(args.seq_data_path, 'r') as f:
            seq_data = json.load(f)
    else:
        seq_data = None
    data = generate_and_simulate_data(task_config, model, seed, seq_data=seq_data)
    pltkwargs = prepare_plot_kwargs(config)
    # out_params = ["config"], [config["id"]]
    out_params = ["model"], [model_config['id']]
    model_text = model.name
    outdirname = "simulations"
    if "default" not in task_config["id"]:
        outdirname = op.join(outdirname, task_config["id"])
    is_cp_task = config["task"]["dynamics"]["hmean"]["type"] == "change_point"
    run_lr_analysis(data, config, out_params, pltkwargs=pltkwargs, text=model_text,
        analysis_level="group", outdirname=outdirname, do_count=True)
    run_betwid_analysis(data, config, out_params, pltkwargs=pltkwargs, text=model_text,
        analysis_level="group", outdirname=outdirname, do_ape_analysis=is_cp_task)
    if is_cp_task:
        run_around_event_analysis(data, config, out_params, pltkwargs=pltkwargs, text=model_text,
            analysis_level="group", outdirname=outdirname)
    run_performance_analysis(data, config, out_params, outdirname=outdirname)

def generate_and_simulate_data(task_config, model, seed, seq_data=None):
    nsub = task_config["nsub"]
    nseq = task_config["nseq"]
    nobs = task_config["nobs"]
    true_gen_model = infer_pred_model.InferPredGenerativeModel(task_config)
    data = sim_data.generate_experiment(true_gen_model, nsub, nseq, nobs, seed=seed,
        seq_data=seq_data, add_prev_hidstate=True, add_hidstate_change=True)
    sim_data.simulate_model_on_experiment(model, data,
        add=["bet", "prevbet", "ape", "lr", "plrgwid", "pincwid", "pdecwid", "reward"])
    return data

def prepare_plot_kwargs(config):
    # Default plot parameters
    plotkwargs = dict(marker='o', linestyle='-', ms=2, lw=1.5,
            color=utils.plot.BLACK_COLOR)
    barplotkwargs = dict(width=0.5, color=utils.plot.BLACK_COLOR)
    pltkwargs_dflt = {"plot": plotkwargs, "bar": barplotkwargs}
    pltkwargs = copy.deepcopy(pltkwargs_dflt)
    # Plot parameters for different levels of stochasticity or bet width
    hsd_levels = config["task"]["state_space"]["hsd"].get("values", [0, 1])
    betwid_levels = config["task"]["responses"]["widths"]
    hsd_labels = ["low", "high"]
    betwid_labels = ["small", "large"]
    for factor, values, labels in [("hsd", hsd_levels, hsd_labels), (f"prevhsd", hsd_levels, hsd_labels),
        ("hsd_bin", hsd_levels, hsd_labels), (f"prevhsd_bin", hsd_levels, hsd_labels),
        ("betwid", betwid_levels, betwid_labels), ("prevbetwid", betwid_levels, betwid_labels)]:
        pltkwargs[factor] = {v: copy.deepcopy(pltkwargs_dflt) for v in values}
        for i, v in enumerate(values):
            pltkwargs[factor][v]['plot']["color"] = f'{1/4 + 1/2 * i:.2f}'
            pltkwargs[factor][v]['plot']["label"] = labels[i]
            pltkwargs[factor][v]['bar']["color"] = f'{1/4 + 1/2 * i:.2f}'
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
    pltkwargs["chghmean"]["plot"]["color"] = f'{1/4:.2f}'
    pltkwargs["chghsd"] = copy.deepcopy(pltkwargs_dflt)
    pltkwargs["chghsd"]["plot"]["color"] = f'{1/4 + 1/2:.2f}'
    return pltkwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-tcp", "--task_config_path", default=None)
    parser.add_argument("-mcp", "--model_config_path", default=None)
    parser.add_argument("-s", "--seed", default=0, type=(int))
    parser.add_argument("-seq", "--seq_data_path", default=None)
    args = parser.parse_args()
    main(args)

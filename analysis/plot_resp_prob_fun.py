"""
Script to make an illustration of the models' response probability function in
paper figures.
"""

import argparse
import infer_pred_model
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import utils

def response_probability_fun(d):
    model = infer_pred_model.InferPredBehaviorModel({})
    return model.compute_update_probability(d,
        # parameters used in the main model configs
        fun_type="truncnorm", fun_params=[0, 8]
        )

def main(args):
    utils.plot.setup_mpl_style(fontsize=5)

    xmax = 61
    x = np.linspace(0, xmax, 1000)
    y = response_probability_fun(x)

    aspect_ratio = 1.5
    height = 0.8
    fig = plt.figure(figsize=(aspect_ratio * height, height))
    ax = fig.gca()
    ax.plot(x, y, color='black', clip_on=False)
    ax.set_xlabel('Distance (°)')
    ax.set_ylabel('Resp. probability')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 60])
    ax.xaxis.set_label_coords(0.5, -0.18)  # Centered, y same as tick labels
    ax.set_yticks([0, 1])
    output_dir = op.join(utils.output.DIR, "global")
    utils.output.create_dir_if_needed(output_dir)
    figpath = f'{output_dir}/model_response_probability_function.{args.ext}'
    utils.output.save_figure(fig, figpath, dpi=300)

# Plot the response probability function in a small figure
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-ext", default="pdf")
    args = parser.parse_args()
    main(args)
    
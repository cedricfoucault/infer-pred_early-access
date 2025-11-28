"""
Master script to reproduce all main results and figures reported in the paper.

This script sequentially runs all analysis modules for human behavior, model
behavior, human-model comparisons, example behavioral sequences, and
conceptual model illustrations. Running this script produces the full set
of results included in the paper.
"""

import argparse
import group_analyses
import group_model_analyses
import model_analyses
import plot_resp_prob_fun
import os.path as op
import sub_plot_sequences
import utils

def main():
    datasets = ["change-point-env", "random-walk-env"]
    model_config_paths = {
        "change-point-env": [
            "config/model/main/change-point.yaml",
            "config/model/main/random-walk.yaml",
            "config/model/variants/change-point_no-prob-resp.yaml",
            "config/model/variants/change-point_fixed-var-high.yaml",
            "config/model/variants/change-point_fixed-var-low.yaml",
        ],
        "random-walk-env": [
            "config/model/main/random-walk.yaml",
            "config/model/main/change-point.yaml",
            "config/model/variants/random-walk_no-prob-resp.yaml",
            "config/model/variants/random-walk_fixed-var-high.yaml",
            "config/model/variants/random-walk_fixed-var-low.yaml",
        ]
    }
    main_model_config_paths = ["config/model/main/change-point.yaml",
                               "config/model/main/random-walk.yaml"]
    
    for dataset in datasets:
        # Run group analyses on human data
        args_group = argparse.Namespace(
            dataset=dataset,
            ext="pdf"
        )
        group_analyses.main(args_group)

        # Run model analyses
        for model_config_path in model_config_paths[dataset]:
            args_model = argparse.Namespace(
                dataset=dataset,
                model_config_path=model_config_path,
                ext="pdf",
                skip_lr=(True if "fixed-var" in model_config_path else False),
                skip_lr_control=(True if "no-prob-resp" in model_config_path else False),
                skip_padlwid=(True if "no-prob-resp" in model_config_path else False),
                skip_perf=True,
                do_stats=False,
            )
            model_analyses.main(args_model)

        # Run group model comparison analyses
        args_group_model = argparse.Namespace(
            dataset=dataset,
            model_config_paths=main_model_config_paths,
            do_save_correlation_values=False
        )
        group_model_analyses.main(args_group_model)

        # Plot example sequences - Examples used in the paper
        examp_subs_blocks = {
            "change-point-env": ("5c4f5967aac8be0001716a65", 2),
            "random-walk-env": ("6575c7da000442e5a54f5e24", 14)
        }
        args_seq = argparse.Namespace(
            dataset=dataset,
            ext="pdf",
            sub=examp_subs_blocks[dataset][0],
            block=examp_subs_blocks[dataset][1]
        )
        sub_plot_sequences.main(args_seq)

    # Plot model response probability function
    args_resp_prob = argparse.Namespace(ext="pdf")
    plot_resp_prob_fun.main(args_resp_prob)

    # Plot example model inference
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    FRAMEWORK_DIR = REPO_ROOT / "modeling_framework"
    import sys
    sys.path.append(str(FRAMEWORK_DIR))
    import use_case_infer_pred
    output_dir = op.join(utils.output.DIR, "global", "examples")
    utils.output.create_dir_if_needed(output_dir)
    use_case_infer_pred.main(fpath=f"{output_dir}/example_model_inference.pdf")

if __name__ == '__main__':
    main()
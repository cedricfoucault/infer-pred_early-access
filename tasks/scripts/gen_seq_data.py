"""
Script to generate a sequence data file for the infer-pred task.
The data file can be loaded and used by the task code to present
corresponding stimuli.
"""

# Add the analysis directory of this repository to the python search path
# so that we can import the modules contained within that directory
import sys
import os.path as op
analysis_dir = op.realpath(op.join(__file__ , "../../../analysis"))
sys.path.append(analysis_dir)

import argparse
import infer_pred_model
import json
import sim_data
import utils

OUT_DIR = op.realpath(op.join(__file__ , "../../seq-data"))
JS_VARIABLE_NAME = "seqData"

def main(args):
    output_dir = OUT_DIR
    utils.output.create_dir_if_needed(OUT_DIR)
    task_config = utils.config.load_config_at_path_with_default_task_config(args.task_config_path)
    gen_model = infer_pred_model.InferPredGenerativeModel(task_config)

    # nsub = task_config["nsub"]
    nsub = 1
    nseq = args.nseq
    nobs = task_config["nobs"]
    dataframe = sim_data.generate_experiment(gen_model,
            nsub, nseq, nobs, seed=args.seed)

    # Convert data into a dictionary easy to work with in the task javascript code.
    data = {
        "nobs": nobs,
        "nseq": nseq,
    }
    seqs = []
    assert len(dataframe["blockIdx"].unique()) == nseq
    for _, seqData in dataframe.groupby("blockIdx"):
        seq = {
            "obs": seqData["obs"].values.tolist(),
            "hmean": seqData["hmean"].values.tolist(),
            "hsd": seqData["hsd"].values.tolist()
        }
        seqs.append(seq)
    data["seqs"] = seqs

    prefix = f"seq-data"
    # if (args.task_config_path is not None):
    #     prefix += f"_{task_config['id']}"
    outname = utils.output.name_with_params(prefix,
        ["config", "nseq", "seed"], [task_config["id"], args.nseq, args.seed])
    for ext in ["json"]:
    # for ext in ["json", "js"]:
    # for ext in ["js"]:
        if ext == "json":
            outstring = json.dumps(data)
        elif ext == "js":
            outstring = f"const {JS_VARIABLE_NAME} = {json.dumps(data)}"
        outpath = utils.output.get_path(output_dir, outname, ext)
        print("outpath", outpath)
        with open(outpath, 'w') as outfile:
            outfile.write(outstring)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-tcp", "--task_config_path", default=None)
    parser.add_argument("-s", "--seed", default=1, type=(int))
    parser.add_argument("-nseq", default=100, type=(int))
    args = parser.parse_args()
    main(args)

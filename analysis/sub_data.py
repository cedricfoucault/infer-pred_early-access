import json
import numpy as np
import os
import os.path as op
import pandas as pd
import utils

SUB_KEY = "subjectId"

def get_subject_files(subject_list_path):
    subject_list_df = pd.read_csv(subject_list_path)
    return subject_list_df["data_file"].to_list()

def load_subject_data(data_file, data_types=["blocks", "expEvents"]):
    # Load the main data file's JSON content
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract directory from the path stored in data_file
    data_dir = os.path.dirname(data_file)
    
    # Load the data for the files of the requested data_types that are pointed to
    # in the main data file.
    if "blocks" in data_types:
        data["blocks"] = []
        for file_name in data["blockDataFiles"]:
            file_path = op.join(data_dir, file_name)
            with open(file_path, 'r') as f:
                data["blocks"].append(json.load(f))
    
    if "blockEvents" in data_types:
        data["blockEvents"] = []
        for file_name in data["blockEventDataFiles"]:
            file_path = op.join(data_dir, file_name)
            with open(file_path, 'r') as f:
                data["blockEvents"].append(json.load(f)["events"])
    
    if "expEvents" in data_types:
        file_path = op.join(data_dir, data["expEventDataFile"])
        with open(file_path, 'r') as f:
            data["expEvents"] = json.load(f)["events"]

    return data

def make_subject_dataframe(data, config):
    return make_dataframe_for_subject_data_and_model(data, config, model=None)

def make_dataframe_for_subject_data_and_model(data, config, model=None):
    """
    If a model is given, the model's behavior replaces the subject's behavior
    in the constructed dataframe.
    """
    sub_id = data["subjectId"]
    n_trials = data["blocks"][0]["nTrials"]
    i_blocks_completed = [ev[1] for ev in data["expEvents"]["finishBlock"]]
    df = pd.concat((pd.DataFrame(data["blocks"][i]) for i in i_blocks_completed))
    cols = ["blockIdx", "trialIdx", "hmean", "hsd", "obs",  "betloc", "betwid", "score"]
    df = df[cols]
    df[SUB_KEY] = sub_id

    betwid_levels = config["task"]["responses"]["widths"]
    lrgwid = max(betwid_levels)

    for i in i_blocks_completed:
        filt = df["blockIdx"] == i
        obs = df.loc[filt, "obs"].to_numpy()
        if model is None:
            betloc = df.loc[filt, "betloc"].to_numpy()
            betwid = df.loc[filt, "betwid"].to_numpy()
        else:
            bet = model.get_prev_bet(model.compute_bet_given_obs(obs))
            betloc = bet[:, 0]
            betwid = bet[:, 1]
            df.loc[filt, "betloc"] = betloc
            df.loc[filt, "betwid"] = betwid
        # prediction error
        pe = utils.calc.subtract_angle(obs, betloc)
        df.loc[filt, "ape"] = np.abs(pe)
        # update
        updt_loc = utils.calc.subtract_angle(betloc[1:], betloc[:-1])
        # Use NaN for the update and learning rate following the last stimulus,
        # since the subject has not reported a response after seeing that stimulus
        updt_loc = np.append(updt_loc, np.nan)
        df.loc[filt, "updt_loc"] = updt_loc
        # learning rate
        lr = utils.calc.clean_learning_rate_with_update_and_pred_error(
                    updt_loc[:-1], pe[:-1])
        lr = np.append(lr, np.nan)
        # Exclude the first trial (using NaN) from the considered learning
        # rates, since the participant's had not yet seen a stimulus and their
        # prediction was arbitrary
        lr = np.append(np.nan, lr[1:])
        df.loc[filt, "lr"] = lr
        has_udpt_loc = np.where(~np.isnan(updt_loc), ~np.isclose(updt_loc, 0), np.nan)
        df.loc[filt, "p_updt_loc"] = has_udpt_loc.astype(float)
        df.loc[filt, "lr_exclude-no-update"] = np.where(has_udpt_loc, lr, np.nan)
        # plrgwid, pincwid, pdecwid
        islrgwid = np.isclose(betwid, lrgwid)
        df.loc[filt, "plrgwid"] = islrgwid.astype(float)
        df.loc[filt, "nextplrgwid"] = np.append(islrgwid[1:].astype(float), np.nan)
        updt_wid = betwid[1:] - betwid[:-1]
        updt_wid = np.append(updt_wid, np.nan)
        df.loc[filt, "updt_wid"] = updt_wid
        df.loc[filt, "pincwid"] = np.where((~islrgwid) & (~np.isnan(updt_wid)),
            updt_wid > 0, np.nan)
        df.loc[filt, "pdecwid"] = np.where((islrgwid) & (~np.isnan(updt_wid)),
            updt_wid < 0, np.nan)
        # change in hmean and hsd
        hmean = df.loc[filt, "hmean"].to_numpy()
        hsd = df.loc[filt, "hsd"].to_numpy()
        hmeanchg = np.append(np.nan, utils.calc.subtract_angle(hmean[1:], hmean[:-1]))
        hsdchg = np.append(np.nan, hsd[1:] - hsd[:-1])
        prevhsd = np.append(np.nan, hsd[:-1])
        df.loc[filt, "hmeanchg"] = hmeanchg
        df.loc[filt, "hsdchg"] = hsdchg
        df.loc[filt, "prevhsd"] = prevhsd
        # whether the subject caught the beam
        df.loc[filt, "did_catch"] = (np.abs(pe) < (betwid / 2))
        # mean absolute error between paddle location and true previous hmean
        df.loc[filt, "mae"] = np.append(np.nan,
            np.abs(utils.calc.subtract_angle(betloc[1:], hmean[:-1])))

        # Reward in points per trial (note that this should be consistent with
        # the total 'score' entry for the block, i.e. score should be the sum of
        # rewards over trials plus the initial score participants start from)
        reward_table = config["task"]["responses"]["reward_table"]
        smlwid = min(betwid_levels)
        df.loc[filt, "reward"] = 0
        df.loc[filt, "reward"] += np.where(df.loc[filt, "did_catch"] & ~islrgwid,
            reward_table["catch_per_width"][smlwid], 0)
        df.loc[filt, "reward"] += np.where(df.loc[filt, "did_catch"] & islrgwid,
            reward_table["catch_per_width"][lrgwid], 0)
        df.loc[filt, "reward"] += np.where(~df.loc[filt, "did_catch"] & ~islrgwid,
            reward_table["miss_per_width"][smlwid], 0)
        df.loc[filt, "reward"] += np.where(~df.loc[filt, "did_catch"] & islrgwid,
            reward_table["miss_per_width"][lrgwid], 0)

    return df
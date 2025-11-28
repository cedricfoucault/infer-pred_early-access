from analyses import SUB_KEY
import numpy as np
import pandas as pd
import utils

def generate_experiment(gen_model, nsub, nseq, nobs, seed=0,
    seq_data=None,
    add_prev_hidstate=False, hidstate_initial_val=[np.nan, np.nan],
    add_hidstate_change=False, hidstate_change_initial_val=[np.nan, np.nan]):
    np.random.seed(seed)
    n_rows = nsub * nseq * nobs
    n_rows_per_sub = nseq * nobs
    data = {
        "nsub": nsub,
        "nseq": nseq,
        "nobs": nobs,
        SUB_KEY: np.empty(n_rows, dtype=int),
        "blockIdx": np.empty(n_rows, dtype=int),
        "t": np.empty(n_rows, dtype=int),
        "obs": np.empty(n_rows),
        "hmean": np.empty(n_rows),
        "hsd": np.empty(n_rows)
    }
    if add_prev_hidstate:
        data["prevhmean"] = np.empty(n_rows)
        data["prevhsd"] = np.empty(n_rows)
    if add_hidstate_change:
        data["hmeanchg"] = np.empty(n_rows)
        data["hsdchg"] = np.empty(n_rows)
    for i_sub in range(nsub):
        if seq_data is not None:
            seq_indices = np.random.choice(len(seq_data["seqs"]), size=nseq)
        for i_seq in range(nseq):
            start = n_rows_per_sub * i_sub + nobs * i_seq
            if seq_data is not None:
                seq = seq_data["seqs"][seq_indices[i_seq]]
                obs = np.array(seq["obs"])
                hmean = np.array(seq["hmean"])
                hsd = np.array(seq["hsd"])
            else:
                obs, hid = gen_model.generate(nobs)
                hmean = hid[:, 0]
                hsd = hid[:, 1]
            data[SUB_KEY][start:start+nobs] = i_sub+1
            data["blockIdx"][start:start+nobs] = i_seq+1
            data["t"][start:start+nobs] = np.arange(nobs)
            data["obs"][start:start+nobs] = obs
            data["hmean"][start:start+nobs] = hmean
            data["hsd"][start:start+nobs] = hsd
            if add_prev_hidstate:
                data["prevhmean"][start] = hidstate_initial_val[0]
                data["prevhsd"][start] = hidstate_initial_val[1]
                data["prevhmean"][start+1:start+nobs] = hmean[:-1]
                data["prevhsd"][start+1:start+nobs] = hsd[:-1]
            if add_hidstate_change:
                data["hmeanchg"][start] = hidstate_change_initial_val[0]
                data["hsdchg"][start] = hidstate_change_initial_val[1]
                data["hmeanchg"][start+1:start+nobs] = utils.calc.subtract_angle(hmean[1:], hmean[:-1])
                data["hsdchg"][start+1:start+nobs] = hsd[1:] - hsd[:-1]

    return pd.DataFrame(data)

def simulate_model_on_experiment(model, data, add=["bet"]):
    """
    Simulate the behavioral model on the given experiment data, i.e. generate
    the sequence of bets produced by the model for each observation sequence
    in the data. The bet responses (location and width) are added to the
    dataframe as new columns.

    Parameters
    ----------
    data: pandas DataFrame
        Dataframe containing the data of the experiment. The data should
        contain the columns SUB_KEY, "blockIdx", "t", "obs", "hmean", "hsd".
    add: list of strings.
        Possible values: "bet", "nextbet", "updt", "pe", "ape", "lr",
        "plrgwid", "pincwid", "pdecwid".
    """
    # Determine what values need to be computed
    compute = {}
    compute["bet"] = True
    compute["nextbet"] = (("nextbet" in add))
    compute["updt"] = (("updt" in add) or ("lr" in add)
        or ("pincwid" in add) or ("pdecwid" in add))
    compute["pe"] = (("pe" in add)
        or ("ape" in add) or ("lr" in add) or ("reward" in add))
    compute["ape"] = ("ape" in add)
    compute["lr"] = ("lr" in add)
    # Initialize arrays for values that need to be computed
    nsub = data[SUB_KEY].nunique()
    nseq = data["blockIdx"].nunique()
    nobs = data["t"].nunique()
    betloc_col = np.empty((nsub * nseq * nobs))
    betwid_col = np.empty((nsub * nseq * nobs))
    if compute["nextbet"]:
        next_betloc_col = np.empty((nsub * nseq * nobs))
        next_betwid_col = np.empty((nsub * nseq * nobs))
    if compute["updt"]:
        update_loc_col = np.empty((nsub * nseq * nobs))
        update_wid_col = np.empty((nsub * nseq * nobs))
    if compute["pe"]:
        pred_error_col = np.empty((nsub * nseq * nobs))
    if compute["ape"]:
        abs_pred_error_col = np.empty((nsub * nseq * nobs))
    if compute["lr"]:
        lr_col = np.empty((nsub * nseq * nobs))
    # Compute bet responses and values computed from them
    for i_sub in range(nsub):
        for i_seq in range(nseq):
            start = nobs * (nseq * i_sub + i_seq)
            obs = data.loc[(data[SUB_KEY] == i_sub+1)
                & (data["blockIdx"] == i_seq+1), "obs"].values
            next_bet = model.compute_bet_given_obs(obs)
            bet = model.get_prev_bet(next_bet)
            betloc_col[start:start+nobs] = bet[:, 0]
            betwid_col[start:start+nobs] = bet[:, 1]
            if compute["nextbet"]:
                next_betloc_col[start:start+nobs] = next_bet[:, 0]
                next_betwid_col[start:start+nobs] = next_bet[:, 1]
            if compute["updt"]:
                update_loc_col[start:start+nobs] = np.append(
                    utils.calc.subtract_angle(bet[1:, 0], bet[:-1, 0]),
                    np.nan)
                update_wid_col[start:start+nobs] = np.append(
                    bet[1:, 1] - bet[:-1, 1], np.nan)
            if compute["pe"]:
                pred_error_col[start:start+nobs] = utils.calc.subtract_angle(
                    obs, bet[:, 0])
            if compute["ape"]:
                abs_pred_error_col[start:start+nobs] = np.abs(pred_error_col[start:start+nobs])
            if compute["lr"]:
                lr_col[start:start+nobs] = np.append(
                    utils.calc.clean_learning_rate_with_update_and_pred_error(
                        update_loc_col[start:start+nobs-1], pred_error_col[start:start+nobs-1]),
                    np.nan)
    # Add computed values to the dataframe
    if "bet" in add:
        data["betloc"] = betloc_col
        data["betwid"] = betwid_col
    if "nextbet" in add:
        data["nextbetloc"] = next_betloc_col
        data["nextbetwid"] = next_betwid_col
    if "updt" in add:
        data["updtloc"] = update_loc_col
        data["updtwid"] = update_wid_col
    if "pe" in add:
        data["pe"] = pred_error_col
    if "ape" in add:
        data["ape"] = abs_pred_error_col
    if "lr" in add:
        data["lr"] = lr_col
    betwid_levels = model.config["task"]["responses"]["widths"]
    lrgwid = max(betwid_levels)
    smlwid = min(betwid_levels)
    islrgwid = np.isclose(betwid_col, lrgwid)
    if "plrgwid" in add:
        data["plrgwid"] = islrgwid.astype(float)
        data["nextplrgwid"] = np.append(islrgwid[1:].astype(float), np.nan)
    if "pincwid" in add:
        data["pincwid"] = np.where((~islrgwid) & (~np.isnan(update_wid_col)),
            update_wid_col > 0, np.nan)
    if "pdecwid" in add:
        data["pdecwid"] = np.where(islrgwid & (~np.isnan(update_wid_col)),
            update_wid_col < 0, np.nan)
    if "reward" in add:
        reward_table = model.config["task"]["responses"]["reward_table"]
        did_catch = np.abs(pred_error_col) < betwid_col / 2
        data["reward"] = 0
        data["reward"] += np.where(did_catch & ~islrgwid, reward_table["catch_per_width"][smlwid], 0)
        data["reward"] += np.where(did_catch & islrgwid, reward_table["catch_per_width"][lrgwid], 0)
        data["reward"] += np.where(~did_catch & ~islrgwid, reward_table["miss_per_width"][smlwid], 0)
        data["reward"] += np.where(~did_catch & islrgwid, reward_table["miss_per_width"][lrgwid], 0)


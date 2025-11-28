import argparse
import numpy as np
import sub_data
import utils

SMALL_BETWID = 60
LARGE_BETWID = 120

N_POINTS_PER_POUND = 2500

def str_from_ms(ms, include_s=True, include_ms=False, sep=" "):
    ms = int(ms)
    one_s_ms = 1000
    one_min_ms = (60 * one_s_ms)
    t_min = ms // one_min_ms
    t_rem_s = (ms % one_min_ms) // one_s_ms
    t_rem_ms = ms % one_s_ms
    components = []
    if t_min > 0:
        components += [f"{t_min:d}min"]
    if t_rem_s > 0 and include_s:
        components += [f"{t_rem_s:d}s"]
    if t_rem_ms > 0 and include_ms:
        components += [f"{t_rem_ms:d}ms"]
    return sep.join(components)

def main(args):
    data = sub_data.load_subject_data(args.data_file)

    # Find indices and number of blocks that the subject completed
    i_blocks_completed = [ev[1] for ev in data["expEvents"]["finishBlock"]]
    blocks_completed = [data["blocks"][i] for i in i_blocks_completed]
    
    # Find total score and average per block
    total_score = data["totalScore"]
    avg_score = np.mean([b["score"] for b in blocks_completed])

    # Compute money bonus
    if "totalPounds" in data:
        money_bonus = data["totalPounds"]
    else:
        money_bonus = total_score / N_POINTS_PER_POUND

    # Compute average performance across blocks
    prop_catch = {k: [] for k in ["smallwid", "largewid"]}
    prop_updt = {k: [] for k in ["loc", "wid"]}
    for block in blocks_completed:
        # Proportion of beams caught
        nTrials = block["nTrials"]
        betloc = np.array(block["betloc"][:nTrials])
        betwid = np.array(block["betwid"][:nTrials])
        obs = block["obs"][:nTrials]
        angle_delta = utils.calc.subtract_angle(betloc[:nTrials],
            obs[:nTrials])
        did_catch = np.abs(angle_delta) < (betwid / 2)
        did_catch_smallwid = did_catch & np.isclose(betwid, SMALL_BETWID)
        did_catch_largewid = did_catch & np.isclose(betwid, LARGE_BETWID)
        assert np.all(did_catch == (did_catch_smallwid | did_catch_largewid))
        prop_catch["smallwid"] += [np.mean(did_catch_smallwid)]
        prop_catch["largewid"] += [np.mean(did_catch_largewid)]
        # Frequency of trials at which subject updated their paddle location/width
        locdelta = utils.calc.subtract_angle(betloc[:-1], betloc[1:])
        widdelta = (betwid[:-1] - betwid[1:])
        did_updt_loc = ~np.isclose(locdelta, 0)
        did_updt_wid = ~np.isclose(widdelta, 0)
        prop_updt["loc"] += [np.mean(did_updt_loc)]
        prop_updt["wid"] += [np.mean(did_updt_wid)]

    # Compute completion times
    t_start_consent = data["expEvents"]["startConsent"][0][0]
    t_start_instructions = data["expEvents"]["startInstructions"][0][0]
    t_start_fstblock = data["expEvents"]["startBlock"][0][0]
    t_end_lstblock = data["expEvents"]["finishBlock"][-1][0]
    delta_t_each_block = [data["expEvents"]["finishBlock"][i][0]
        - data["expEvents"]["startBlock"][i][0] for i in i_blocks_completed]
    avg_time_per_block = np.mean(delta_t_each_block)

    avg_catch = {k: np.mean(np.array(prop)) for k, prop in prop_catch.items()}
    avg_updt = {k: np.mean(np.array(prop)) for k, prop in prop_updt.items()}

    table = []
    table += [f"Blocks completed: {len(i_blocks_completed)}"]
    table += [f"Score (total | avg.): {total_score} | {avg_score}"]
    table += [f"Beams caught [all (small | large)]: " +
        f"""{(avg_catch["smallwid"]+avg_catch["largewid"])*100:.1f}% """ +
        f"""({avg_catch["smallwid"]*100:.1f}% | {avg_catch["largewid"]*100:.1f}%)"""
        ]
    table += [f"Update frequency (location | width): " +
        f"""{avg_updt["loc"]*100:.1f}% | {avg_updt["wid"]*100:.1f}%"""]
    table += [f"Time taken [total (consent | instructions | task)]: " +
        f"""{str_from_ms(t_end_lstblock - t_start_consent)} """ +
        f"""({str_from_ms(t_start_instructions - t_start_consent)} | """ +
        f"""{str_from_ms(t_start_fstblock - t_start_instructions)} | """ +
        f"""{str_from_ms(t_end_lstblock - t_start_fstblock)})"""]
    table += [f"Time taken per block (avg.): {str_from_ms(avg_time_per_block)}"]
    table += [f"Money bonus: £{money_bonus:.2f}"]

    # Print and save table to file
    print("\n".join(table))
    if args.do_save:
        output_file = args.output_file
        if output_file is None:
            output_file = args.data_file.replace(".json", "_summary-table.txt")
        with open(output_file, 'w') as f:
            print("\n".join(table), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    parser.add_argument("-o", "--output_file", default=None)
    parser.add_argument("--do_save", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
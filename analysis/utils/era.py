"""
Module for event-related analyses
"""

import numpy as np

def get_window_around_cp(w_min=-2, w_max=15):
    return np.arange(w_max - w_min + 1) + w_min

def aggregate_values_in_window_around_cp(values, window_around_cp, change_vals,
    discard_nan=True,
    change_sign=None):
    """
    Aggregate the values of all trials that fall into the provided window
    around a change point by their distance to the change point.
    The 'change_vals' array indicate when the change points occur.
    Returns a list of lists of values where each list corresponds to one distance
    to the change point within the provided window.
    """
    assert values.ndim == 1
    assert change_vals.ndim == 1
    assert change_vals.shape[0] == values.shape[0]
    n_trials = values.shape[0]
    win_around_cp_min = window_around_cp.min()
    win_around_cp_max = window_around_cp.max()
    values_in_win_around_cp = [[] for t in window_around_cp]
    # Iterate over change points
    did_changes = ~np.isclose(change_vals, 0) & ~np.isnan(change_vals)
    if change_sign and (change_sign > 0):
        did_changes = did_changes & (change_vals > 0)
    elif change_sign and (change_sign < 0):
        did_changes = did_changes & (change_vals < 0)
    cp_trial_indices = np.arange(n_trials)[did_changes]
    dist_between_cps = np.diff(cp_trial_indices)
    for i_cp, trial_cp in enumerate(cp_trial_indices):
        # Compute the window of trials to take around this change point
        # as the intersection of
        # - the defined window to aggregate over
        # - the window of trials between the previous and the next change point
        # (or starting at the first trial of the session if there is no previous c.p.,
        # and ending at the last trial of the session if there is no next c.p.)
        if i_cp > 0:
            dist_to_prev_cp = dist_between_cps[i_cp-1]
            trial_win_min = max(trial_cp+win_around_cp_min,
                trial_cp-dist_to_prev_cp+1)
        else:
            trial_win_min = max(trial_cp+win_around_cp_min, 0)
        if i_cp < len(dist_between_cps):
            dist_to_next_cp = dist_between_cps[i_cp]
            trial_win_max = min(trial_cp+win_around_cp_max,
                trial_cp+dist_to_next_cp-1)
        else:
            trial_win_max = min(trial_cp+win_around_cp_max, n_trials-1)
        for trial in range(trial_win_min, trial_win_max+1):
            v = values[trial]
            if not (discard_nan and np.isnan(v)):
                t_from_cp = trial-trial_cp
                i_win = t_from_cp - win_around_cp_min
                values_in_win_around_cp[i_win] += [v]
    return values_in_win_around_cp

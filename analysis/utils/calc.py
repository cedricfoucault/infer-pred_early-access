import numpy as np
import scipy

def shift_back(v, v0):
    """Shift the sequence back in time by one time step and use the provided
    value as first element of the new sequence. This function works on 1-D or
    N-D arrays assuming that the time axis corresponds to the first dimension
    of the array."""
    return np.insert(v, 0, v0, axis=0)[:-1, ...]

def subtract_angle(a, b, period=360):
   return ((a - b) + period/2) % period - period/2

def circmeandegrees(angles, axis=None):
    return np.rad2deg(scipy.stats.circmean(np.deg2rad(angles), axis=axis))

def align_angle_series_to(series_to_align, series_to_align_to,
    period=360):
    """
    Align the values of the given series to those of the other series
    by shifting the values by a multiple of the period.
    This could introduce discontinuities in the aligned series
    (i.e., jumps greater than period/2).
    """
    assert (series_to_align.ndim == 1
        and (series_to_align.shape == series_to_align_to.shape))
    aligned_series = series_to_align.copy()
    for i in range(aligned_series.shape[0]):
        diff = aligned_series[i] - series_to_align_to[i]
        if diff >= period/2:
            aligned_series[i] -= period * (
                (diff+period/2) // period)
        if diff <= -period/2:
            aligned_series[i] += period * (
                (-diff+period/2) // period)
    return aligned_series   

def angular_learning_rate(x_t, v_t, v_prior=0, lr_min=-np.inf, lr_max=+np.inf,
    abs_pred_error_min=0.):
    v_tminus1 = np.insert(v_t, 0, v_prior)[:-1]
    delta_t = subtract_angle(x_t, v_tminus1)
    update_t = subtract_angle(v_t, v_tminus1)
    return learning_rate_with_update_and_pred_error(update_t, delta_t,
        lr_min=lr_min, lr_max=lr_max, abs_pred_error_min=abs_pred_error_min)

def learning_rate_with_update_and_pred_error(update, pred_error,
    lr_min=-np.inf, lr_max=+np.inf, abs_pred_error_min=0.):
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_lr = update / pred_error
    lr_is_outside_bounds = (np.isclose(pred_error, 0)
                           | (raw_lr > lr_max)
                           | (raw_lr < lr_min)
                           | (np.abs(pred_error) < abs_pred_error_min))
    return np.where(lr_is_outside_bounds, np.nan, raw_lr)

def angular_update(v_t, v_prior=0):
    v_tminus1 = np.insert(v_t, 0, v_prior)[:-1]
    return subtract_angle(v_t, v_tminus1)

def did_update_angle(v_t, v_prior = 0):
    v_tminus1 = np.insert(v_t, 0, v_prior)[:-1]
    update_t = subtract_angle(v_t, v_tminus1)
    return ~np.isclose(update_t, 0)

# Here we use the values that were used in the Ada-Learn study from Foucault & Meyniel (2024).
# These values were determined from the data of the pilot Ada-Pos study of that paper,
# taking the 95% percentile of the empirical distribution of subjects' apparent learning rate.
# For reference, the 95% percentiles in the Vaghi et al. study (2016) across the control and patient groups
# was +1.3 on the upper bound (the lower bound was 0 due to the way learning rates
# were computed in that study: they could not be negative).
CLEAN_LR_MAX = +1.3
CLEAN_LR_MIN = -0.6

def clean_angular_learning_rate(x_t, v_t, v_prior=0):
    return angular_learning_rate(x_t, v_t, v_prior=v_prior,
        lr_min=CLEAN_LR_MIN, lr_max=CLEAN_LR_MAX)

def clean_learning_rate_with_update_and_pred_error(update, pred_error):
    return learning_rate_with_update_and_pred_error(update, pred_error,
    lr_min=CLEAN_LR_MIN, lr_max=CLEAN_LR_MAX)

def compute_updates_from_seq(x_t, v_t, v_prior,
    update_measure="lr",
    exclude_no_update=False):
    """
    Compute prediction updates measures for each observation in the given sequence.

    Parameters
    ----------
    x_t: 1-D array
        the sequence of observations (stimulus locations)
    v_t: 1-D array
        the sequence of predictions after receiving the observation x_t
        for the same time step.
    v_prior: scalar
        the location of the prediction prior to receiving any observations.

    Returns
    -------
    1-D array representing the update values for the given update measure.
    """
    if update_measure is None or update_measure == "lr":
        vals = clean_angular_learning_rate(x_t, v_t, v_prior=v_prior)
    elif update_measure == "updtmgn":
        vals = np.abs(angular_update(v_t, v_prior=v_prior))
    elif update_measure == "updtfreq":
        vals = (did_update_angle(v_t, v_prior=v_prior)).astype(float)
    else:
        assert False, f"update_measure={update_measure} not among the available options"

    if exclude_no_update:
        did_not_update = ~did_update_angle(v_t, v_prior=v_prior)
        vals = np.where(did_not_update, np.nan, vals)

    return vals


def get_deltarule_estimates(x_t, alpha=0.3, v_prior=0):
    assert x_t.ndim == 1
    v_t = np.empty_like(x_t)
    for t in range(x_t.shape[0]):
        v_tminus1 = v_t[t-1] if t > 0 else v_prior
        v_t[t] = v_tminus1 + alpha * (x_t[t] - v_tminus1)
    return v_t


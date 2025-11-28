"""Case study 4 variant: jointly inferring mean, stochasticity, and volatility.

This reproduces the task structure from Weber, Hunt, et al. (in prep.) but
extends use_case_4 by treating the volatility level as a latent,
categorical third dimension of the hidden state.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import bayeslearner.plotutils as plut

# To get reproducible results, initialize the seed of the random number generator
np.random.seed(0)

#
# Parameters of the continuous inference task from Weber, Hunt, et al., in prep.
#
# Ref: Source code for the generation of the sequence in this task:
# github.com:amyxli/coins-meg_meg-analysis/experiment/ses-2-meg/stimgen/generateMainSessionVaJump.m
#
# True stochasticity levels
coins_meg_stochasticity_sds = [10, 20, 30]
# Mean inter-stimulus interval (ISIs).
# Here we computed it empirically by generating the ISIs as in the original source
# code and computing the mean from a large sample of generated ISIs.
coins_meg_samp_isi = scipy.stats.expon.rvs(scale=0.3, size=int(1e4))
coins_meg_samp_isi = coins_meg_samp_isi[(coins_meg_samp_isi >= 0.1) & (coins_meg_samp_isi <= 1.0)]
coins_meg_mean_isi = coins_meg_samp_isi.mean()
# Duration of one block (i.e. one sequence).
coins_meg_block_duration = 3*60
# Mean number of stimuli per sequence
coins_meg_mean_num_stimuli = coins_meg_block_duration / coins_meg_mean_isi
# Mean duration between two change points in hidden mean (one for each volatility condition)
coins_meg_hmean_mean_duration = [15, 5]
# Mean duration between two change points in hidden s.d.
coins_meg_hsd_mean_duration = 10

#
# Parameters of the generative model
#

vmin = 0
vmax = 360
h_mean_step = 1 # 1 , 1/3
h_sd_vals = coins_meg_stochasticity_sds
n_t = round(coins_meg_mean_num_stimuli)
p_cp_means = [coins_meg_mean_isi / d for d in coins_meg_hmean_mean_duration]
p_cp_sd = coins_meg_mean_isi / coins_meg_hsd_mean_duration

#
# Define the generative model and create the corresponding Bayes learner model
#

hidden_mean_space = bayeslearner.ContinuousUnidimensionalSpace(
        min_value=vmin, max_value=vmax, step=h_mean_step, include_min=True, include_max=False,
        is_angular=True, # /!\ It is important to mark the space as angular
        # for subsequent computations on this space to be handled appropriately
        )
hidden_sd_space = bayeslearner.DiscreteUnidimensionalSpace(h_sd_vals)
hidden_vol_space = bayeslearner.DiscreteUnidimensionalSpace([0, 1]) # here the
# volatility condition is treated as a categorical variable, its scalar value is
# meaningless
hidden_state_space = bayeslearner.MultimensionalSpace(
    # This defines a 2D space for the hidden state.
    # Dimension 1 and 2 of the hidden state will represent
    # the mean and the sd of the  normal distribution generating the observations.
    [hidden_mean_space, hidden_sd_space, hidden_vol_space])
init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
obs_dist = bayeslearner.circnorm_dist(
    lambda h: dict(mu=h[..., 0], sigma=h[..., 1]))

# The dynamics distribution here is specific to the continuous inference study.
# In that study, there was a fixed set of change amounts that the mean could
# undergo when a change point occured.
# This defines the possible change amounts as ratio of the generative SD as was
# done to generate the stimulus sequences in the study.
allowed_hmean_change_sd_ratio = np.array([-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3])
# This defines the change amounts in degrees that can possibly occur across the
# experiment, considering all possible levels of SD.
allowed_hmean_change_amounts = np.sort(np.concatenate([
    allowed_hmean_change_sd_ratio * sd for sd in h_sd_vals]))
# Here, for simplicity, we define the dynamics distribution such that the possible
# change amounts for the mean do not depend on the current SD, but we could
# also define them such that they depend on the current SD by considering
# the two dimensions of the state jointly in the dynamics distribution matrix.
def is_allowed_change_mean(i_from, i_to):
    diff = bayeslearner.utils.subtract_angle(hidden_mean_space.values[i_to], hidden_mean_space.values[i_from])
    # diff = hidden_mean_space.values[i_to] - hidden_mean_space.values[i_from]
    return np.any(np.isclose(diff, allowed_hmean_change_amounts))

hmean_dynamics_dist_per_vol = [bayeslearner.change_point_dynamics_dist(
        hidden_mean_space.n_values, p_cp_mean,
        is_allowed_change_fun=is_allowed_change_mean) for p_cp_mean in p_cp_means]
hsd_dynamics_dist = bayeslearner.change_point_dynamics_dist(
        hidden_sd_space.n_values, p_cp_sd)
def dynamics_p_fun(h_from, h_to, i_from_ND, i_to_ND):
    vol = h_from[2]
    if vol != h_to[2]:
        # p=0 when the volatility level is not the same across the two states,
        # since we assume the volatility level to be static.
        p = 0
    else:
        # Since the changes in hsd and hmean are independent, the probability
        # for the given change in hsd and hmean is given by the product of the
        # probabilities of the changes in each of the two.
        p = (hmean_dynamics_dist_per_vol[vol][i_to_ND[0], i_from_ND[0]]
            * hsd_dynamics_dist[i_to_ND[1], i_from_ND[1]])
    return p
state_dynamics_dist = bayeslearner.construct_dynamics_dist(
    hidden_state_space, p_fun=dynamics_p_fun)

gen_model = bayeslearner.GenerativeModel(hidden_state_space,
        init_state_dist, state_dynamics_dist,
        obs_dist)
blearner = bayeslearner.BayesLearner(gen_model)

labels_dict = plut.LABELS_DICT.copy()
labels_dict.update({
    "hid": "True hidden value",
    "post_mean": "Posterior mean",
    "post_mode": "Posterior mode",
})

for seed in [0, 1]:
    np.random.seed(seed)
    #
    # Generate a sequence of observations and hidden states using the generative model
    #
    obs, hid = gen_model.generate(n_t)
    hid_means = hid[:, 0]
    hid_sds = hid[:, 1]
    hid_vols = hid[:, 2]

    #
    # Run the Bayes learner model inference on that sequence.
    #
    post_dists_dict = blearner.compute_post_dists_dict(
        obs, keys=['prob', 'margprob', 'mean', 'mode'])
    post_dists_marg_hmean = post_dists_dict['margprob'][0]
    post_dists_marg_hsd = post_dists_dict['margprob'][1]
    post_dists_marg_hvol = post_dists_dict['margprob'][2]
    post_mean_hmean = post_dists_dict['mean'][:, 0]
    post_mean_hsd = post_dists_dict['mean'][:, 1]
    post_mean_hvol = post_dists_dict['mean'][:, 2]
    post_mode_hmean = post_dists_dict['mode'][:, 0]
    post_mode_hsd = post_dists_dict['mode'][:, 1]
    post_mode_hvol = post_dists_dict['mode'][:, 2]

    #
    # Plot the results
    #
    # Plot the generated sequence and the Bayes learner inference
    # for the two dimensions of the hidden state
    plut.setup_mpl_style()
    t = np.arange(n_t)+1
    fig, axes = plt.subplots(figsize=(plut.A4_PAPER_CONTENT_WIDTH,
        plut.DEFAULT_HEIGHT * 3),
        nrows=3)
    ax = axes[0]
    ax.set_ylabel("Hidden mean")
    # title = f"Hidden mean change-point probability: {p_cp_mean:.3f}"
    # ax.set_title(title)
    plut.plot_sequence_on_ax(ax, t, obs, hid_means, hidden_mean_space,
        post_dists_marg_hmean, post_mean_hmean, post_mode_hmean,
        labels_dict=labels_dict)
    ax.set_ylim(vmin, vmax)
    ax = axes[1]
    ax.set_ylabel("Hidden s.d.")
    plut.plot_sequence_on_ax(ax, t, None, hid_sds, hidden_sd_space,
            post_dists_marg_hsd, post_mean_hsd, post_mode_hsd,
            labels_dict=labels_dict)
    ax = axes[2]
    ax.set_ylabel("Hidden vol.")
    plut.plot_sequence_on_ax(ax, t, None, hid_vols, hidden_vol_space,
            post_dists_marg_hvol, post_mean_hvol, post_mode_hvol,
            labels_dict=labels_dict)

    fprefix = plut.outpath_prefix_for_script(__file__)
    fpath = f"{fprefix}_seed-{seed}.png"
    plut.save_figure(fig, fpath)

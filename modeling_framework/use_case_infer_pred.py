"""
Case study: Infer-Pred study, change-point environment.

Similar environment structure as use case 4.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import bayeslearner.plotutils as plut

def main(fpath=None):
    # To get reproducible results, initialize the seed of the random number generator
    np.random.seed(0)

    #
    # Parameters of the generative model
    #

    vmin = 0
    vmax = 360
    h_mean_step = 1
    h_sd_vals = [10, 30]
    n_t = 100
    p_cp_mean = 0.0625
    p_cp_sd = 0.0625

    #
    # Define the generative model and create the corresponding Bayes learner model
    #

    hidden_mean_space = bayeslearner.ContinuousUnidimensionalSpace(
            min_value=vmin, max_value=vmax, step=h_mean_step, include_min=True, include_max=False,
            is_angular=True, # /!\ It is important to mark the space as angular
            # for subsequent computations on this space to be handled appropriately
            )
    hidden_sd_space = bayeslearner.DiscreteUnidimensionalSpace(h_sd_vals)
    hidden_state_space = bayeslearner.MultimensionalSpace(
        # This defines a 2D space for the hidden state.
        # Dimension 1 and 2 of the hidden state will represent
        # the mean and the sd of the  normal distribution generating the observations.
        [hidden_mean_space, hidden_sd_space])
    init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
    obs_dist = bayeslearner.circnorm_dist(
        lambda h: dict(mu=h[..., 0], sigma=h[..., 1]))
    # Define the state dynamics distribution
    min_change_amount = 90
    max_change_amount = np.inf
    def is_allowed_change_mean(i_from, i_to):
        absdiff = np.abs(bayeslearner.utils.subtract_angle(
            hidden_mean_space.values[i_to], hidden_mean_space.values[i_from]))
        return ((absdiff >= min_change_amount)
            and (absdiff <= max_change_amount))
    hmean_dynamics_dist = bayeslearner.change_point_dynamics_dist(
        hidden_mean_space.n_values, p_cp_mean,
        is_allowed_change_fun=is_allowed_change_mean)
    hsd_dynamics_dist = bayeslearner.change_point_dynamics_dist(
        hidden_sd_space.n_values, p_cp_sd)
    def dynamics_p_fun(h_from, h_to, i_from_ND, i_to_ND):
        # Since the changes in hsd and hmean are independent, the probability
        # for the given change in hsd and hmean is given by the product of the
        # probabilities of the changes in each of the two.
        p = (hmean_dynamics_dist[i_to_ND[0], i_from_ND[0]]
            * hsd_dynamics_dist[i_to_ND[1], i_from_ND[1]])
        return p
    state_dynamics_dist = bayeslearner.construct_dynamics_dist(
        hidden_state_space, p_fun=dynamics_p_fun)
        
        
    gen_model = bayeslearner.GenerativeModel(hidden_state_space,
            init_state_dist, state_dynamics_dist,
            obs_dist)
    blearner = bayeslearner.BayesLearner(gen_model)

    #
    # Generate a sequence of observations and hidden states using the generative model
    #
    obs, hid = gen_model.generate(n_t)
    hid_means = hid[:, 0]
    hid_sds = hid[:, 1]

    #
    # Run the Bayes learner model inference on that sequence.
    #
    post_dists_dict = blearner.compute_post_dists_dict(obs,
        keys=['prob', 'margprob', 'mean', 'mode'])
    post_dists_marg_hmean = post_dists_dict['margprob'][0]
    post_dists_marg_hsd = post_dists_dict['margprob'][1]
    post_mean_hmean = post_dists_dict['mean'][:, 0]
    post_mean_hsd = post_dists_dict['mean'][:, 1]
    post_mode_hmean = post_dists_dict['mode'][:, 0]
    post_mode_hsd = post_dists_dict['mode'][:, 1]

    #
    # Plot the results
    #
    # Plot the generated sequence and the Bayes learner inference
    # for the two dimensions of the hidden state
    fontsize = 10
    cticklabelsize = 6
    markersize = 3
    labelspacing = 0.25
    cbaraspect = 40
    cbarpad = 0.01
    plut.setup_mpl_style(fontsize=fontsize)
    t = np.arange(n_t)+1
    labels_dict = {
        "obs": "Beam locations",
        "hid": "True value",
        "post_dist": "Posterior probability",
        "post_mean": "Posterior mean",
        "post_mode": "Posterior mode",
    }

    cmap = "hot_r"
    is_reversed_cmap = cmap.endswith("_r")
    cmap2 = "bone_r" if is_reversed_cmap else "bone"
    labelcolor = "#000000" if is_reversed_cmap else plut.COLORS_GRAY[8]
    fig, axes = plt.subplots(figsize=(
        plut.A4_PAPER_CONTENT_WIDTH,
        plut.DEFAULT_HEIGHT * 2),
        nrows=2,
        gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.set_ylabel("Generative mean")
    # title = f"Hidden mean change-point probability: {p_cp_mean:.3f}"
    # ax.set_title(title)
    plut.plot_sequence_on_ax(ax, t, obs, hid_means, hidden_mean_space,
            post_dists_marg_hmean, post_mean_hmean,
            post_mode_hmean,
            # post_modes=None,
            xlabel="Trial",
            labels_dict=labels_dict,
            markersize=markersize,
            labelspacing=labelspacing,
            cbaraspect=cbaraspect,
            cticklabelsize=cticklabelsize, cbarpad=cbarpad,
            cmap=cmap, labelcolor=labelcolor)
    ax.set_ylim(vmin, vmax)
    ax = axes[1]
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.set_ylabel("Generative variance (s.d.)")
    plut.plot_sequence_on_ax(ax, t, None, hid_sds, hidden_sd_space,
            post_dists_marg_hsd, post_mean_hsd,
            post_mode_hsd,
            xlabel="Trial",
            labels_dict=labels_dict,
            markersize=markersize,
            labelspacing=labelspacing,
            cbaraspect=cbaraspect,
            cticklabelsize=cticklabelsize, cbarpad=cbarpad,
            cmap=cmap2,labelcolor=labelcolor)
    
    if fpath is None:
        ext = "pdf"
        fprefix = plut.outpath_prefix_for_script(__file__)
        fpath = f"{fprefix}.{ext}"
    plut.save_figure(fig, fpath)

if __name__ == "__main__":
    main()
"""
Case study: Learning of a mean magnitude and a stochasticity level that
both undergo independent random change points.

The learner observes continuous values generated from a normal distribution
whose mean and s.d. each follow an independent change point dynamics process.

Example tasks from the literature:
- Weber, Hunt, et al., in prep.

Components of the generative model:
- Hidden state space: [a, b] x {sd_1, sd_2, sd_3}
- Observation space: [a, b]
- Dynamics distribution: One independent change point dynamics process for each of
    the two dimensions of the hidden state, parameterized by p_cp_mean and p_cp_sd,
    the probability of a change point occurring for the first and second dimension,
    respectively. Here, we use an uniform resampling distribution when a change point
    occurrs: all state values have an equal probability of being chosen for the new value
    of the hidden state after the change point.
- Observation distribution: Normal distribution with mean and s.d. equal to the
    first and second dimension of the hidden state, respectively.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import bayeslearner.plotutils as plut

# To get reproducible results, initialize the seed of the random number generator
seed = 15 # 6, 15
np.random.seed(seed)

# Parameters of the generative model
vmin = 0
vmax = 100
h_mean_step = 1/3
h_sd_vals = [2, 4, 8]
n_t = 100
p_cp_means = [0.04, 0.08, 0.12]
p_cp_sd = 0.04

for p_cp_mean in p_cp_means:
    #
    # Define the generative model and create the corresponding Bayes learner model
    #
    hidden_mean_space = bayeslearner.ContinuousUnidimensionalSpace(
        min_value=vmin, max_value=vmax, step=h_mean_step, include_min=True, include_max=True)
    hidden_sd_space = bayeslearner.DiscreteUnidimensionalSpace(h_sd_vals)
    hidden_state_space = bayeslearner.MultimensionalSpace(
        # This defines a 2D space for the hidden state.
        # Dimension 1 and 2 of the hidden state will represent
        # the mean and the sd of the  normal distribution generating the observations.
        [hidden_mean_space, hidden_sd_space])
    init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
    state_dynamics_dist = bayeslearner.change_point_dynamics_dist(
        [hidden_mean_space.n_values, hidden_sd_space.n_values],
        [p_cp_mean, p_cp_sd])
    obs_dist = bayeslearner.truncnorm_dist(
            lambda h: dict(mu=h[..., 0], sigma=h[..., 1], min=vmin, max=vmax))
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
        keys=['prob', 'margprob', 'mean'])
    post_dists_marg_hmean = post_dists_dict['margprob'][0]
    post_dists_marg_hsd = post_dists_dict['margprob'][1]
    post_mean_hmean = post_dists_dict['mean'][:, 0]
    post_mean_hsd = post_dists_dict['mean'][:, 1]

    #
    # Plot the results
    #
    # Plot the generated sequence and the Bayes learner inference
    # for the two dimensions of the hidden state
    plut.setup_mpl_style()
    t = np.arange(n_t)+1
    labels_dict = {
        "obs": "Observations",
        "hid": "True hidden mean",
        "post_dist": "Posterior probability",
        "post_mean": "Posterior mean",  
    }
    fig, axes = plt.subplots(figsize=(plut.A4_PAPER_CONTENT_WIDTH,
        plut.DEFAULT_HEIGHT * 2),
        nrows=2)
    ax = axes[0]
    ax.set_ylabel("Hidden mean value")
    title = f"Hidden mean change-point probability: {p_cp_mean}"
    ax.set_title(title)
    plut.plot_sequence_on_ax(ax, t, obs, hid_means, hidden_mean_space,
            post_dists_marg_hmean, post_mean_hmean,
            labels_dict=labels_dict)    
    ax = axes[1]
    ax.set_ylabel("Hidden s.d. value")
    plut.plot_sequence_on_ax(ax, t, None, hid_sds, hidden_sd_space,
            post_dists_marg_hsd, post_mean_hsd,
            labels_dict=labels_dict)

    fprefix = plut.outpath_prefix_for_script(__file__)
    fname_params = f"p-cp-mean-{p_cp_mean}"
    fpath = f"{fprefix}_{fname_params}.png"
    plut.save_figure(fig, fpath)

    do_kalman_comparison = False
    if do_kalman_comparison:
        #
        # Compare the Bayes learner solution with a Kalman filter
        # with best possible parameters
        #

        from pykalman import KalmanFilter
        initial_state_mean = (vmax + vmin) / 2
        initial_state_covariance = (vmax - vmin) * 1e6
        # Compute an equivalent volatility level for the Kalman filter's postulated
        # Gaussian random walk dynamics from the true generative hidden mean dynamics distribution.
        # The equivalent volatility level is defined as the expected squared
        # difference between two consecutive values of the hidden mean
        # under the dynamics distribution:
        #   E[(h_{t+1, 0}-h_{t, 0})^2]
        # (0 indexes the first dimension of the hidden state, which corresponds to the mean
        # of the observation distribution).
        hmean_state_dynamics_dist = bayeslearner.change_point_dynamics_dist(
            hidden_mean_space.n_values, p_cp_mean)
        state_dynamics_joint_mat = (hmean_state_dynamics_dist
            / hidden_mean_space.n_values) # p(a, b) = p(a | b) * p(b)
        squared_differences_mat = np.array([[(hto-hfrom)**2
                        for hfrom in hidden_mean_space.values]
                        for hto in hidden_mean_space.values])
        expected_squared_difference = np.sum(state_dynamics_joint_mat * squared_differences_mat)
        root_expected_squared_difference = np.sqrt(expected_squared_difference)
        transition_covariance = expected_squared_difference
        observation_covariance = np.mean([sd**2 for sd in h_sd_vals])
        kf = KalmanFilter(initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance)
        kf_post_means, kf_post_vars = kf.filter(obs)
        kf_post_means = np.squeeze(kf_post_means)
        kf_post_vars = np.squeeze(kf_post_vars)

        # Add the Kalman Filter's mean estimates onto the panel of the Bayes learner
        # figure showing the inference of the hidden mean, and save it as a new figure.
        ax = axes[0]
        obs_sd = np.sqrt(observation_covariance)
        kf_label = f"Kalman filter (eq. vol.-sd={root_expected_squared_difference:.1f}, stoch.-sd.={obs_sd:.1f})"
        ax.plot(t, kf_post_means, 'o',
            label=kf_label, color="black", fillstyle="none", ms=2)
        ax.legend(labelcolor=plut.COLORS_GRAY[8])
        fpath = f"{fprefix}_with-kf-mean_{fname_params}.png"
        plut.save_figure(fig, fpath)


        # Clear the last-plotted Kalman filter estimates so that we can reuse
        # the figure below
        ax.lines[-1].remove()

        # Add the Kalman Filter's mean +- sd estimates onto the panel of the Bayes learner
        # figure showing the inference of the hidden mean, and save it as a new figure.
        ax.errorbar(t, kf_post_means, yerr=np.sqrt(kf_post_vars), label=kf_label,
            fmt='o', color=plut.BLACK_COLOR, fillstyle="none", ms=2,
            ecolor=plut.BLACK_COLOR + "66")
        ax.legend(labelcolor=plut.COLORS_GRAY[8])
        fpath = f"{fprefix}_with-kf-mean-sd_{fname_params}.png"
        plut.save_figure(fig, fpath)

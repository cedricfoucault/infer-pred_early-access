"""
Case study: Learning of a mean magnitude undergoing random change points.

The learner observes continuous values generated from a normal distribution
whose mean follows a change point dynamics process.

Example tasks from the literature:
- Nassar et al. (2012)
- McGuire et al. (2014)
- Vaghi et al. (2017)

Components of the generative model:
- Hidden state space: [a, b]
- Observation space: [a, b]
- Dynamics distribution: Change point dynamics, parameterized by a p_cp, the
    probability of a change point occurring. Here, we use an uniform resampling
    distribution when a change point occurrs: all state values have an equal probability
    of being chosen for the new value of the hidden state after the change point.
- Observation distribution: Normal distribution with mean equal to the hidden state,
    and whose s.d. is a constant fixed parameter called the stochasticity level.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import bayeslearner.plotutils as plut

# To get reproducible results, initialize the seed of the random number generator
np.random.seed(0)

# Parameters of the generative model
hmin = 0
hmax = 100
step = 1/3
n_t = 100
p_cps = [0.04, 0.12]
stochasticity_sds = [2, 4, 8]

for p_cp in p_cps:
    for stochasticity_sd in stochasticity_sds:
        #
        # Define the generative model and create the corresponding Bayes learner model
        #
        hidden_state_space = bayeslearner.ContinuousUnidimensionalSpace(
            min_value=hmin, max_value=hmax, step=step, include_min=True, include_max=True)
        init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
        state_dynamics_dist = bayeslearner.change_point_dynamics_dist(
            hidden_state_space.n_values, p_cp)
        obs_dist = bayeslearner.truncnorm_dist(
            lambda h: dict(mu=h, sigma=stochasticity_sd, min=hmin, max=hmax))
        gen_model = bayeslearner.GenerativeModel(hidden_state_space,
                init_state_dist, state_dynamics_dist,
                obs_dist)
        blearner = bayeslearner.BayesLearner(gen_model)

        #
        # Generate a sequence of observations and hidden states using the generative model
        #
        obs, hid = gen_model.generate(n_t)

        #
        # Run the Bayes learner model inference on that sequence.
        #
        post_dists_dict = blearner.compute_post_dists_dict(obs)
        next_post_dists_dict = blearner.compute_next_post_dists_dict(obs)

        #
        # Plot the results
        #
        # Plot the generated sequence and the Bayes learner inference
        plut.setup_mpl_style()
        t = np.arange(n_t)+1
        fig = plut.plot_sequence(t, obs, hid, hidden_state_space,
                post_dists_dict["prob"], post_dists_dict["mean"])
        ax = fig.gca()
        ax.set_ylabel("Value")

        title = f"Change-point probability: {p_cp}, Stochasticity (s.d.): {stochasticity_sd}"
        ax.set_title(title)

        fprefix = plut.outpath_prefix_for_script(__file__)
        fname_params = f"p-cp-{p_cp}_stoch-{stochasticity_sd}" 
        fpath = f"{fprefix}_{fname_params}.png"
        plut.save_figure(fig, fpath)

        do_kf_comparison = False
        if do_kf_comparison:
            #
            # Compare the Bayes learner solution with a Kalman filter
            # with best possible parameters (i.e. choosing the parameters of its
            # generative model to be as close to the true generative model as they can be,
            # although they can't match exactly since they have a different structure).
            #

            from pykalman import KalmanFilter
            initial_state_mean = (hmax + hmin) / 2
            initial_state_covariance = (hmax - hmin) * 1e6
            # Compute an equivalent volatility level for the Kalman filter's postulated
            # Gaussian random walk dynamics from the true generative dynamics distribution.
            # The equivalent volatility level is defined as the expected squared
            # difference between two consecutive values of the hidden state
            # under the dynamics distribution:
            #   E[(h_{t+1}-h_{t})^2]
            state_dynamics_joint_mat = (state_dynamics_dist
                / hidden_state_space.n_values) # p(a, b) = p(a | b) * p(b)
            squared_differences_mat = np.array([[(hto-hfrom)**2
                            for hfrom in hidden_state_space.values]
                            for hto in hidden_state_space.values])
            expected_squared_difference = np.sum(state_dynamics_joint_mat * squared_differences_mat)
            root_expected_squared_difference = np.sqrt(expected_squared_difference)
            transition_covariance = expected_squared_difference
            observation_covariance = stochasticity_sd ** 2
            kf = KalmanFilter(initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance)
            kf_post_means, kf_post_vars = kf.filter(obs)
            kf_post_means = np.squeeze(kf_post_means)
            kf_post_vars = np.squeeze(kf_post_vars)

            # Add the Kalman Filter's mean estimates onto the Bayes learner figure
            # and save it as a new figure
            kf_label = f"Kalman filter (eq.-vol.-sd={root_expected_squared_difference:.1f}, stoch.-sd.={stochasticity_sd:.1f})"
            ax.plot(t, kf_post_means, 'o',
                label=kf_label, color="black", fillstyle="none", ms=2)
            ax.legend(labelcolor=plut.COLORS_GRAY[8])
            fpath = f"{fprefix}_with-kf-mean_{fname_params}.png"
            plut.save_figure(fig, fpath)
            
            # Clear the last-plotted Kalman filter estimates so that we can reuse
            # the figure below
            ax.lines[-1].remove()

            # Add the Kalman Filter's mean +- sd estimates onto the Bayes learner figure
            # and save it as a new figure
            ax.errorbar(t, kf_post_means, yerr=np.sqrt(kf_post_vars), label=kf_label,
                fmt='o', color=plut.BLACK_COLOR, fillstyle="none", ms=2,
                ecolor=plut.BLACK_COLOR + "66")
            ax.legend(labelcolor=plut.COLORS_GRAY[8])
            fpath = f"{fprefix}_with-kf-mean-sd_{fname_params}.png"
            plut.save_figure(fig, fpath)

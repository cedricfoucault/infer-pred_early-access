"""
Case study: Learning of a mean magnitude following a Gaussian random walk.

The learner observes continuous values generated from a normal distribution 
whose mean follows a Gaussian random walk dynamics process.

Example tasks from the literature:
- Piray & Daw (2023)
- Daw et al. (2006)
- Lee, Gold & Kable (2020)
- Findling et al. (2019)

Components of the generative model:
- Hidden state space: [a, b]
- Observation space: [a, b]
- Dynamics distribution: Gaussian random walk bounded in [a, b], parameterized by
    a volatility level, which is the s.d. of the step (difference between two consecutive
    values of the hidden state).
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
volatility_sds = [2, 4]
stochasticity_sds = [2, 4, 8]

for stochasticity_sd in stochasticity_sds:
    for volatility_sd in volatility_sds:
        #
        # Define the generative model and create the corresponding Bayes learner model
        #
        # Define the hidden state space
        hidden_state_space = bayeslearner.ContinuousUnidimensionalSpace(
            min_value=hmin, max_value=hmax, step=step, include_min=True, include_max=True)
        # Define the probability distribution of the initial hidden state, p(h_0).
        init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
        # Define the probability distribution of the dynamics of the hidden state, p(h_{t+1} | h_t).
        state_dynamics_dist = bayeslearner.gaussian_random_walk_dynamics_dist(
            hidden_state_space.values, volatility_sd)
        # Define the probability distribution generating the observation given the hidden state, p(x_t | h_t).
        obs_dist = bayeslearner.truncnorm_dist(
            lambda h: dict(mu=h, sigma=stochasticity_sd, min=hmin, max=hmax))
        # Create the object that represents the generative model.
        gen_model = bayeslearner.GenerativeModel(hidden_state_space,
                init_state_dist, state_dynamics_dist,
                obs_dist)
        # Create the object that represents the Bayes learner model.
        blearner = bayeslearner.BayesLearner(gen_model)

        #
        # Generate a sequence of observations and hidden states using the generative model
        #
        obs, hid = gen_model.generate(n_t)

        #
        # Run the Bayes learner model inference on that sequence
        #
        # The outputs of the inference are the posterior probability distributions
        # for each time step of the sequence, and measures derived from that distribution
        # (mean, sd).
        post_dists_dict = blearner.compute_post_dists_dict(obs)

        #
        # Plot the results
        #
        # Plot the sequence of observations, the true hidden states,
        # and the inference of hidden states given the observations
        # (posterior probability distribution and posterior mean).
        plut.setup_mpl_style()
        t = np.arange(n_t)+1
        fig = plut.plot_sequence(t, obs, hid, hidden_state_space,
                post_dists_dict["prob"], post_dists_dict["mean"])
        ax = fig.gca()
        ax.set_ylabel("Value")
        title = f"Volatility/Stochasticity (s.d.): {volatility_sd}/{stochasticity_sd}"
        ax.set_title(title)
        fprefix = plut.outpath_prefix_for_script(__file__)
        fname_params = f"stoch-{stochasticity_sd:02d}_vol-{volatility_sd:02d}"
        fpath = f"{fprefix}_{fname_params}.png"
        plut.save_figure(fig, fpath)

        do_kf_check = False
        if do_kf_check:
            #
            # Sanity check:
            # Compare the Bayes learner solution with the Kalman filter solution
            #

            from pykalman import KalmanFilter
            initial_state_mean = (hmax + hmin) / 2
            initial_state_covariance = (hmax - hmin) * 1e6
            transition_covariance = volatility_sd ** 2
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
            kf_label = f"Kalman filter (true vol. and stoch.)"
            ax.plot(t, kf_post_means, 'o',
                label=kf_label,
                color="black", fillstyle="none", ms=2)
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

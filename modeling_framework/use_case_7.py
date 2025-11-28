"""
Case study: Binary hidden state discrimination task with dynamic probabilistic reversals.

The learner observes continuous values generated from one of two possible distributions,
and must infer which of the two distributions is the one currently generating the
observations. The hidden state is a binary variable that indicates which distribution
is the one currently generating the observations. The hidden state follows a reversal
dynamics process: it switches between the two values based on a given probability
of reversal.

Example tasks:
- Ossmy et al. (2013)
- Glaze, Kable, & Gold (2015)
- Murphy et al. (2021)

Components of the generative model:
- Hidden state space: {H_1, H_2} (two categorical values)
- Observation space: Real numbers
- Dynamics distribution: Reversal dynamics, parametrized by p_rev, the
    probability that the hidden state switches from one value to the other.
- Observation distribution: Normal distribution with mean equal to -m or +m
    depending on whether the hidden state is equal to H_1 or H_2, and with a given
    fixed s.d. The ratio of the mean to the s.d. determines the signal-to-noise ratio.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import bayeslearner.plotutils as plut

# To get reproducible results, initialize the seed of the random number generator
np.random.seed(1)

# Parameters of the generative model
n_t = 40
snrs = [1/4, 1/10]
ratio_sd_dists = [0.24, 0.33, 0.41] # parameter levels used in Glaze et al. (2015)
snrs = [(1/r)**2 for r in ratio_sd_dists]
p_revs = [0.05, 0.3] # in Glaze et al., the levels were: (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)

for i_snr, snr in enumerate(snrs):
    for i_prev, p_rev in enumerate(p_revs):
        #
        # Define the generative model and create the corresponding Bayes learner model
        #
        hidden_state_vals = [-1, 1]
        hidden_state_space = bayeslearner.DiscreteUnidimensionalSpace(hidden_state_vals)
        init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
        state_dynamics_dist = bayeslearner.binary_reversal_dynamics_dist(p_rev)
        stochasticity_sd = np.sqrt((hidden_state_vals[1] - hidden_state_vals[0]) ** 2 / snr)
        obs_dist = bayeslearner.norm_dist(lambda h: dict(mu=h, sigma=stochasticity_sd))
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

        #
        # Plot the results
        #
        # Plot the generated sequence
        plut.setup_mpl_style()
        t = np.arange(n_t)+1
        fig, axes = plt.subplots(figsize=(plut.A4_PAPER_CONTENT_WIDTH,
            plut.DEFAULT_HEIGHT * 3),
            nrows=3)
        ax = axes[0]
        plut.plot_sequence_on_ax(ax, t, obs, hid, None,
                None, None, labelcolor=plut.BLACK_COLOR)
        # ax.axhline(y=0, color=plut.BLACK_COLOR, lw=1., ls=':')
        y0_lw = ax.spines['left'].get_linewidth()
        ax.axhline(y=0, color=plut.BLACK_COLOR, lw=y0_lw, ls='-', zorder=-1)
        # Set a symmetric ylim
        yabsmax = np.max(np.abs(obs))
        ax.set_ylim(-yabsmax*1.05, yabsmax*1.05)
        ax.set_ylabel("Value")
        title = f"Reversal probability: {p_rev:.02f}, Stochasticity (s.d.): {stochasticity_sd:.2f}"
        ax.set_title(title)
        # Plot posterior probabilities
        ax = axes[1]
        plut.plot_sequence_on_ax(ax, t, None, None, hidden_state_space,
                post_dists_dict["prob"], None)
        ax.set_ylabel("Hidden state")
        ax.set_yticks(hidden_state_vals)
        # Plot the posterior odds
        # (log of the ratio of the posterior probabilities of the two states)
        ax = axes[2]
        dv = np.log(post_dists_dict["prob"][:, 1] / post_dists_dict["prob"][:, 0])
        dv_color = "#996600"
        ax.plot(t, dv, '-', ms=1, lw=1, color=dv_color)
        ax.set_ylabel("Decision variable (posterior log-odds)")
        # Plot decision threshold to achieve a given % error
        crit_err = 0.05 # i.e. 95% correct
        thresh = np.log((1-crit_err) / crit_err)
        thresh_color = plut.COLORS_GRAY[3]
        thresh_ls = "--"
        thresh_lw = 1.
        ax.axhline(y=thresh, ls=thresh_ls, lw=thresh_lw, color=thresh_color)
        ax.axhline(y=-thresh, ls=thresh_ls, lw=thresh_lw, color=thresh_color)
        ax.axhline(y=0, color=plut.BLACK_COLOR, lw=y0_lw, ls='-', zorder=-1)
        # Set a symmetric ylim
        yabsmax = np.max(np.abs(dv))
        ax.set_ylim(-yabsmax*1.05, yabsmax*1.05)

        fprefix = plut.outpath_prefix_for_script(__file__)
        fname_params = f"snr-level-{i_snr+1}_p-rev-level-{i_prev+1}" 
        fpath = f"{fprefix}_{fname_params}.png"
        plut.save_figure(fig, fpath)
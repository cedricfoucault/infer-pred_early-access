"""
Case study: Binary hidden state discrimination task.

The learner observes continuous values generated from one of two possible distributions,
and must infer which of the two distributions is the one that is generating
the observations. The hidden state is a binary variable that indicates which of
the two distributions is the one generating the observations. Here the hidden state
does not change (it is randomly chosen at the beginning of the sequence, and then
fixed for the rest of the sequence).

Example tasks:
- Kim & Shadlen (1999)
- Gold & Shadlen (2007) (review)
- Wyart … Summerfield (2012)
- Drugowitsch*, Wyart*, et al. (2016)
- Hanks & Summerfield (2017) (review)

Components of the generative model:
- Hidden state space: {H_1, H_2}
    (H_1 and H_2 are categorical values that could represent, for example,
    "left" and "right" in a motion discrimination task)
- Observation space: Real numbers
- Dynamics distribution: Static - the hidden state is fixed to its initial value.
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
snrs = [1, 1/4, 1/10]

for i_snr, snr in enumerate(snrs):
    #
    # Define the generative model and create the corresponding Bayes learner model
    #
    hidden_state_vals = [-1, 1]
    hidden_state_space = bayeslearner.DiscreteUnidimensionalSpace(hidden_state_vals)
    init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
    state_dynamics_dist = None
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
    title = f"Signal-to-noise ratio: {snr:.2f}"
    ax.set_title(title)
    # Plot posterior probabilities
    ax = axes[1]
    plut.plot_sequence_on_ax(ax, t, None, None, hidden_state_space,
            post_dists_dict["prob"], None, None)
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
    thresh_label = f"{(1-crit_err)*100:.0f}% correct decision thresholds"
    thresh_color = plut.COLORS_GRAY[3]
    thresh_ls = "--"
    thresh_lw = 1.
    ax.axhline(y=thresh, ls=thresh_ls, lw=thresh_lw, color=thresh_color, label=thresh_label)
    ax.axhline(y=-thresh, ls=thresh_ls, lw=thresh_lw, color=thresh_color)
    ax.axhline(y=0, color=plut.BLACK_COLOR, lw=y0_lw, ls='-', zorder=-1)
    ax.legend()
    # Set a symmetric ylim
    yabsmax = np.max(np.abs(dv))
    ax.set_ylim(-yabsmax*1.05, yabsmax*1.05)

    fprefix = plut.outpath_prefix_for_script(__file__)
    fname_params = f"snr-level-{i_snr+1}" 
    fpath = f"{fprefix}_{fname_params}.png"
    plut.save_figure(fig, fpath)
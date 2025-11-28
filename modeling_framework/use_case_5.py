"""
Case study: Learning of a probability undergoing random change points.

The learner observes binary values generated from a Bernoulli distribution whose
parameter follows a change point dynamics process.

Example tasks from the literature:
- Behrens et al., 2007
- Gallistel et al., 2014
- Foucault & Meyniel, 2024

Components of the generative model:
- Hidden state space: [0, 1]
- Observation space: {0, 1}
- Dynamics distribution: Change point dynamics, parameterized by a p_cp, the
    probability of a change point occurring. Here, we use an uniform resampling
    distribution when a change point occurrs: all state values have an equal probability
    of being chosen for the new value of the hidden state after the change point.
- Observation distribution: Bernoulli distribution with parameter equal to the
    hidden state.
"""

import bayeslearner
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import bayeslearner.plotutils as plut

# To get reproducible results, initialize the seed of the random number generator
seed = 2
np.random.seed(seed)

# Parameters of the generative model
hmin = 0
hmax = 1
step = 1/300
n_t = 100
p_cps = [1/20, 1/50]

for p_cp in p_cps:
    #
    # Define the generative model and create the corresponding Bayes learner model
    #
    hidden_state_space = bayeslearner.ContinuousUnidimensionalSpace(
        min_value=hmin, max_value=hmax, step=step, include_min=True, include_max=True)
    init_state_dist = bayeslearner.uniform_dist(hidden_state_space.n_values)
    state_dynamics_dist = bayeslearner.change_point_dynamics_dist(
        hidden_state_space.n_values, p_cp)
    obs_dist = bayeslearner.bernoulli_dist(lambda h: dict(p=h))
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
    # Plot the results (generated sequence and the Bayes learner inference)
    #
    plut.setup_mpl_style()
    t = np.arange(n_t)+1
    fig = plut.plot_sequence(t, obs, hid, hidden_state_space,
            post_dists_dict["prob"], post_dists_dict["mean"])
    ax = fig.gca()
    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel("Value")

    title = f"Change-point probability: {p_cp}"
    ax.set_title(title)

    fprefix = plut.outpath_prefix_for_script(__file__)
    fname_params = f"p-cp-{p_cp}"
    fpath = f"{fprefix}_{fname_params}.png"
    plut.save_figure(fig, fpath)

    do_problearner_comparison = False
    if do_problearner_comparison:
        #
        # Compare the Bayes learner solution with the solution implemented by
        # the Bayesian model used in https://github.com/cedricfoucault/ada-learn,
        # which we call "prob learner" below.
        #

        import sys
        # Adjust this path to point to the location of the "model_learner.py" file
        module_fpath = "/Users/cedric/Code_and_Repositories/Ada-Learn_-_Code/ada-learn_public/analysis"
        sys.path.append(module_fpath)
        import model_learner as problearner

        resol = hidden_state_space.n_values
        problearner_post_dists_dict = problearner.ideal_learner_prob_inference_from_outcomes(obs, p_cp,
            keys=['mean', 'SD', 'dist'],
            p1_min=hmin, p1_max=hmax,
            do_inference_on_current_trial=True,
            resol=resol)

        # Plot two rows, each showing the generated sequence and the learner inference,
        # the top row showing the Bayes learner's inference and the bottom row
        # showing the prob learner's inference.

        fig, axes = plt.subplots(figsize=(plut.A4_PAPER_CONTENT_WIDTH,
            plut.DEFAULT_HEIGHT * 2),
            nrows=2)
        ax = axes[0]
        labels_dict = plut.LABELS_DICT.copy()
        labels_dict["post_dist"] += " - Bayes learner"
        labels_dict["post_mean"] += " - Bayes learner"
        plut.plot_sequence_on_ax(ax, t, obs, hid, hidden_state_space,
                post_dists_dict["prob"], post_dists_dict["mean"],
                labels_dict=labels_dict)
        ax.set_ylabel("Value")
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(title)
        ax = axes[1]
        labels_dict = plut.LABELS_DICT.copy()
        labels_dict["post_dist"] += " - Prob learner"
        labels_dict["post_mean"] += " - Prob learner"
        plut.plot_sequence_on_ax(ax, t, obs, hid, hidden_state_space,
                problearner_post_dists_dict["dist"], post_dists_dict["mean"],
                labels_dict=labels_dict)
        ax.set_ylabel("Value")
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(title)

        fpath = f"{fprefix}_with-prob-learner_{fname_params}.png"
        plut.save_figure(fig, fpath)

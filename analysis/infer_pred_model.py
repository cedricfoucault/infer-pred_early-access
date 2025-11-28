"""
Main module for the modeling of the Infer-Pred task.
"""

import numpy as np
import scipy.stats
import utils

# Import bayeslearner
# Add the framework directory to sys.path so the bayeslearner package can be imported
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DIR = REPO_ROOT / "modeling_framework"
import sys
sys.path.append(str(FRAMEWORK_DIR))
import bayeslearner

class InferPredStateSpace(bayeslearner.MultimensionalSpace):
    def __init__(self, space_config, grid_step=1):
        """
        Create the state space for the task matching the specifications
        provided in the given parameters.
        """
        dim_keys = ["hmean"]
        if space_config.get("hsd"):
            dim_keys += ["hsd"]
            self.has_hsd = True
        else:
            self.has_hsd = False
        if space_config.get("hvol"):
            dim_keys += ["hvol"]
            self.has_hvol = True
        else:
            self.has_hvol = False
        dimensions = []
        for k in dim_keys:
            dim_type = space_config[k]["type"]
            if dim_type == "continuous":
                min_value = space_config[k]["min"]
                max_value = space_config[k]["max"]
                step = space_config[k].get("step", grid_step)
                is_angular = space_config[k].get("is_angular", False)
                n_samples = space_config[k].get("n_samples", None)
                sample_fun = space_config[k].get("sample_fun", None)
                include_max = space_config[k].get("include_max", False)
                dim = bayeslearner.ContinuousUnidimensionalSpace(
                    min_value=min_value, max_value=max_value, step=step,
                    include_min=True, include_max=include_max,
                    is_angular=is_angular,
                    n_samples=n_samples, sample_fun=sample_fun)
            elif dim_type == "discrete":
                values = space_config[k]["values"]
                dim = bayeslearner.DiscreteUnidimensionalSpace(values)
            dimensions += [dim]
        super().__init__(dimensions)

    @property
    def hmean_dim(self):
        return 0

    @property
    def hsd_dim(self):
        return 1 if self.has_hsd else None

    @property
    def hvol_dim(self):
        if self.has_hvol:
            if self.has_hsd:
                return 2
            else:
                return 1
        else:
            return None

    @property
    def hmean_space(self):
        return self.dimensions[self.hmean_dim]

    @property
    def hsd_space(self):
        return self.dimensions[self.hsd_dim] if self.has_hsd else None

    @property
    def hvol_space(self):
        return self.dimensions[self.hvol_dim] if self.has_hvol else None

class InferPredGenerativeModel(bayeslearner.GenerativeModel):
    def __init__(self, config):
        """
        Create the generative for the task matching the specifications
        provided in the given config parameters.
        """
        # Define the hidden state space
        self.hidstate_space = InferPredStateSpace(
            config["state_space"])
        has_hsd = self.hidstate_space.has_hsd
        has_hvol = self.hidstate_space.has_hvol
        hmean_space = self.hidstate_space.hmean_space
        hsd_space = self.hidstate_space.hsd_space
        hvol_space = self.hidstate_space.hvol_space
        hmean_dim = self.hidstate_space.hmean_dim
        hsd_dim = self.hidstate_space.hsd_dim
        hvol_dim = self.hidstate_space.hvol_dim

        # Define the probability distribution of the initial hidden state, p(h_0).
        # This is a uniform distribution over the entire state space.
        init_state_dist = bayeslearner.uniform_dist(self.hidstate_space.n_values)
        # Define the probability distribution generating the observation given
        # the hidden state, p(x_t | h_t).
        # This is a normal distribution whose mean is one dimension of the hidden state,
        # and whose s.d. may either be fixed or another dimension of the hidden state.
        if has_hsd:
            obs_dist = bayeslearner.circnorm_dist(lambda h:
                dict(mu=h[..., hmean_dim], sigma=h[..., hsd_dim]))
        else:
            sd = config["sd"]
            obs_dist = bayeslearner.circnorm_dist(lambda h:
                dict(mu=h[..., hmean_dim], sigma=sd))
        # Define the probability distribution of the dynamics of the hidden
        # state, p(h_{t+1} | h_t).
        dynamics = config["dynamics"]
        if (("hmean_hsd" in dynamics)
            and (dynamics["hmean_hsd"]["type"] == "change_point_hmean_reset_hsd")):
            # In this case, the dynamics of the mean and standard deviation are
            # defined jointly.
            # Change points in mean are coupled with a reset of the standard
            # deviation. Change points in standard deviation occur independently.
            cp_dynamics_dists = []
            for k in ["hmean", "hsd"]:
                p_cp = dynamics["hmean_hsd"][f"p_cp_{k}"]
                min_change_amount = dynamics["hmean_hsd"].get(f"min_change_amount_{k}", 0)
                max_change_amount = dynamics["hmean_hsd"].get(f"max_change_amount_{k}", np.inf)
                if k == "hmean":
                    def is_allowed_change_mean(i_from, i_to):
                        absdiff = np.abs(utils.calc.subtract_angle(
                            hmean_space.values[i_to], hmean_space.values[i_from]))
                        return ((absdiff >= min_change_amount)
                            and (absdiff <= max_change_amount))
                    cp_dynamics_dists += [bayeslearner.change_point_dynamics_dist(
                        hmean_space.n_values, p_cp,
                        is_allowed_change_fun=is_allowed_change_mean)]
                elif k == "hsd":
                    def is_allowed_change_sd(i_from, i_to):
                        absdiff = np.abs(hsd_space.values[i_to] - hsd_space.values[i_from])
                        return ((absdiff >= min_change_amount)
                            and (absdiff <= max_change_amount))
                    cp_dynamics_dists += [bayeslearner.change_point_dynamics_dist(
                        hsd_space.n_values, p_cp,
                        is_allowed_change_fun=is_allowed_change_sd)]
            def dynamics_p_fun(h_from, h_to, i_from_ND, i_to_ND):
                    # Probability for the change in hmean considering the current hvol
                    p = cp_dynamics_dists[0][i_to_ND[hmean_dim], i_from_ND[hmean_dim]]
                    # Probability for the change in hsd
                    did_mean_change = (i_to_ND[hmean_dim] != i_from_ND[hmean_dim])
                    if did_mean_change:
                        # When the mean changes, the hsd is reset
                        # (resampled from a uniform distribution)
                        p *= 1 / hsd_space.n_values
                    else:
                        # When the mean does not change, the hsd follows its own
                        # change-point dynamics
                        p *= cp_dynamics_dists[1][i_to_ND[hsd_dim], i_from_ND[hsd_dim]]
                    return p
            hidstate_dynamics_dist = bayeslearner.construct_dynamics_dist(
                self.hidstate_space, p_fun=dynamics_p_fun)
        else:
            # Dynamics of the mean:
            if "change_point" in dynamics["hmean"]["type"]:
                # Change point dynamics.
                min_change_amount = dynamics["hmean"].get("min_change_amount", 0)
                max_change_amount = dynamics["hmean"].get("max_change_amount", np.inf)
                def is_allowed_change_mean(i_from, i_to):
                    absdiff = np.abs(utils.calc.subtract_angle(
                        hmean_space.values[i_to], hmean_space.values[i_from]))
                    return ((absdiff >= min_change_amount)
                        and (absdiff <= max_change_amount))
                if not has_hvol:
                    # Known, fixed p_cp
                    p_cp = dynamics["hmean"]["p_cp"] # change point probability
                    # is_allowed_change_mean = None
                    hmean_dynamics_dist = bayeslearner.change_point_dynamics_dist(
                        hmean_space.n_values, p_cp,
                        is_allowed_change_fun=is_allowed_change_mean)
                else:
                    # Inferred p_cp
                    hmean_dynamics_dist_per_hvol = {
                        h_p_cp: bayeslearner.change_point_dynamics_dist(
                        hmean_space.n_values, h_p_cp,
                        is_allowed_change_fun=is_allowed_change_mean)
                        for h_p_cp in hvol_space.values}

            elif "random_walk" in dynamics["hmean"]["type"]:
                # Circular normal random walk.
                if not has_hvol:
                    # Known, fixed volatility
                    volatility = dynamics["hmean"]["volatility"]
                    hmean_dynamics_dist = bayeslearner.circnorm_random_walk_dynamics_dist(
                        hmean_space.values, volatility)
                else:
                    # Inferred volatlity
                    hmean_dynamics_dist_per_hvol = {
                        hvol: bayeslearner.circnorm_random_walk_dynamics_dist(
                        hmean_space.values, hvol)
                        for hvol in hvol_space.values}
            else:
                raise ValueError("Invalid dynamics")
            # Dynamics of the standard deviation:
            if has_hsd:
                if "change_point" in dynamics["hsd"]["type"]:
                    p_cp = dynamics["hsd"]["p_cp"]
                    min_change_amount = dynamics["hsd"].get("min_change_amount", 0)
                    max_change_amount = dynamics["hsd"].get("max_change_amount", np.inf)
                    def is_allowed_change_sd(i_from, i_to):
                        absdiff = np.abs(hsd_space.values[i_to] - hsd_space.values[i_from])
                        return ((absdiff >= min_change_amount)
                            and (absdiff <= max_change_amount))
                    # is_allowed_change_sd = lambda i_from, i_to: (i_from != i_to)
                    hsd_dynamics_dist = bayeslearner.change_point_dynamics_dist(
                        hsd_space.n_values, p_cp,
                        is_allowed_change_fun=is_allowed_change_sd)
                elif "random_walk" in dynamics["hsd"]["type"]:
                    # Normal random walk
                    hsd_volatility = dynamics["hsd"]["volatility"]
                    hsd_dynamics_dist = bayeslearner.gaussian_random_walk_dynamics_dist(
                        hsd_space.values, hsd_volatility)
                else:
                    raise ValueError("Invalid dynamics")
            
            if not has_hvol:
                # The dynamics of the hidden state are defined by the dynamics of the mean
                # and the dynamics of the standard deviation.
                # These two dynamics are here independent of each other.
                # The joint dynamics distribution is given by the product
                # of the two dynamics distributions.
                dynamics_dists = [hmean_dynamics_dist]
                if has_hsd:
                    dynamics_dists += [hsd_dynamics_dist]
                def dynamics_p_fun(h_from, h_to, i_from_ND, i_to_ND):
                    return np.prod([p[i_to_ND[dim], i_from_ND[dim]]
                        for dim, p in enumerate(dynamics_dists)])
                hidstate_dynamics_dist = bayeslearner.construct_dynamics_dist(
                    self.hidstate_space, p_fun=dynamics_p_fun)
            else:
                # In case where the volatility is inferred,
                # the dynamics are obtained by computing the product of the dynamics
                # of the sd, the dynamics of the volatility, and the dynamics of the mean
                # considering the current volatility value encoded in the hidden state.
                if "change_point" in dynamics["hvol"]["type"]:
                    p_cp_hvol = dynamics["hvol"]["p_cp"]
                    hvol_dynamics_dist = bayeslearner.change_point_dynamics_dist(
                        hvol_space.n_values, p_cp_hvol)
                elif "random_walk_lognormal" in dynamics["hvol"]["type"]:
                    hvol_volatility = dynamics["hvol"]["volatility"]
                    hvol_dynamics_dist = bayeslearner.lognormal_random_walk_dynamics_dist_1D_state(
                        hvol_space.values, hvol_volatility)
                else:
                    raise ValueError("Invalid dynamics")
                def dynamics_p_fun(h_from, h_to, i_from_ND, i_to_ND):
                    # Probability for the change in hmean considering the current hvol
                    vol = h_to[hvol_dim]
                    p = hmean_dynamics_dist_per_hvol[vol][i_to_ND[hmean_dim], i_from_ND[hmean_dim]]
                    # Probability for the change in hsd (if applicable)
                    if has_hsd:
                        p *= hsd_dynamics_dist[i_to_ND[hsd_dim], i_from_ND[hsd_dim]]
                    # Probability for the change in hvol
                    p *= hvol_dynamics_dist[i_to_ND[hvol_dim], i_from_ND[hvol_dim]]
                    return p
                hidstate_dynamics_dist = bayeslearner.construct_dynamics_dist(
                    self.hidstate_space, p_fun=dynamics_p_fun)
        # Create the generative model
        super().__init__(self.hidstate_space, init_state_dist, hidstate_dynamics_dist,
            obs_dist)
        # Keep track of constraints included in the config dictionary
        # that will be applied  when generating sequences through the generate()
        # method. These constraints won't be used for inference over the
        # generative model, as they are unknown to the subject that performs the task.
        constraints = config.get("constraints", None)
        if constraints:
            self.has_constraints = True
            self.constraints = constraints
        else:
            self.has_constraints = False

    @property
    def hmean_dim(self):
        return self.hidstate_space.hmean_dim

    @property
    def hsd_dim(self):
        return self.hidstate_space.hsd_dim

    @property
    def has_hsd(self):
        return self.hidstate_space.has_hsd

    def generate(self, nobs):
        """
        Generate a sequence of observations and hidden states.
        """
        if not self.has_constraints:
            return super().generate(nobs)
        else:
            # Override super.generate() behavior in case of constraints
            hmean_dim = self.hmean_dim
            if self.has_hsd:
                hsd_dim = self.hsd_dim
                keys_dims = [("hmean", hmean_dim), ("hsd", hsd_dim)]
            else:
                keys_dims = [("hmean", hmean_dim)]
            constraints = self.constraints
            # Define run length as the number of time steps since the last
            # change point. This definition matches that of a geometric
            # distribution defined as the number of Bernoulli trials needed to
            # get the one success. Thus, the reset value for the run length at
            # the first time step and after each change point is equal to 1.
            runlen_reset_val = 1 
            data = { "h_prev": None}
            def are_constraints_satisfied_fun(t, h_t, data):
                # Initial time step
                if (data["h_prev"] is None):
                    data["h_prev"] = h_t
                    data["curr_runlen"] = {"hmean": runlen_reset_val, "hsd": runlen_reset_val}
                    return True
                # All subsequent time steps
                # Check whether constraints are satisfied at the current time
                # step, and keep track of the current run lengths and the
                # previous value of the hidden state, in order to check
                # constraints at the next time step.
                new_runlen = data["curr_runlen"].copy()
                for k, dim in keys_dims:
                    cst = constraints.get(k, {})
                    prev_val = data["h_prev"][dim]
                    curr_val = h_t[dim]
                    diff = (utils.calc.subtract_angle(prev_val, curr_val)
                        if k == "hmean" else (prev_val - curr_val))
                    change_amt = np.abs(diff)
                    if np.isclose(change_amt, 0):
                        # no change point
                        if (data["curr_runlen"][k] + 1) > cst.get("max_run_length", np.inf):
                            return False
                        else:
                            new_runlen[k] += 1
                    else:
                        # change point
                        if change_amt < cst.get("min_change_amount", 0):
                            return False
                        if change_amt > cst.get("max_change_amount", np.inf):
                            return False
                        if data["curr_runlen"][k] < cst.get("min_run_length", 0):
                            return False
                        else:
                            new_runlen[k] = runlen_reset_val
                data["h_prev"] = h_t
                data["curr_runlen"] = new_runlen
                return True
            return super().generate_with_constraints(nobs,
                are_constraints_satisfied_fun=are_constraints_satisfied_fun,
                data=data)

class InferPredInferenceModel(bayeslearner.BayesLearner):
    """
    Bayesian inference model for the infer-pred task.
    """

    def __init__(self, config):
        gen_model = InferPredGenerativeModel(config)
        super().__init__(gen_model)

    @property
    def hmean_dim(self):
        return self.generative_model.hmean_dim

    @property
    def hsd_dim(self):
        return self.generative_model.hsd_dim

    @property
    def has_hsd(self):
        return self.generative_model.has_hsd

class InferPredBehaviorModel:
    """
    Base class for a behavior model in the infer-pred task.
    """
    def __init__(self, config):
        self.config = config

    @property
    def id(self):
        return self.config["id"]

    @property
    def name(self):
        return self.config["model"]["name"]

    @property
    def initial_bet(self):
        initial_vals = self.config["task"]["responses"]["initial_values"]
        return np.array([initial_vals["location"], initial_vals["width"]])

    @property
    def has_probabilistic_updates(self):
        return (self.config["model"].get("prob_updt", None) is not None)

    @property
    def prob_updt_fun_type(self):
        return self.config["model"]["prob_updt"]["fun_type"]

    @property
    def prob_updt_fun_params(self):
        return self.config["model"]["prob_updt"]["fun_params"]

    def compute_bet_given_obs(self, obs):
        """
        Compute the sequence of bets (bet location and width)
        produced by the model having received the given observation sequence.

        Parameters
        ----------
        obs: 1-D array of shape (nobs,)
            Observation sequence.

        Returns
        -------
        bet: 2-D array of shape (nobs, 2)
            Bet center location and width for each time step of the
            sequence. The first and second dimension on axis 1 of the array
            (i.e. bet[:, 0] and bet[:, 1]) correspond to the center location
            location and width of the bet interval, respectively.
        """
        assert False, "Should be implemented by subclasses"

    def get_prev_bet(self, bet):
        """
        Given a sequence of bets, return the sequence of previous bets, i.e.
        i.e. shifted back by one time step. The first bet of the sequence is the
        initial bet value as set up in the task configuration.
        """
        betloc = bet[:, 0]
        betwid = bet[:, 1]
        initial_bet = self.initial_bet
        betloc_prior = utils.calc.shift_back(betloc,
            initial_bet[0])
        betwid_prior = utils.calc.shift_back(betwid,
            initial_bet[1])
        return np.stack([betloc_prior, betwid_prior], axis=1)

    def generate_probabilistic_updates(self, target_bet, fun_type, fun_params):
        """
        Generate bet responses (bet location and width) from the given targets
        using probabilistic updates, such that bets are updated to the target
        location only probabilistically.
        """
        bet = np.empty_like(target_bet)
        curr_bet = self.initial_bet
        r = np.random.rand(target_bet.shape[0])
        for i in range(target_bet.shape[0]):
            curr_betloc = curr_bet[0]
            target_betloc = target_bet[i, 0]
            discrepancy = np.abs(utils.calc.subtract_angle(
                curr_betloc, target_betloc))
            updt_prob = self.compute_update_probability(discrepancy, fun_type, fun_params)
            if r[i] < updt_prob:
                curr_bet = target_bet[i, :]
            bet[i, :] = curr_bet
        return bet

    def compute_update_probability(self, d, fun_type, fun_params):
        """
        Compute the update probability for the given discrepancy, using the
        given type of probability function and its parameters.

        Parameters
        ----------
        d: float
            discrepancy (angular difference) between the target and current position,
            ranging between 0 and 180 degrees
        fun_type: string
            type of function to use for computing the update probability.
            Possible values: "logistic", "truncnorm", "beta".
        fun_params: list
            parameters for the update probability function.

        Returns
        -------
        p: float
            update probability (between 0 and 1)
        """
        if fun_type == "logistic":
            d0 = fun_params[0]
            k = fun_params[1]
            return 1 / (1 + np.exp(-k * (d - d0)))
        elif fun_type == "truncnorm":
            mean = fun_params[0]
            std = fun_params[1]
            lb, ub = 0, 180
            lb_norm, ub_norm = (lb - mean) / std, (ub - mean) / std
            trunc = scipy.stats.truncnorm(lb_norm, ub_norm, loc=mean, scale=std)
            return trunc.cdf(d)
        elif fun_type == "beta":
            alpha = fun_params[0]
            beta = fun_params[1]
            return scipy.stats.beta.cdf(d / 180, alpha, beta)
        else:
            assert False, "Invalid function type for computing update probability"

class BayesInferPredBehaviorModel(InferPredBehaviorModel):
    """
    Bayesian behavioral model for the infer-pred task.
    """
    def __init__(self, config):
        super().__init__(config)
        self.inference_model = InferPredInferenceModel(config["model"]["inference"])
        self._init_bet_space()

    def _init_bet_space(self, grid_step=0.1):
        # Define the bet space (bet location and width)
        betloc_space = bayeslearner.ContinuousUnidimensionalSpace(
            min_value=0, max_value=360, step=grid_step, include_min=True, include_max=False,
            is_angular=True)
        betwid_space = bayeslearner.DiscreteUnidimensionalSpace(
            self.config["task"]["responses"]["widths"])
        betlocwid_space = bayeslearner.MultimensionalSpace(
            [betloc_space, betwid_space])
        self.bet_space = betlocwid_space

    @property
    def hmean_dim(self):
        return self.inference_model.hmean_dim

    @property
    def hsd_dim(self):
        return self.inference_model.hsd_dim

    @property
    def has_hsd(self):
        return self.inference_model.has_hsd

    def compute_bet_given_obs(self, obs):
        _, bet = self.compute_post_dist_and_bet_given_obs(obs)
        return bet

    def compute_post_dist_and_bet_given_obs(self, obs,
        post_dist_keys=['prob']):
        """
        Given the provided observation sequence, compute, for each time step of the sequence,
        the posterior distribution over the hidden state at the next observation,
        and the bet location and width calculated from the posterior based on
        the model's response function as specified in the model configuration.

        Parameters
        ----------
        obs: 1-D array of shape (nobs,)
            Observation sequence (angular locations).
        post_dist_keys: list of strings
            Possible keys: 'prob', 'mean', 'mode', 'margprob'.

        Returns
        -------
        (post_dist, bet)
        post_dist: dictionary of arrays
            Contains the following key-value if that key was included in the post_dist_keys parameter.
            post_dist["prob"]: 2-D array of shape (nobs, n_hidden_state_values)
                Probability values of the posterior distribution.
            post_dist["mean"]: 2-D array of shape (nobs, 2)
                Mean of the posterior (first and second dimension of axis 1 correspond
                to the two dimension of the hidden state: hidden mean and hidden sd).
            post_dist["mode"]: 2-D array of shape (nobs, 2)
                Mode of the posterior
            post_dist["margprob"]: List of two 1-D arrays of shape (nobs, )
                The first and second array correspond to the marginal posterior
                distribution over the first and second dimension of the hidden state
                (hidden mean and hidden sd).
        bet: 2-D array of shape (nobs, 2)
            Bet location and width for each time step of the sequence.
        """
        if "prob" not in post_dist_keys:
            post_dist_keys += ["prob"]
        bet_fun = self.config["model"].get("bet_fun", "max_reward")
        if "mean" in bet_fun:
            post_dist_keys += ["mean"]
        if "mode" in bet_fun:
            post_dist_keys += ["mode"]
        post_dist = self.inference_model.compute_next_post_dists_dict(obs, keys=post_dist_keys)
        if bet_fun == "max_reward":
            # With this type of bet response function, the bets
            # are computed so as to maximize the expected reward.
            bet = self.compute_opt_bet_given_post_prob(post_dist['prob'])
        elif (bet_fun == "post_mode"
            or bet_fun == "post_mean"):
            # With this type of bet response function, the bets are computed
            # as a function of the mode or mean of the posterior distribution.
            stat = bet_fun[len("post_"):]
            bet = self.compute_bet_given_point_estimate(post_dist[stat])
        else:
            raise ValueError(f"bet_fun={bet_fun} Not implemented")

        if self.has_probabilistic_updates:
            # Apply probabilistic updates to the bet location and width
            # print("original bet locations:", bet[:, 0])
            bet = self.generate_probabilistic_updates(bet, self.prob_updt_fun_type,
                self.prob_updt_fun_params)

        return (post_dist, bet)

    def compute_opt_bet_given_post_prob(self, post_prob):
        """
        Compute the optimal bet location and width so as to maximize the expected reward
        incurred by the next observation under the given posterior distribution,
        which should be the posterior distribution over the hidden state at
        the time step of the next observation.

        Parameters
        ----------
        post_prob: 2-D array of shape (nobs, n_hidden_state_values)
            Posterior probability distribution over the hidden state at the time
            step of the next observation, p(h_{t+1} | x_{1:t}).
            Each entry in the array gives the probability for one time step and one
            possible value of the hidden state.

        Returns
        -------
        bet
        bet: 2-D array of shape (nobs, 2)
            Optimal bet (location and width) for each time step of the sequence.
            The first and second dimension on axis 1 of the array
            (i.e. opt_bet[:, 0] and opt_bet[:, 1]) correspond to the bet
            location and width, respectively.
        """
        # Compute the expected reward incurred by the upcoming observation given the
        # posterior probabilities over the hidden state, having observed the
        # sequence of observations x_{1:t},
        # and given a bet (v_{t+1}, w_{t+1}).
        #
        # E[l_{t+1} | x_{1:t}; v_{t+1}, w_{t+1}]
        #
        self._compute_cached_computations_if_needed()
        expct_rewards = (
            # shape (nobs, n_hidstate_values)
            post_prob
            @
            # shape (n_hidstate_values, n_bet_values)
            self._expct_rewards_given_hidstate_and_bet
            )
            # -> shape (nobs, n_bet_values)
        return self.find_opt_bet_given_expected_rewards(expct_rewards)

    def find_opt_bet_given_expected_rewards(self, expct_rewards):
        """Find the bet (location and width) that maximizes the expected reward."""
        i_bet_opt = np.argmax(expct_rewards, axis=1) # shape (nobs,)
        bet = self.bet_space.values[i_bet_opt]
        return bet

    def _compute_cached_computations_if_needed(self):
        """
        Compute, and save as attributes (to avoid repeated computation and save
        computation time), the catch probabilities and expected rewards for the
        upcoming observation given the hidden state and bet
        for each possible pair of hidden state value and bet value
        (see below for mathematical definitions).

        The results are stored in 2-D arrays of shape (n_hidstate_values, n_bet_values).
        The first dimension of the array indexes the hidden state value and the
        second dimension indexes the bet value.

        These arrays are saved as a private attributes of the object
        (self._catch_probs_given_hidstate_and_bet for catch probabilities,
        self._expct_rewards_given_hidstate_and_bet for expected rewards).

        The catch probability (i.e., the probability that the upcoming observation
        will fall in the bet interval), given the hidden state and bet, is defined
        mathematically as:
            p(v_{t+1} - w_{t+1} <= x_{t+1} <= v_{t+1} + w_{t+1}
               |  h_{t+1}; v_{t+1}, w_{t+1})

        The expected reward for the upcoming observation, given the hidden state
        and bet, is defined mathematically as:
           E[l_{t+1} | h_{t+1}; v_{t+1}, w_{t+1}]

        where:
        - x_{t+1} is the upcoming observation value
        - h_{t+1} is the hidden state at the time step of the upcoming observation
        - v_{t+1}, w_{t+1} is the bet: center location and width
        - l_{t+1} is the reward incurred by the upcoming observation
        """
        if (getattr(self, "_catch_probs_given_hidstate_and_bet", None) is not None):
            # Computation already performed. Nothing to do.
            return
        inf_gen_model = self.inference_model.generative_model
        # Compute catch probabilities
        hmean_dim = inf_gen_model.hmean_dim
        betloc_rel_to_hmean = np.array([[utils.calc.subtract_angle(bet[0], hidstate[hmean_dim])
            for bet in self.bet_space.values]
            for hidstate in inf_gen_model.hidstate_space.values]) # shape (n_h_vals, n_betlocwid_vals)
        if inf_gen_model.has_hsd:
            hsd_dim = inf_gen_model.hsd_dim
            hsd = np.array([[hidstate[hsd_dim]
                for bet in self.bet_space.values]
                for hidstate in inf_gen_model.hidstate_space.values])
        else:
            hsd = self.config["model"]["inference"]["sd"]
        betwid = np.array([[bet[1]
            for bet in self.bet_space.values]
            for hidstate in inf_gen_model.hidstate_space.values])
        betlow = betloc_rel_to_hmean - betwid / 2
        bethigh = betloc_rel_to_hmean + betwid / 2
        catch_probs = np.empty_like(betloc_rel_to_hmean)
        cdf = scipy.stats.norm(loc=0, scale=hsd).cdf
        # Take the area under the curve of the portion of the normal distribution pdf
        # covered by the bet interval within the [-180, 180] interval, taking
        # into account the circular nature of the space. The s.d.s are small
        # enough that the area under the curve of the portion of the normal 
        # distribution outside of the [-180, 180] interval can be considered 0.
        np.putmask(catch_probs,
            ((betlow >= -180) & (bethigh <= 180)), (cdf(bethigh) - cdf(betlow)))
        np.putmask(catch_probs,
            ((betlow < -180) & (bethigh <= 180)), ( (cdf(bethigh) - cdf(-180))
                                                + (cdf(180) - cdf(betlow+360))
                                                ))
        np.putmask(catch_probs,
            ((betlow >= -180) & (bethigh > 180)), ((cdf(bethigh-360) - cdf(-180))
                                                + (cdf(180) - cdf(betlow))
                                                ))
        self._catch_probs_given_hidstate_and_bet = catch_probs
        # Compute expected rewards
        reward_table = self.config["task"]["responses"]["reward_table"]
        reward_if_catched = np.array([[reward_table["catch_per_width"][bet[1]]
            for bet in self.bet_space.values]
            for hidstate in inf_gen_model.hidstate_space.values])
        reward_if_missed = np.array([[reward_table["miss_per_width"][bet[1]]
            for bet in self.bet_space.values]
            for hidstate in inf_gen_model.hidstate_space.values])
        expct_rewards = reward_if_catched * catch_probs + reward_if_missed * (1-catch_probs)
        self._expct_rewards_given_hidstate_and_bet = expct_rewards

    def compute_bet_given_point_estimate(self, hidstate_estimate):
        """
        Compute the bet location and width given a point estimate of the hidden state.
        The bet location is chosen as the location that is nearest to the
        estimate of the hidden mean (i.e., the first dimension of the hidden state)
        among the available bet locations. The bet width is chosen as the width
        that is nearest to six times the estimate of the hidden standard deviation
        (i.e., the second dimension of the hidden state), yielding a bet interval
        of +-3 standard deviations.
        """
        betlocchoices = self.bet_space.dimensions[0].values
        betwidchoices = self.bet_space.dimensions[1].values
        refloc = hidstate_estimate[:, self.hmean_dim]
        refwid = 6 * hidstate_estimate[:, self.hsd_dim]
        bet = np.empty((hidstate_estimate.shape[0], 2))
        for i, choices, refval in [(0, betlocchoices, refloc), (1, betwidchoices, refwid)]:
            choices_2D = np.expand_dims(choices, axis=0) # add dimension for time
            refval_2D = np.expand_dims(refval, axis=1) # add dimension for possible choice
            # For each time step, find the choice index that is nearest to the reference value
            i_nearest = np.abs(choices_2D - refval_2D).argmin(axis=1)
            bet[:, i] = choices[i_nearest]
        return bet

class DeltaRuleInferPredBehaviorModel(InferPredBehaviorModel):
    def __init__(self, config):
        super().__init__(config)

    def compute_bet_given_obs(self, obs):
        """
        Implements the computation of bet locations using a delta-rule
        over location prediction errors, and the computation of bet widths
        based on an estimate of the generative variance using a delta-rule
        over squared prediction errors. The bet width is then chosen
        from the available ones as the one that is closest to the
        estimated SD (i.e. the square root of the estimated variance).
        """
        initial_bet = self.initial_bet
        nobs = obs.shape[0]
        v_loc = np.empty((nobs+1))
        v_loc[0] = initial_bet[0] # self.task_config["bet_initial_location"]
        alpha_loc = self.config["model"]["alpha_loc"]
        alpha_var = self.config["model"].get("alpha_var")
        do_var = (alpha_var is not None)
        if do_var:
            v_var = np.empty((nobs+1))
            v_var[0] = initial_bet[1]**2 # self.task_config["state_space"]["hsd"]["values"][1] ** 2 
        for t in range(nobs):
            loc_pred_error = utils.calc.subtract_angle(obs[t], v_loc[t])
            loc_update = alpha_loc * loc_pred_error
            v_loc[t+1] = v_loc[t] + loc_update
            if do_var:
                var_pred_error = loc_pred_error ** 2 - v_var[t]
                v_var[t+1] = v_var[t] + alpha_var * var_pred_error

        betloc = v_loc[1:]
        if do_var:
            v_sd = np.sqrt(v_var)
            sd_levels = np.array(self.config["task"]["state_space"]["hsd"]["values"])
            betwid_i = np.abs(v_sd[:, np.newaxis] - sd_levels).argmin(axis=1)[1:]
            betwid_levels = np.array(self.config["task"]["responses"]["widths"])
            betwid = betwid_levels[betwid_i]
        else:
            betwid = np.ones((nobs)) * initial_bet[1]

        bet = np.stack((betloc, betwid), axis=1)

        if self.has_probabilistic_updates:
            # Apply probabilistic updates to the bet location and width
            bet = self.generate_probabilistic_updates(bet, self.prob_updt_fun_type,
                self.prob_updt_fun_params)

        return bet

def get_model(config):
    klass = _get_model_class(config)
    return klass(config)

def _get_model_class(config):
    if "bayes" in config["model"]["type"]:
        return BayesInferPredBehaviorModel
    elif config["model"]["type"] == "delta-rule":
        return DeltaRuleInferPredBehaviorModel

"""
Generative Models.
Any Bayes learner must be provided a generative model so that it can perform inference.
A generative model can also be used to generate sequences of observations and
hidden states.
"""

from .utils import (normalize)
import numpy as np

class GenerativeModel:
    """
    A generative model defines a hidden state space and a set of probability distributions
    that allows one to generate sequences of hidden states and observations.

    Parameters
    ----------
    hidden_state_space: Space object
        Defines the hidden state space.
    init_state_dist: 1-D array of shape (n_hidden_state_values)
        Represents p(h_0), the initial hidden state probability distribution.
        Each element in the array is the probability of the initial hidden state being equal
        to one possible hidden state value as defined in the provided hidden_state_space:
            init_state_dist[i] = p(h_0 = c_i),
        where
        c_i = hidden_state_space.values[i]
        n_hidden_state_values = hidden_state_space.n_values
    state_dynamics_dist_mat : 2-D array of shape (n_hidden_state_values, n_hidden_state_values), or None.
        Represents p(h_{t+1} | h_{t}),
        the hidden state dynamics probability distribution.
        The distribution is represented in the matrix such that:
            state_dynamics_dist_mat[i, j] = p(h_{t+1} = c_i | h_{t} = c_i)
        where c_i and c_j are the i-th and j-th value in the hidden state space.
        If None is provided, it is set to the identity matrix, meaning that the
        hidden state does not change (no dynamics).
    obs_dist: dictionary of functions described below.
        obs_dist['p_fun']: function
            Represents the observation distribution probability function,
            as a function of the two variables (observation and hidden state):
                 obs_dist_p_fun: (x, h) -> p(x | h)
             The function should return:
             - a float representing p(x | h) when the input for (x, h) are both scalars.
             - a 1D-array of floats when either x or h is a 1-D array and the other is a scalar.
         obs_dist['samp_fun']: function
            Generates one or multiple observation samples given the hidden state according
            to the observation distribution.
                obs_dist_samp_fun: h -> x ~ p(x | h)
            The function should return a scalar if the input is a scalar,
            or an array of the same size as the input if it is an array.
    """
    def __init__(self,
        hidden_state_space,
        init_state_dist,
        state_dynamics_dist_mat,
        obs_dist):
        self.hidden_state_space = hidden_state_space
        self.init_state_dist = init_state_dist
        if state_dynamics_dist_mat is None:
            state_dynamics_dist_mat = np.identity(hidden_state_space.n_values)
        self.state_dynamics_dist_mat = state_dynamics_dist_mat
        self.obs_dist_p_fun = obs_dist['p_fun']
        self.obs_dist_samp_fun = obs_dist['samp_fun']
        self._validate_attributes()

    def _validate_attributes(self):
        assert (self.init_state_dist.shape
            == (self.n_hidden_state_values, ))
        assert (self.state_dynamics_dist_mat.shape
            == (self.n_hidden_state_values, self.n_hidden_state_values))
        h = self.hidden_state_values[0]
        x_samp = self.obs_dist_samp_fun(h)
        lik = self.obs_dist_p_fun(x_samp, self.hidden_state_values)
        assert (lik.shape == (self.n_hidden_state_values, ))

    @property
    def n_hidden_state_values(self):
        return self.hidden_state_space.n_values

    @property
    def hidden_state_values(self):
        return self.hidden_state_space.values

    def generate(self, n_t):
        """
        Generate a sequence of observations and hidden states.
        """
        hid = np.empty((n_t, self.hidden_state_space.ndim))
        for t in range(n_t):
            # Sample a hidden state value given the probabilities associated
            # with each value
            if t == 0:
                # At time 0, the probabilities are given by the initial hidden
                # state distribution
                p_ht = self.init_state_dist
            else:
                # At time t, the probabilities are given by the hidden state
                # dynamics distribution given the hidden state value at time t-1
                p_ht = self.state_dynamics_dist_mat[:, i_ht_prev]
            i_ht = np.random.choice(self.n_hidden_state_values, p=p_ht)
            hid[t, ...] = self.hidden_state_values[i_ht]
            i_ht_prev = i_ht
        # Sample observations given the hidden states
        obs = self.obs_dist_samp_fun(hid)
        return obs, hid

    def generate_with_constraints(self, n_t, are_constraints_satisfied_fun=None,
        data={}):
        """
        Generate a sequence of observations and hidden states.
        Compared to the generate() method, this method adds the possibility to
        enforce additional constraints on the generated sequence of hidden states
        (constraints which are not part of the generative model itself).
        This is typically useful for generating experimental data that must
        satisfy some constraints aimed at optimizing the experimental design,
        which are unknown to subjects. To enforce the constraints, the constraints
        function is tested at each time step of the generation process,
        and a new hidden state value is generated until the constraints
        are satisfied.
        """
        hid = np.empty((n_t, self.hidden_state_space.ndim))
        for t in range(n_t):
            # Sample a hidden state value given the probabilities associated
            # with each value
            if t == 0:
                # At time 0, the probabilities are given by the initial hidden
                # state distribution
                p_ht = self.init_state_dist
            else:
                # At time t, the probabilities are given by the hidden state
                # dynamics distribution given the hidden state value at time t-1
                p_ht = self.state_dynamics_dist_mat[:, i_ht_prev]
            # Generate a new hidden state value until the constraints are satisfied
            while True:
                i_ht = np.random.choice(self.n_hidden_state_values, p=p_ht)
                ht = self.hidden_state_values[i_ht]
                if ((are_constraints_satisfied_fun is None)
                    or are_constraints_satisfied_fun(t, ht, data)):
                    break
            hid[t, ...] = ht
            i_ht_prev = i_ht
        # Sample observations given the hidden states
        obs = self.obs_dist_samp_fun(hid)
        return obs, hid


"""
The bayeslearner module implements Bayes Learner inference models.
"""

from .generative_model import GenerativeModel
from .utils import (normalize, compute_expected_value,
    compute_means_from_dist_probs_and_vals,
    compute_sds_from_dist_probs_and_vals)
import numpy as np

class BayesLearner:
    """
    The Bayes Learner class provides a generic solution to the problem of
    sequentially inferring hidden states from observations, assuming a
    given generative model. This solution is the optimal solution if the
    Bayes learner's generative model is the true generative model
    (i.e., the one that has generated the observations on which the inference
    is performed).

    Parameters
    ----------
    generative_model: GenerativeModel object
    """
    def __init__(self, generative_model):
        self.generative_model = generative_model

    @property
    def hidden_state_space(self):
        return self.generative_model.hidden_state_space

    @property
    def hidden_state_values(self):
        return self.hidden_state_space.values

    @property
    def hidden_state_ndim(self):
        return self.generative_model.hidden_state_space.ndim

    def compute_post_dists_dict(self, obs,
        keys=['prob', 'mean', 'sd']):
        post_dist_probs = self.compute_post_dists(obs)
        return self.compute_post_dists_dict_from_prob(post_dist_probs, keys=keys)

    def compute_next_post_dists_dict(self, obs,
        keys=['prob', 'mean', 'sd']):
        next_post_dist_probs = self.compute_next_post_dists(obs)
        return self.compute_post_dists_dict_from_prob(next_post_dist_probs, keys=keys)

    def compute_post_dists_dict_from_prob(self, post_dist_probs,
        keys=['prob', 'mean', 'sd']):
        out = {}
        if 'prob' in keys:
            out['prob'] = post_dist_probs
        if 'mean' in keys:
            out['mean'] = self.compute_post_means_from_dists(post_dist_probs)
        if 'sd' in keys:
            out['sd'] = self.compute_post_sds_from_dists(post_dist_probs)
        if 'margprob' in keys:
            out['margprob'] = self.compute_marginal_dists_from_joint(post_dist_probs)
        if 'mode' in keys:
            out['mode'] = self.hidden_state_values[np.argmax(post_dist_probs, axis=1)]
        return out

    def compute_post_dists(self, obs):
        """
        Returns the posterior distributions for each time step of the sequence,
        p(h_t | x_{1:t}).

        Parameters
        ----------
        obs: 1-D array of shape (n_t), or 2-D array of shape (n_t, n_dim_obs)
            The sequence of observation values.

        Returns
        -------
        post_dists : 2-D array of shape (n_t, n_hidden_state_values)
            The posterior distributions at each time step in the sequence:
                post_dists[t, i] = p(h_t = c_i | x_{1:t})
        """
        obs_dist_fun = lambda x : self.generative_model.obs_dist_p_fun(x,
            self.hidden_state_values)
        obs_liks = compute_obs_liks(obs, obs_dist_fun)
        post_dists = compute_post_dists(obs_liks,
            self.generative_model.init_state_dist,
            self.generative_model.state_dynamics_dist_mat)
        return post_dists

    def compute_next_post_dists(self, obs):
        """
        Returns, for each time step of the sequence, the posterior distribution
        over the hidden state at the next time step of the sequence, i.e.
        p(h_{t+1} | x_{1:t}).
        """
        post_dists = self.compute_post_dists(obs)
        return self.compute_next_from_current_post_dists(post_dists)

    def compute_next_from_current_post_dists(self, post_dists):
        # Transpose the post_dists array before doing the matrix multiplication,
        # so that the rows correspond to the possible states,
        # and then transpose the result back into the original shape,
        # where the rows correpond to the time steps.
        return _compute_next_from_current_post_dist(post_dists.T,
            self.generative_model.state_dynamics_dist_mat).T

    def compute_pred_dist_from_next_post(self, next_post_dists, obs_vals):
        """
        Compute the predictive posterior distribution, i.e., the posterior
        distribution over the next observation, p(x_{t+1} | x_{1:t}),
        from the posterior distribution over the hidden state at the next
        time step, p(h_{t+1} | x_{1:t}).

        Parameters
        ----------
        next_post_dists: 2D array of shape (n_t, n_hidden_state_values)
            The probabilities p(h_{t+1} | x_{1:t})
        obs_vals: 1D array of shape (n_obs_values)
            Possible observation values. This will be the support of the
            computed predictive posterior distribution.

        Returns
        -------
        pred_dists: 2D array of shape (n_t, n_obs_values)
            The predictive posterior distribution probabilities, p(x_{t+1} | x_{1:t}).
        """
        # obs_probs = p(x_{t+1} | h_{t+1}
        obs_probs = np.array([self.generative_model.obs_dist_p_fun(obs_vals, hid)
            for hid in self.hidden_state_values]) # shape (n_hidden_state_values, n_obs_values)
        # p(x_{t+1} | x_{1:t}) = \sum_{h_{t+1}} [p(x_{t+1} | h_{t+1}) p(h_{t+1} | x_{1:t})]
        pred_dists = next_post_dists @ obs_probs
        return pred_dists

    def compute_post_means_from_dists(self, post_dists):
        return compute_means_from_dist_probs_and_vals(post_dists, self.hidden_state_values,
            has_angular_dimensions=self.hidden_state_space.has_angular_dimensions,
            angular_dimension_indices=self.hidden_state_space.angular_dimension_indices)

    def compute_post_sds_from_dists(self, post_dists):
        return compute_sds_from_dist_probs_and_vals(post_dists, self.hidden_state_values,
            has_angular_dimensions=self.hidden_state_space.has_angular_dimensions,
            angular_dimension_indices=self.hidden_state_space.angular_dimension_indices)

    def compute_marginal_dists_from_joint(self, joint_dist):
        """
        Compute the marginal probability distributions over single dimensions of
        the hidden state given the joint probability distribution over the
        multidimensional hidden state.

        Parameters
        ----------
        joint_dist: 2-D array of shape (n_t, n_hidden_state_values)
            The joint probability distribution over the multidimensional hidden
            state. The last dimension of the array indexes the possible values
            of the multidimensional state. The first dimension indexes different
            time points/samples.

        Returns
        -------
        marg_dists: List of 2-D arrays
            The marginal probability distributions over each dimension of the
            hidden state (i.e. after having marginalized out all other dimensions).
            Each array in the list is the marginal probability distribution of
            one dimension. It is represented by an array whose last dimension
            indexes the possible values of the given dimension of the state.
        """
        assert joint_dist.ndim == 2
        n_t = joint_dist.shape[0]
        # Reshape the joint probability distribution array
        # into a (n+1)-dimensional array,
        # where the n last dimensions index the value of each dimension of the
        # multidimensional hidden state.
        joint_dist_nd = joint_dist.reshape(
            (n_t, ) + self.hidden_state_space.dim_nvalues)
        # Compute the marginal probability distribution over each dimension by
        # summing the probabilities of the joint distribution over all the other
        # dimensions.
        marg_dists = [None for i_dim in range(self.hidden_state_ndim)]
        for i_dim in range(self.hidden_state_ndim):
            axes_to_sum_over = (tuple(range(1, i_dim+1))
                + tuple(range(i_dim+2, joint_dist_nd.ndim)))
            marg_dists[i_dim] = np.sum(joint_dist_nd, axis=axes_to_sum_over)
        return marg_dists

def compute_post_dists(obs_liks,
    init_state_dist_vec,
    state_dynamics_dist_mat):
    """
    Compute the sequence of posterior distributions p(h_t | x_{1:t}) for
    a given sequence of observation likelihoods, given the initial hidden
    state distribution and the hidden state dynamics distribution,
    following the Bayesian filtering algorithm.

    Probability distributions over the hidden state are here represented as
    1-D arrays whose size (n_hidden_state_values) is the number of possible
    state values.

    Applying this algorithm thus requires the state space to be discretised,
    if necessary, so that a finite set of possible values for the hidden state
    has been defined and each of these values is represented by an index in the array.

    Parameters
    ----------
    obs_liks : 2-D array of shape (n_t, n_hidden_state_values)
        The sequence of observation likelihoods.
            obs_liks[t, i] = p(x_t | h_t = c_i)
        where
        n_t is the number of time steps in the sequence
        n_hidden_state_values is the number of possible hidden state values
        x_t is the observation value observed at time t
        h_t is the hidden state at time t
        c_i is the i-th value in the set of all possible values of the
        hidden state.

    init_state_dist_vec : 1-D array of shape (n_hidden_state_values) 
        Represents p(h_0),
        the initial hidden state probability distribution.
        Each element in the array is the probability the hidden state being equal
        to one possible hidden state value:
            init_state_dist[i] = p(h_0 = c_i),
        where c_i is the i-th value in the set of all possible values of the
        hidden state.

    state_dynamics_dist_mat : 2-D array of shape (n_hidden_state_values, n_hidden_state_values)
        Represents p(h_{t+1} | h_{t}),
        the hidden state dynamics probability distribution.
        The distribution is represented in the matrix such that:
            state_dynamics_dist_mat[i, j] = p(h_{t+1} = c_i | h_{t} = c_i)
        where c_i and c_j are the i-th and j-th value in the set of all
        possible values of the hidden state.

    Returns
    -------
    post_dists : 2-D array of shape (n_t, n_hidden_state_values)
        The posterior distributions at each time step in the sequence:
            post_dists[t, i] = p(h_t = c_i | x_{1:t})
    """
    
    # 0. Initialization
    n_t = obs_liks.shape[0]
    n_hidden_state_values = init_state_dist_vec.shape[0]
    post_dists = np.empty((n_t, n_hidden_state_values))
    prior_dist = init_state_dist_vec # p(h_0)
    for t in range(n_t):
        # 1. Compute the posterior for the current time step by multiplying the
        #    prior with the likelihood (Bayes' rule).
        #
        #    p(h_t | x_{1:t}) \propto p(x_t | h_t) p(h_t | x_{1:t-1})
        #
        # This step is sometimes called the "update step" (Sarkka & Svensson, 2023).
        post_dists[t] = normalize(obs_liks[t] * prior_dist)

        # 2. Compute the posterior over hidden state at the next time step,
        # which becomes the prior for the next time step of the algorithm.
        prior_dist = _compute_next_from_current_post_dist(post_dists[t], state_dynamics_dist_mat)

    return post_dists

def _compute_next_from_current_post_dist(post_dist, state_dynamics_dist_mat):
    """
    Compute the posterior over the hidden state at the next time step of the sequence
    given the the posterior over the hidden state at the current time step
    and the hidden state dynamics.

    This is done by computing the joint distribution over (h_{t+1}, h_t) and
    marginalizing over h_{t}, according to the equation:

        p(h_{t+1} | x_{1:t}) = \sum_{h_{t}} [p(h_{t+1} | h_t) p(h_t | x_{1:t})]

    This step is sometimes called the "prediction step" (Sarkka & Svensson, 2023).

    The summation is here implemented with a matrix multiplication.
    """
    return (state_dynamics_dist_mat @ post_dist)

def compute_obs_liks(obs, obs_dist_fun):
    """
    Compute the sequence of observation likelihoods
    from the sequence of observation values.

    Parameters
    ----------
    obs : 1-D or 2-D array of shape (n_t) or (n_t, n_obs_features)
        Sequence of observation values, x_{1:t}.
        1-D if the observations are scalars,
        2-D if the observations are vectors.
        n_t is the number of time steps in the sequence
        n_obs_features is the number of features of each observation if 
        observations are vectors (i.e., the dimensionality of the observation 
        vector space).

    obs_dist_fun: function
        Represents x -> p(x | h),
        the observation distribution probability function.
        obs_dist_fun(x) should return a 1-D array such that
            obs_dist_fun(x)[i] = p(x | h = c_i)


    Returns
    -------
    obs_liks : 2-D array of shape (n_t, n_hidden_state_values)
        The observation likelihoods for each time step in the sequence:
            obs_liks[t, i] = p(x_{t} | h_t = c_i)
    """

    return np.array([obs_dist_fun(obs[t]) for t in range(obs.shape[0])])


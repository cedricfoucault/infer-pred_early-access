import itertools
import numpy as np
import scipy.stats

def normalize(dist, axis=None):
    """
    Normalize the given distribution so that it sums up to 1.
    Modifies the distribution array in place and returns it for convenience.

    Parameters
    ----------
    dist: 1-D array representing a (discrete or discretized) probability distribution
        up to a normalization constant, or 2-D array representing a set of
        discrete probability distributions up to a normalization constant
    axis: if dist is a 2-D array, axis specifies the axis along which the values
        should sum up to 1 (i.e. the axis indexing the possible values of the
        random variable).

    Returns
    -------
    dist: 1-D or 2-D array representing the normalized probability distribution(s)
    """
    dist /= np.sum(dist, axis=axis)
    return dist


def compute_expected_value(values, probabilities, axis=None):
    """
    Returns the expected value of one or multiple random variables.
    The parameters specify the possible values of the random variables and
    their probabilities.

    Parameters
    ----------
    values : array-like
        Possible values of the random variables.
    probabilities : array-like
        Probabilities of the random variable being equal to each value,
        p(x = values[i]), where x denotes the random variable.
    axis : int, optional
        The axis along which the possible random variable values are indexed
        after the two input arrays (values, probabilities) have been broadcasted
        to the same shape.

    Returns
    -------
    scalar or array-like, equal to the expected value of the random variables.

    """
    return np.sum(values * probabilities, axis=axis)

def compute_means_from_dist_probs_and_vals(dist_probs, vals,
    has_angular_dimensions=False, angular_dimension_indices=None):
    """
    Compute the mean (expected value) of a random variable for each time step
    of the sequence, given its possible values and the probability of each
    value at each time step.

    Parameters
    ----------
    disp_probs: array of shape (n_t, n_values)
        dis_probs[t, j] should be the probability that the variable is equal
        the k-th value in the given array (vals[k]) at time step t.

    vals: array of shape (n_values) or (n_values, n_dim)
        The possible values of the random variable. The values can
        be scalars (in which case vals should be a 1-D array) or
        vectors (in which case vals should be a 2-D array)

    has_angular_dimensions: bool
        Indicates whether any of the dimension of the random variable
        should be treated as an angular (circular) dimension.
        The mean is computed differently on angular dimensions than
        on non-angular dimensions.

    angular_dimension_indices: list of int or None
        Indicates which dimensions of the random variable should be
        treated as angular, if any.

    Returns
    -------
    means: array of shape (n_t) or (n_t, n_dim)
        The mean (expected value) of the random variable at each time step
        of the sequence.
    """
    # insert a new axis in the value array for the time dimension in dist_probs
    values = np.expand_dims(vals, 0)
    if vals.ndim > 1:
        # insert a new axis in the probabilities array for the different
        # dimensions of the values if the values are multidimensional
        probabilities = np.expand_dims(dist_probs, 2)
    else:
        probabilities = dist_probs

    if has_angular_dimensions:
        if vals.ndim > 1:
            means = compute_expected_value(values, probabilities, axis=1)
            for i_dim in angular_dimension_indices:
                means[:, i_dim] = _compute_angular_mean_from_probs_and_vals_deg(
                    probabilities[:, :, i_dim],
                    values[:, :, i_dim], axis=1)
        else:
            means = _compute_angular_mean_from_probs_and_vals_deg(
                probabilities, values, axis=1)
    else:
        means = compute_expected_value(values, probabilities, axis=1)

    return means

def compute_sds_from_dist_probs_and_vals(dist_probs, vals,
    has_angular_dimensions=False, angular_dimension_indices=None):
    """
    Compute the standard deviation of a random variable for each time step
    of the sequence, given its possible values and the probability of each
    value at each time step.

    Parameters
    ----------
    disp_probs: array of shape (n_t, n_values)
        dis_probs[t, j] should be the probability that the variable is equal
        the k-th value in the given array (vals[k]) at time step t.

    vals: array of shape (n_values) or (n_values, n_dim)
        The possible values of the random variable. The values can
        be scalars (in which case vals should be a 1-D array) or
        vectors (in which case vals should be a 2-D array)

    has_angular_dimensions: bool
        Indicates whether any of the dimension of the random variable
        should be treated as an angular (circular) dimension.
        The SD is computed differently on angular dimensions than
        on non-angular dimensions.

    angular_dimension_indices: list of int or None
        Indicates which dimensions of the random variable should be
        treated as angular, if any.

    Returns
    -------
    sds: array of shape (n_t) or (n_t, n_dim)
        The standard deviation of the random variable at each time step
        of the sequence.
    """
    # insert a new axis in the value array for the time dimension in dist_probs
    values = np.expand_dims(vals, 0)
    if vals.ndim > 1:
        # insert a new axis in the probabilities array for the different
        # dimensions of the values if the values are multidimensional
        probabilities = np.expand_dims(dist_probs, 2)
    else:
        probabilities = dist_probs

    if has_angular_dimensions:
        if vals.ndim > 1:
            sds = compute_sds_from_dist_probs_and_vals(dist_probs, vals,
                has_angular_dimensions=False) # non-angular sds
            for i_dim in angular_dimension_indices:
                sds[:, i_dim] = _compute_angular_sd_from_probs_and_vals_deg(
                    probabilities[:, :, i_dim],
                    values[:, :, i_dim], axis=1)
        else:
            sds = _compute_angular_sd_from_probs_and_vals_deg(
                probabilities, values, axis=1)
    else:
        means = compute_means_from_dist_probs_and_vals(dist_probs, vals,
            has_angular_dimensions=False)
        sq_diffs_from_mean = (values - np.expand_dims(means, 1)) ** 2
        variances = compute_expected_value(sq_diffs_from_mean, probabilities, axis=1)
        sds = np.sqrt(variances)

    return sds

def _compute_angular_mean_from_probs_and_vals_deg(probabilities, values_deg,
    axis=1):
    """
    Compute the circular mean of a random variable with the given possible
    values, given in degrees, and probabilities, for each time step of a sequence.

    The circular mean is calculated by converting the angles to (x, y)
    coordinates with cosinus and sinus (which represents the coordinates of the
    point on the unit circle at the given angle), computing the mean (x, y)
    coordinate, and calculating the angle of the vector with those coordinate.

    Parameters
    ----------
    probabilities: array of shape (n_t, n_values)
    values_deg: array of shape (1, n_values)

    Returns
    -------
    means_deg: array of shape (n_t)
    """
    values_rad = np.deg2rad(values_deg)
    values_x = np.cos(values_rad)
    values_y = np.sin(values_rad)
    values_xy = np.stack([values_x, values_y], axis=axis+1)
    if probabilities.ndim < values_xy.ndim:
        probabilities = np.expand_dims(probabilities, axis=axis+1)
    evs_xy = compute_expected_value(values_xy,
        probabilities, axis=axis)
    evs_rad = np.arctan2(evs_xy[:, 1], evs_xy[:, 0])
    evs_rad[evs_rad < 0] += 2*np.pi # bring in the range [0, 2pi]
    evs_deg = np.rad2deg(evs_rad)
    return evs_deg

def _compute_angular_sd_from_probs_and_vals_deg(probabilities, values_deg,
    axis=1):
    """
    Compute the circular standard deviation of a random variable with the given
    possible values, given in degrees, and probabilities, for each time step of
    a sequence.

    The circular SD is calculated as detailed in these references:
    - https://en.wikipedia.org/wiki/Directional_statistics
    - scipy.stats.circstd

    Parameters
    ----------
    probabilities: array of shape (n_t, n_values)
    values_deg: array of shape (1, n_values)

    Returns
    -------
    sds_deg: array of shape (n_t)
    """
    values_rad = np.deg2rad(values_deg)
    values_x = np.cos(values_rad)
    values_y = np.sin(values_rad)
    values_xy = np.stack([values_x, values_y], axis=axis+1)
    if probabilities.ndim < values_xy.ndim:
        probabilities = np.expand_dims(probabilities, axis=axis+1)
    evs_xy = compute_expected_value(values_xy,
        probabilities, axis=axis)
    Rs = np.sqrt(evs_xy[:, 0]**2 + evs_xy[:, 1]**2)
    Rs = np.minimum(Rs, 1)
    sds_rad = np.sqrt(-2*np.log(Rs))
    sds_deg = np.rad2deg(sds_rad)
    return sds_deg

#
# Functions to create distribution objects for specific types of distributions
#

def uniform_dist(n):
    """
    Probability masses of a discrete uniform distribution with n possible values.
    """
    return np.ones(n) / n

def norm_dist_p_fun(x, mu, sigma):
    """
    Probability distribution function of a normal distribution with
    mean mu and s.d. sigma.
    """
    return scipy.stats.norm.pdf(x, loc=mu, scale=sigma)


def norm_dist_samp_fun(mu, sigma):
    """
    Returns samples from a normal distribution with mean mu and s.d. sigma.
    """
    return scipy.stats.norm.rvs(loc=mu, scale=sigma)

def truncnorm_dist_p_fun(x, mu, sigma, min, max):
    """
    Probability distribution function of a truncated normal distribution with
    mean mu and s.d. sigma, truncated on the interval [min, max].
    """
    a, b = truncnorm_ab_parameters(mu, sigma, min, max)
    return scipy.stats.truncnorm.pdf(x, a, b, loc=mu, scale=sigma)


def truncnorm_dist_samp_fun(mu, sigma, min, max):
    """
    Returns samples from a truncated normal distribution with mean mu and s.d. sigma,
    truncated on the interval [min, max].
    """
    a, b = truncnorm_ab_parameters(mu, sigma, min, max)
    return scipy.stats.truncnorm.rvs(a, b, loc=mu, scale=sigma)

def truncnorm_ab_parameters(mu, sigma, min, max):
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return a, b

def circnorm_dist_p_fun(x, mu, sigma, period=360):
    """
    Probability distribution function of a circular normal distribution with mean mu
    and s.d. sigma, assuming that sigma is small enough p(|x-mu| > period/2)
    can be considered 0. (For reference, at sigma=30°, the largest stochasticity
    level in the continuous inference study, p(|x-mu| > period/2) = 1e-9.)
    """
    angle_diff = subtract_angle(x, mu, period=period)
    return scipy.stats.norm.pdf(angle_diff, loc=0, scale=sigma)

def circnorm_dist_samp_fun(mu, sigma, period=360):
    """
    Returns the remainder after division by the period of a sample from a normal
    distribution with mean mu and s.d. sigma. (For reference, this is how
    the observed stimuli were generated in the continuous inference task.)
    """
    x = norm_dist_samp_fun(mu, sigma)
    # Note that np.mod is an alias for np.remainder, and is the equivalent of
    # MATLAB's mod() function
    return np.mod(x, period)

def subtract_angle(a, b, period=360):
   return ((a - b) + period/2) % period - period/2

def bernoulli_dist_p_fun(x, p):
    """
    Probability distribution function of a Bernoulli distribution with parameter p.
    """
    return scipy.stats.bernoulli.pmf(x, p)

def bernoulli_dist_samp_fun(p):
    """
    Returns samples from a Bernoulli distribution with parameter p
    """
    return scipy.stats.bernoulli.rvs(p)

def obs_dist(dist_params_for_state_fun, p_fun, samp_fun):
    """
    Construct hidden state-dependent observation distribution functions (probability and sampling function)
    given:
    - a function that maps a latent/hidden state h to distribution parameters, and
    - the base distribution functions (a probability/density/mass function and a sampling function)
      that take as input those distribution parameters.

    Parameters
    ----------
    dist_params_for_state_fun :
        Function that takes a state h and returns a dict of keyword arguments for the
        underlying distribution (e.g., {"loc": ..., "scale": ...} or {"p": ...}).
    p_fun :
        Base probability/density/mass function with signature p_fun(x, **params)
        that evaluates the distribution at observation x given params.
    samp_fun : Callable[..., Any]
        Base sampling function with signature samp_fun(**params) that draws a sample
        from the distribution defined by params.

    Returns
    -------
    dict
        A dictionary with:
          - "p_fun": Callable[[Any, Any], float]
              A function p(x, h) that evaluates the probability/density/mass at x
              for parameters derived from h, i.e., p_fun(x, **dist_params_for_state_fun(h)).
          - "samp_fun":
              A function sample(h) that draws a sample from the distribution whose
              parameters are derived from h, i.e., samp_fun(**dist_params_for_state_fun(h)).

    Examples
    --------
    >>> # Example with a normal distribution (you can use the `norm_dist` function instead but this shows how to use `obs_dist` to achieve the same result):
    >>> import numpy as np
    >>> from math import exp, sqrt, pi
    >>>
    >>> def dist_params_for_state(h):
    ...     return {"mu": h[0], "sigma": h[1]}
    >>>
    >>> def normal_pdf(x, *, mu, sigma):
    ...     return scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    >>>
    >>> def normal_sample(*, mu, sigma):
    ...     return scipy.stats.norm.rvs(loc=mu, scale=sigma)
    >>>
    >>> obs = obs_dist(dist_params_for_state, normal_pdf, normal_sample)
    >>> h = [0.0, 1.0] # 2D hidden state: first dimension is mean, second is sd
    >>> p_x_given_h = obs["p_fun"](0.2, h)   # density at x=0.2 given state h
    >>> x_sample = obs["samp_fun"](h)        # draw a sample given state h
    """
    p_fun_of_xh = lambda x, h: p_fun(x, **(dist_params_for_state_fun(h)))
    samp_fun_of_h = lambda h: samp_fun(**(dist_params_for_state_fun(h)))
    return dict(p_fun=p_fun_of_xh, samp_fun=samp_fun_of_h)

def norm_dist(dist_params_for_state_fun):
    return obs_dist(dist_params_for_state_fun, norm_dist_p_fun, norm_dist_samp_fun)

def truncnorm_dist(dist_params_for_state_fun):
    return obs_dist(dist_params_for_state_fun, truncnorm_dist_p_fun, truncnorm_dist_samp_fun)

def circnorm_dist(dist_params_for_state_fun):
    return obs_dist(dist_params_for_state_fun, circnorm_dist_p_fun, circnorm_dist_samp_fun)

def bernoulli_dist(dist_params_for_state_fun):
    return obs_dist(dist_params_for_state_fun, bernoulli_dist_p_fun, bernoulli_dist_samp_fun)

def construct_dynamics_dist(state_space, p_fun):
    """
    Constructs and returns the matrix representing the state dynamics distribution
    for the given state space and (unnormalized) probability function.

    Parameters
    ----------
    state_space: instance of Space 
        The state space. The index of the values in state_space.values determines
        the index of the pairs of values in the dynamics distribution matrix.
    p_fun: function
        The conditional probability function: p(h_{t+1} | h_{t}).
        The given function should have the following signature:
            p_fun(h_from, h_to, i_from_ND, i_to_ND) -> float
        where:
        - h_from is the value of the state at time t, h_{t}
        - h_to is the value of the state at time t+1, h_{t+1}
        - i_from_ND is the n-dimensional index (list of int)
            corresponding to value h_from,
            as given by the state space's get_index_ND() function.
        - i_to_ND is the n-dimensional index corresponding to value h_to,
            as given by the state space's get_index_ND() function.
        The indices provided by i_from_ND and i_to_ND can be used to
        calculate p_fun based on other matrices that represent the dynamics
        for a given dimension of the state.

    Returns
    -------
    dynamics_dist_mat : 2-D array of shape (n_state_values, n_state_values)
        Matrix representing p(h_{t+1} | h_{t}) in the following way:
            dynamics_dist_mat[i, j] = p(h_{t+1} = v_i | h_{t} = v_j)
        where:
        - v_i is the ith value of the state in the state space, i.e. state_space.values[i]
        - v_j is the ith value of the state in the state space, i.e. state_space.values[j]
    """
    dynamics_dist_mat = np.empty((state_space.n_values, state_space.n_values))
    for i_from, i_to in itertools.product(range(state_space.n_values), 
        range(state_space.n_values)):
        h_from = state_space.values[i_from]
        h_to = state_space.values[i_to]
        i_from_ND = state_space.get_index_ND(i_from)
        i_to_ND = state_space.get_index_ND(i_to)
        dynamics_dist_mat[i_to, i_from] = p_fun(h_from, h_to, i_from_ND, i_to_ND)
    # Make sure the probabilities p(h_to | h_form) sum up to 1 along the h_to values
    dynamics_dist_mat = normalize(dynamics_dist_mat, axis=0)
    return dynamics_dist_mat

def construct_init_dist(state_space, p_fun):
    """
    Constructs and returns the matrix representing the initial state distribution
    for the given state space and (unnormalized) probability function.

    Parameters
    ----------
    state_space: instance of Space 
        The state space. The index of the values in state_space.values determines
        the index of the pairs of values in the dynamics distribution matrix.
    p_fun: function
        The probability function: p(h_{0}).
        The given function should have the following signature:
            p_fun(h_0) -> float
        where:
        - h_0 is the value of the state at time 0, h_{0}.

    Returns
    -------
    init_dist : 1-D array of shape (n_state_values) representing p(h_{0}) in the following way:
            init_dist[i] = p(h_{0} = v_i)
        where:
        - v_i is the ith value of the state in the state space, i.e. state_space.values[i]
    """
    init_dist = np.empty((state_space.n_values,))
    for i_0 in range(state_space.n_values):
        h_0 = state_space.values[i_0]
        init_dist[i_0] = p_fun(h_0)
    # Make sure the probabilities sum up to 1
    init_dist = normalize(init_dist, axis=0)
    return init_dist

def change_point_dynamics_dist(n_state_values, p_cp, is_allowed_change_fun=None):
    """
    Returns a matrix representing a change-point dynamics distribution.
    This function works flexibly for both unidimensional and multidimensional state spaces,
    interpreting the state space as either uni- or multi-dimensional depending whether
    the n_state_values input parameter is a scalar or not.
    See change_point_dynamics_dist_1D_state() and change_point_dynamics_dist_ND_state()
    for the specification of this function in case of a unidimensional state and multidimensional state,
    respectively.
    """
    if np.isscalar(n_state_values):
        return change_point_dynamics_dist_1D_state(n_state_values, p_cp, is_allowed_change_fun)
    else:
        return change_point_dynamics_dist_ND_state(n_state_values, p_cp, is_allowed_change_fun)

def change_point_dynamics_dist_1D_state(n_state_values, p_cp, is_allowed_change_fun=None):
    """
    Returns a matrix representing a change-point dynamics distribution for
    a unidimensional state space with n_state_values possible values,
    a probability p_cp of a change point occurring, whose possible value changes
    are indicated by the is_allowed_change_fun function.

    Parameters
    ----------
    n_state_values: int
        The number of possible state values
    p_cp: float
        The probability of a change point occurring
    is_allowed_change_fun: function, optional
        This function should take as input a pair of index (i_from, i_to) indexing
        which value the state will go from and to, respectively, and return
        a boolean indicating whether the state can change from that value
        to that other value when a change point occurs.
        If None, defaults to all changes allowed, including going from and to
        the same value.

    Returns
    -------
    dynamics_dist_mat : 2-D array of shape (n_state_values, n_state_values)
        Matrix representing p(h_{t+1} | h_{t}) in the following way:
            dynamics_dist_mat[i, j] = p(h_{t+1} = v_i | h_{t} = v_j)
    """
    if is_allowed_change_fun is None:
        is_allowed_change_fun = lambda i_from, i_to: True
    mat_no_cp = np.identity(n_state_values)
    mat_cp = normalize(np.array([[1. if is_allowed_change_fun(i_col, i_row) else 0.
        for i_col in range(n_state_values)]
        for i_row in range(n_state_values)]), axis=1)
    mat = (1 - p_cp) * mat_no_cp + p_cp * mat_cp
    return mat

def change_point_dynamics_dist_ND_state(n_state_values_eachdim,
    p_cp_eachdim, is_allowed_change_fun_eachdim=None):
    """
    Returns a matrix representing a change-point dynamics distribution for
    a multidimensional state space, with n_state_values_eachdim possible values
    for each dimension, independent probabilities p_cp_eachdim of a change point
    occurring for each dimension, each of which can producing a value change
    in the corresponding dimension among the possible changes indicated by the
    is_allowed_change_fun function for that dimension.

    Parameters
    ----------
    n_state_values_eachdim: array-like of int
        The number of possible values for each dimension of the state
    p_cp_eachdim: array-like of float
        The probability of a change point occurring along each dimension of the state
    is_allowed_change_fun_eachdim: list of function, optional
        The list should have one function per dimension.
        Each function should take as input a pair of index (i_from, i_to) indexing
        which value the dimension of the state will go from and to, respectively,
        and return a boolean indicating whether the state can change from that value
        to that other value when a change point along that dimension occurs.
        If None, defaults to, for each dimension, all changes allowed,
        including going from and to the same value.

    Returns
    -------
    dynamics_dist_mat : 2-D array of shape (n_state_values, n_state_values)
        Matrix representing p(h_{t+1} | h_{t}) in the following way:
            dynamics_dist_mat[i, j] = p(h_{t+1} = v_i | h_{t} = v_j)
        where n_state_values is equal to the product of the number of possible state
        values along all the dimensions.
        The values of the multidimensional state are enumerated in the matrix
        consistently with how itertools.product() enumerates them when given the list of
        possible state values for each dimension.

    """
    n_dims = len(n_state_values_eachdim)
    if is_allowed_change_fun_eachdim is None:
        is_allowed_change_fun_eachdim = [None] * n_dims
    change_point_dynamics_dist_eachdim = [change_point_dynamics_dist_1D_state(
        n_state_values_eachdim[i_dim], p_cp_eachdim[i_dim],
        is_allowed_change_fun_eachdim[i_dim]) for i_dim in range(n_dims)]
    return dynamics_dist_ND_from_1D(change_point_dynamics_dist_eachdim)

def dynamics_dist_2D_from_1D(dynamics_dist_mat_dim1, dynamics_dist_mat_dim2):
    """
    See dynamics_dist_ND_from_1D(), which is a generalization
    of this function from two dimensions to any number of dimensions for the
    state space. The 2-D specific version is kept here because its implementation
    is easier to understand, and helps understand the generalized version.
    """
    n_state_values_dim1 = dynamics_dist_mat_dim1.shape[0]
    n_state_values_dim2 = dynamics_dist_mat_dim2.shape[0]
    n_state_values = n_state_values_dim1 * n_state_values_dim2
    dynamics_dist_mat = np.empty(n_state_values, n_state_values)

    i_possible_states_tuples = itertools.product(
        np.arange(n_state_values_dim1),
        np.arange(n_state_values_dim2))
    for i_from, (i_from_dim1, i_from_dim2) in enumerate(i_possible_states_tuples):
        for i_to, (i_to_dim1, i_to_dim2) in enumerate(i_possible_states_tuples):
            dynamics_dist_mat[i_to, i_from] = (
                dynamics_dist_mat[i_to_dim1, i_from_dim1]
                * dynamics_dist_mat[i_to_dim2, i_from_dim2])

    return dynamics_dist_mat

def dynamics_dist_ND_from_1D(dynamics_dist_mat_eachdim):
    """
    Returns a matrix representing the dynamics distribution for a multidimensional
    state, whose dimensions each follow an independent dynamics distribution
    which is described by the provided matrix in the list.

    Parameters
    ----------
    dynamics_dist_mat_eachdim: list of 2-D arrays
        The i-th element in the list is the matrix representing the dynamics
        distribution of the i-th dimension in the state.

    Returns
    -------
    dynamics_dist_mat : 2-D array
        Square matrix representing the dynamics distribution of the multidimensional
        state. The size of the matrix is the product of the sizes of the matrices
        of each dimension.
    """
    n_dims = len(dynamics_dist_mat_eachdim)
    n_state_values_eachdim = [mat.shape[0] for mat in dynamics_dist_mat_eachdim]
    n_state_values = np.prod(n_state_values_eachdim)
    dynamics_dist_mat = np.empty((n_state_values, n_state_values))

    i_possible_states_tuples = list(itertools.product(
        *[np.arange(n_state_values_eachdim[i_dim]) for i_dim in range(n_dims)]))
    for i_from, i_from_eachdim in enumerate(i_possible_states_tuples):
        for i_to, i_to_eachdim in enumerate(i_possible_states_tuples):
            dynamics_dist_mat[i_to, i_from] = np.prod([
                dynamics_dist_mat_eachdim[i_dim][i_to_eachdim[i_dim], i_from_eachdim[i_dim]]
                for i_dim in range(n_dims)])

    return dynamics_dist_mat

def gaussian_random_walk_dynamics_dist(state_values, process_noise_sd):
    """
    Returns a matrix representing a Gaussian random walk dynamics distribution.

    Parameters
    ----------
    state_values: 1-D array-like or list of 1-D array-like
        The possible values of the state after discretization of the state space.
        This should be a 1-D array if the state space is unidimensional,
        and a list of 1-D array if the state space is multidimensional, with
        each element in the list indexing one dimension of the state.
    process_noise_sd: scalar or 1-D array
        The process s.d. parameterizing the volatility of the random walk.
        This should be a scalar for a unidimensional state space, and a list
        of scalars for a multidimensional state space, each being the process
        the process s.d. for one dimension, considering that here,
        in the multidimensional case, we assume that each dimension follows
        an independent process such that the dimensions are uncorrelated
        with each other.

    Returns
    -------
    dynamics_dist_mat : 2-D array
        Square matrix representing the dynamics distribution p(h_{t+1} | h_{t}).
            dynamics_dist_mat[i, j] = p(h_{t+1} = v_i | h_{t} = v_j)
    """
    if np.isscalar(state_values[0]):
        return gaussian_random_walk_dynamics_dist_1D_state(state_values, process_noise_sd)
    else:
        # TBD: Test that the multidimensional case works correctly
        dynamics_dist_mat_eachdim = [
            gaussian_random_walk_dynamics_dist_1D_state(
                state_values[i_dim], process_noise_sd[i_dim])
            for i_dim in len(state_values)]
        return dynamics_dist_ND_from_1D(dynamics_dist_mat_eachdim)

def gaussian_random_walk_dynamics_dist_1D_state(state_values, process_noise_sd):
    dynamics_dist_mat = normalize(np.array([[scipy.stats.norm.pdf(
            state_values[i_row], loc=state_values[i_col],
            scale=process_noise_sd)
        for i_col in range(len(state_values))]
        for i_row in range(len(state_values))]), axis=0)
    return dynamics_dist_mat

def lognormal_random_walk_dynamics_dist_1D_state(state_values, process_noise_sd):
    dynamics_dist_mat = normalize(np.array([[scipy.stats.norm.pdf(
            np.log(state_values[i_row]), loc=np.log(state_values[i_col]),
            scale=process_noise_sd)
        for i_col in range(len(state_values))]
        for i_row in range(len(state_values))]), axis=0)
    return dynamics_dist_mat

def circnorm_random_walk_dynamics_dist(state_values, process_noise_sd):
    """
    Same as gaussian_random_walk_dynamics_dist() for a random walk using
    a circular normal distribution rather than a regular normal distribution.
    See gaussian_random_walk_dynamics_dist() documentation for more info.
    """
    if np.isscalar(state_values[0]):
        return circnorm_random_walk_dynamics_dist_1D_state(state_values, process_noise_sd)
    else:
        # TBD: Test that the multidimensional case works correctly
        dynamics_dist_mat_eachdim = [
            circnorm_random_walk_dynamics_dist_1D_state(
                state_values[i_dim], process_noise_sd[i_dim])
            for i_dim in len(state_values)]
        return dynamics_dist_ND_from_1D(dynamics_dist_mat_eachdim)

def circnorm_random_walk_dynamics_dist_1D_state(state_values, process_noise_sd):
    dynamics_dist_mat = normalize(np.array([[circnorm_dist_p_fun(
        state_values[i_row], state_values[i_col], process_noise_sd)
        for i_col in range(len(state_values))]
        for i_row in range(len(state_values))]), axis=0)
    return dynamics_dist_mat

def binary_reversal_dynamics_dist(p_rev):
    """
    Generate a binary reversal dynamics transition matrix.
    This function creates a 2x2 transition matrix for binary states where each state
    can transition to the other with a given reversal probability.
    Parameters
    ----------
    p_rev : float
        The probability of reversal/transition between states. Must be in [0, 1].
        - p_rev = 0: no transitions occur (identity matrix)
        - p_rev = 1: states always flip
        - p_rev = 0.5: uniform transition probability
    Returns
    -------
    numpy.ndarray
        A 2x2 transition matrix where:
        - Element [i, i] represents the probability of staying in state i
        - Element [i, j] (i ≠ j) represents the probability of transitioning from state i to state j
    Examples
    --------
    >>> binary_reversal_dynamics_dist(0.3)
    array([[0.7, 0.3],
           [0.3, 0.7]])
    """
    return np.array([[(1-p_rev), p_rev], [p_rev, (1-p_rev)]])

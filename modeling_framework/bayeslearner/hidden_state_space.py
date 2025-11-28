"""
The hidden state space specifies the set of all possible values that the
hidden state can take.

Here, the hidden state space is defined as a discrete, finite set of values,
which can be scalar values or vector values depending on the dimension
(number of variables) of the hidden state (scalar if the state consists of only
one variable/dimension, vector if the state consists of multiple variables/dimensions).

The variables of the hidden state that are originally continuous are discretized
using a grid of equally spaced values over the domain of the variable.

This module may be generalized in the future to "Space" and encompass the
hidden state space as well as the observation space.
"""

import itertools
import numpy as np

class Space:
    """
    Parameters
    ----------
    ndim : int
        Number of dimensions of the space
    values: array of shape (n_values, ndim) or (n_values) if ndim == 1
        The set of all possible values in the space.

    Attributes
    ----------
    n_values : int
        The number of values in the space.
    has_angular_dimensions: bool
        Whether any dimension of the space represents angle.
    angular_dimension_indices: list or None
        A list of dimension indices of the space that represent angle, if any.
    """
    def __init__(self, values, ndim):
        self.values = np.array(values)
        self.ndim = ndim

    @property
    def n_values(self):
        return self.values.shape[0]

    @property
    def has_angular_dimensions(self):
        assert False, "Should be implemented by subclasses"

    @property
    def angular_dimension_indices(self):
        assert False, "Should be implemented by subclasses"

    def get_index_ND(self, i_val):
        """
        Get the n-dimensional index corresponding to the value in the state
        whose index in the .values array is equal to the given index.

        Parameters
        ----------
        i_val: int
            Index of the value of the state in the .values array

        Returns
        -------
        i_val_nd: list of int
            The i-th element in this list should be the index of the i-th component
            of the given value in the i-th dimension of the state space.
        """
        assert False, "Should be implemented by subclasses"
    

class UnidimensionalSpace(Space):
    """
    Base class that defines the space for a unidimensional hidden state
    or a single dimension of a multidimensional hidden state.

    There are two kinds of spaces: discrete spaces, or continuous spaces
    that are discretized for the purpose of implementing inference.
    Each of them are defined in a subclass. The subclasses should be used
    to create instances of unidimensional spaces. The base class
    defines the common attributes.

    Attributes
    ----------
    values : 1D array of shape (n_values)
        The set of all possible values that this dimension can take
    is_continuous : bool
        Indicates whether the original space is continuous or not.
    is_angular : bool
        Whether the values of the space represent angles.
        When values are angles, basic calculations such as calculating differences,
        means, etc., should be handled differently than for non-circular spaces.
    """
    def __init__(self, values, is_continuous, is_angular=False):
        self.is_continuous = is_continuous
        self.is_angular = is_angular
        super().__init__(values, ndim=1)

    @property
    def has_angular_dimensions(self):
        return self.is_angular

    @property
    def angular_dimension_indices(self):
        return [0] if self.is_angular else None

    def get_index_ND(self, i_val):
        return [i_val]

class DiscreteUnidimensionalSpace(UnidimensionalSpace):
    """
    Represents an intrinsically discrete unidimensional space.

    Parameters
    ----------
    values : array_like of shape (n_values)
        The set of all possible values that this dimension can take
    """
    def __init__(self, values):
        super().__init__(values, is_continuous=False)


class ContinuousUnidimensionalSpace(UnidimensionalSpace):
    """
    Represents a unidimensional space that is intrinsically continuous,
    and is discretized on a grid for the purpose of the implementation.

    Parameters
    ----------
    min_value : scalar
    max_value : scalar
        Range in which values of the space are included.
    n_samples : int, optional
        The number of values to sample from the continuous range.
        These will be evenly spaced values.
        This parameter is mutually exclusive with 'step'.
        Exactly one of those two parameters must be provided.
    step : scalar, optional
        The space between each contiguous sampled values.
        This parameter is mutually exclusive with 'n_samples'.
        Exactly one of those two parameters must be provided.
    include_min : bool
    include_max : bool
        Whether include the minimum/maximum value of the domain in the discrete
        grid-based representation.
    is_angular : bool
        Whether the values of the space represent angles.
        When values are angles, basic calculations such as calculating differences,
        means, etc., should be handled differently than for non-circular spaces.
    """
    def __init__(self, min_value, max_value, step=None,
        include_min=True, include_max=True, is_angular=False,
        n_samples=None,
        sample_fun=None,
        logbase=2):
        # Generate a grid of evenly spaced values, including or not the minimum
        # and the maximum of the defined range as specified by the user.
        if n_samples is not None:
            num = n_samples
            start = 0
            stop = None # i.e. stop at the end of the sequence, including the last element
            if not include_min:
                num += 1
                start = 1
            if not include_max:
                num += 1
                stop = -1
            if sample_fun is None or sample_fun == "linspace":
                values = np.linspace(min_value, max_value, num=num)[start:stop]
            elif sample_fun == "log2space":
                values = np.logspace(np.log2(min_value), np.log2(max_value), num=num, base=2)[start:stop]
        elif step is not None:
            start = min_value if include_min else min_value + step
            stop = max_value + step/2 if include_max else max_value
            values = np.arange(start, stop, step)
        super().__init__(values, is_continuous=True, is_angular=is_angular)

class MultimensionalSpace(Space):
    """
    Defines the space for a multidimensional hidden state.

    Parameters
    ----------
    dimensions: list of UnidimensionalSpace objects
        Defines the space for each dimension that make up the multidimensional space.

    Attributes
    ----------
    dimensions
    ndim : int
        Number of dimensions of the space
    values: array of size (n_values_1 * … * n_values_ndim, n_dim)
        This represents the set of all possible vector values in the multidimensional space
        obtained by taking all possible combination of values across each dimension.
        values[i] is one such possible vector value, represented by a 1D array of shape (n_dim).
    dim_nvalues: tuple of integers (n_values_1, ..., n_values_ndim)
        Number of values that each dimension can take
    dimension_values : list of 1D arrays whose num. of elements are n_values_1, ..., n_values_ndim.
        dimension_values[i] consists of all possible values that the ith-dimension
        can take.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.dim_nvalues = tuple((dim.n_values for dim in dimensions))
        values = np.array([list(tup) for tup in itertools.product(*self.dimension_values)])
        ndim = len(dimensions)
        super().__init__(values, ndim=ndim)

    @property
    def dimension_values(self):
        return [dim.values for dim in self.dimensions]

    @property
    def has_angular_dimensions(self):
        return any([d.is_angular for d in self.dimensions])

    @property
    def angular_dimension_indices(self):
        return [i for i, d in enumerate(self.dimensions) if d.is_angular]

    def get_index_ND(self, i_val):
        """
        Get the n-dimensional index corresponding to the value in the state
        whose index in the .values array is equal to the given index.

        Parameters
        ----------
        i_val: int
            Index of the value of the state in the .values array

        Returns
        -------
        i_val_nd: list of int
            The i-th element in this list should be the index of the i-th component
            of the given value in the i-th dimension of the state space.
        """
        if getattr(self, "_nd_indices", None) is None:
            # Compute a table mapping the 1-d index to the n-d index and cache
            # it in an attribute so that it is not recomputed on subsequent
            # calls.
            self._nd_indices = np.array(
                list(itertools.product(*[range(dim.n_values)
                    for dim in self.dimensions])))
        return self._nd_indices[i_val]

if __name__ == '__main__':
    # Example use
    source_mean_space = ContinuousUnidimensionalSpace(0, 360, 36,
        include_min=True, include_max=False)
    source_sd_space = DiscreteUnidimensionalSpace([10, 20, 30])
    hidden_state_space = MultimensionalSpace([source_mean_space, source_sd_space])
    print("possible values for the hidden state", hidden_state_space.values)
    print("hidden_state_space.values.shape", hidden_state_space.values.shape)
    
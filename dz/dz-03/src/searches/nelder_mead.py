# Copyright 2020 Yalfoosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Optional, Tuple, Union

import numpy as np

from . import constants
from .function import Function


def clean_nelder_mead_simplex_search_arguments(
    function: Function,
    alpha: float,
    beta: float,
    gamma: float,
    sigma: float,
    use_jakobovic_expand: bool,
    epsilon: float,
    max_iterations: int,
    verbosity: Optional[str],
    decimal_precision: int,
) -> Tuple[Function, float, float, float, float, bool, float, int, int, int]:
    """
    Checks the Nelder Mead Simplex Search arguments and returns them prepared for work.

    Args:
        function (Function): A Function representing the loss function.
        alpha (float): A float used in point reflection.
        beta (float): A float used in point contraction.
        gamma (float): A float used in point expansion.
        sigma (float): A float used when moving points to the optimum.
        use_jakobovic_expand (bool): A bool determining whether or not to use the
        __expand_jakobovic method instead of the __expand method for point expansion.
        Defaults to False.
        epsilon (float): A float representing the error threshold.
        max_iterations (int): An int representing the maximum number of iterations
        before the algorithm times out and returns the last found optimum.
        verbosity (Optional[str]): A str representing the verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.

    Raises:
        TypeError: Raised if argument function is not a Function.
        TypeError: Raised if argument alpha is not a float.
        TypeError: Raised if argument beta is not a float.
        TypeError: Raised if argument gamma is not a float.
        TypeError: Raised if argument sigma is not a float.
        TypeError: Raised if argument use_jakobovic_expand is not a bool.
        TypeError: Raised if argument epsilon is not a float.
        ValueError: Raised if argument epsilon is a negative number.
        TypeError: Raised if argument max_iterations is not an int.
        ValueError: Raised if argument max_iterations is a negative number.
        TypeError: Raised if argument verbosity is not a str.
        KeyError: Raised if argument verbosity is an invalid key.
        TypeError: Raised if argument decimal_precision is not an int.
        ValueError: Raised if argument decimal_precision is a negative number.

    Returns:
        Tuple[Function, float, float, float, float, bool, float, int, int, int]: Cleaned
        arguments.
    """
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    if isinstance(alpha, int):
        alpha = float(alpha)

    if not isinstance(alpha, float):
        raise TypeError(
            "Expected argument alpha to be a float, instead it is " f"{type(alpha)}."
        )

    if isinstance(beta, int):
        beta = float(beta)

    if not isinstance(beta, float):
        raise TypeError(
            "Expected argument beta to be a float, instead it is " f"{type(beta)}."
        )

    if isinstance(gamma, int):
        gamma = float(gamma)

    if not isinstance(gamma, float):
        raise TypeError(
            "Expected argument gamma to be a float, instead it is " f"{type(gamma)}."
        )

    if isinstance(sigma, int):
        sigma = float(sigma)

    if not isinstance(sigma, float):
        raise TypeError(
            "Expected argument sigma to be a float, instead it is " f"{type(sigma)}."
        )

    if not isinstance(use_jakobovic_expand, bool):
        raise TypeError(
            "Expected argument use_jakobovic_expand to be a bool, instead it is "
            f"{type(use_jakobovic_expand)}."
        )

    if not isinstance(epsilon, float):
        raise TypeError(
            "Expected argument epsilon to be a float, instead it is "
            f"{type(epsilon)}."
        )

    if epsilon < 0:
        raise ValueError(
            "Expected argument epsilon to be a positive float, instead it is "
            f"{epsilon}."
        )

    if not isinstance(max_iterations, int):
        raise TypeError(
            "Expected argument max_interations to be an int, instead it is "
            f"{type(max_iterations)}."
        )

    if max_iterations < 1:
        raise ValueError(
            "Expected argument max_interations to be a positive integer, instead it is "
            f"{max_iterations}."
        )

    if verbosity is None:
        verbosity = "none"

    if not isinstance(verbosity, str):
        raise TypeError(
            f"Expected argument verbosity to be a str, instead it is {type(verbosity)}."
        )

    if verbosity not in constants.NELDER_MEAD_SIMPLEX_VERBOSITY_DICT:
        verbosity_dict_length = len(constants.NELDER_MEAD_SIMPLEX_VERBOSITY_DICT)

        if verbosity_dict_length == 0:
            verbosity_string = "There are no keys available."
        elif verbosity_dict_length == 1:
            _key = list(constants.NELDER_MEAD_SIMPLEX_VERBOSITY_DICT.keys())[0]
            verbosity_string = f'The only available key is "{_key}".'
        else:
            _keys = list(sorted(constants.NELDER_MEAD_SIMPLEX_VERBOSITY_DICT.keys()))
            verbosity_string = "The available keys are "
            verbosity_string += ", ".join([str(f'"{x}"') for x in _keys[:-1]])
            verbosity_string += f' and "{_keys[-1]}"".'

        raise KeyError(
            f'Verbosity key "{verbosity}" is not in the Nelder Mead Simplex Verbosity '
            f"dictionary. {verbosity_string}"
        )

    verbosity = constants.NELDER_MEAD_SIMPLEX_VERBOSITY_DICT[verbosity]

    if not isinstance(decimal_precision, int):
        raise TypeError(
            "Expected argument decimal_precision to be an int, instead it is "
            f"{type(decimal_precision)}."
        )

    if decimal_precision < 1:
        raise ValueError(
            "Expected argument decimal_precision to be a positive int, instead it is"
            f"{decimal_precision}."
        )

    return (
        function,
        alpha,
        beta,
        gamma,
        sigma,
        use_jakobovic_expand,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    )


def clean_get_simplex_points(
    start: np.ndarray, stride: Union[float, int]
) -> Tuple[np.ndarray, float]:
    """
    Checks the __get_simplex_points arguments and returns them prepared for work.

    Args:
        start (np.ndarray): A numpy.ndarray representing the starting point for simplex
        generation.
        stride (Union[float, int]): A float or int representing the stride.

    Raises:
        TypeError: Raised if argument start is not a numpy.ndarray.
        ValueError: Raised if argument start is a zero-length vector.
        TypeError: Raised if argument stride is not a float or int.

    Returns:
        Tuple[np.ndarray, float]: Cleaned arguments.
    """
    if not isinstance(start, np.ndarray):
        raise TypeError(
            "Expected argument start to be a numpy.ndarray, instead it is "
            f"{type(start)}."
        )

    start = np.reshape(start, -1)

    if start.shape[0] == 0:
        raise ValueError(
            "Expected argument starting point to be a vector with at least one "
            "element, instead it is empty."
        )

    if not isinstance(stride, (float, int)):
        raise TypeError(
            "Expected argument stride to be a float or int, instead it is "
            f"{type(stride)}."
        )

    stride = float(stride)

    return start, stride


def __get_simplex_points(start: np.ndarray, stride: float) -> np.ndarray:
    """
    Generates simplex points for a starting point.

    Args:
        start (np.ndarray): A numpy.ndarray representing the starting point for simplex
        generation.
        stride (float): A float representing the stride.

    Returns:
        np.ndarray: A matrix with each row representing a point of the simplex.
    """

    points = np.tile(start, reps=(start.shape[0], 1))
    points = points + stride * np.eye(points.shape[0])

    return np.vstack([start, points])


def __reflect(
    centroid: np.ndarray, maximum_point: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Reflects argument maximum_points wrt centroid by argument alpha.

    Args:
        centroid (np.ndarray): A numpy.ndarray representing the simplex centroid.
        maximum_point (np.ndarray): A numpy.ndarray representing the worst point of a
        simplex.
        alpha (float): A float representing the amount a point will be reflected.

    Returns:
        np.ndarray: A numpy.ndarray representing the reflected point.
    """
    return (1 + alpha) * centroid - alpha * maximum_point


def __contract(
    centroid: np.ndarray, maximum_point: np.ndarray, beta: float
) -> np.ndarray:
    """
    Contracts argument maximum_points wrt centroid by argument beta.

    Args:
        centroid (np.ndarray): A numpy.ndarray representing the simplex centroid.
        maximum_point (np.ndarray): A numpy.ndarray representing the worst point of a
        simplex.
        beta (float): A float representing the amount a point will be contracted.

    Returns:
        np.ndarray: A numpy.ndarray representing the contracted point.
    """
    return (1 - beta) * centroid + beta * maximum_point


def __expand(
    centroid: np.ndarray, reflected_point: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Expands argument reflected_point wrt centroid by argument alpha.

    Args:
        centroid (np.ndarray): A numpy.ndarray representing the simplex centroid.
        maximum_point (np.ndarray): A numpy.ndarray representing the worst point of a
        simplex.
        gamma (float): A float representing the amount a point will be expanded.

    Returns:
        np.ndarray: A numpy.ndarray representing the expanded point.
    """
    return (1 - gamma) * centroid + gamma * reflected_point


def __expand_jakobovic(
    centroid: np.ndarray, reflected_point: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Expands argument reflected_point wrt centroid by argument alpha. This is a modified
    version which is supposedly the correct one, as said by prof. Jakobović.

    Args:
        centroid (np.ndarray): A numpy.ndarray representing the simplex centroid.
        maximum_point (np.ndarray): A numpy.ndarray representing the worst point of a
        simplex.
        gamma (float): A float representing the amount a point will be expanded.

    Returns:
        np.ndarray: A numpy.ndarray representing the expanded point.
    """
    return (1 - gamma) * centroid - gamma * reflected_point


def __time_to_stop(
    simplex_values: np.ndarray, centroid_value: float, epsilon: float
) -> bool:
    """
    Checks if it's time to stop Nelder Mead Simplex Search.

    Args:
        simplex_values (np.ndarray): A numpy.ndarray representing the vector of simplex
        values.
        centroid_value (float): A float representing the value of the simplex centroid.
        epsilon (float): A float representing the error threshold.

    Returns:
        bool: True if the stopping condition of Nelder Mead Simplex Search has been met,
        False otherwise.
    """
    difference_in_values = simplex_values - centroid_value
    squared_difference_in_values = np.square(difference_in_values)
    mean_squared_difference_in_values = np.mean(squared_difference_in_values)

    return np.sqrt(mean_squared_difference_in_values) <= epsilon


def __print_nmss_values(
    function: Function,
    centroid: np.ndarray,
    verbosity: int,
    decimal_precision: int,
):
    """
    Prints the Nelder Mead Simplex Search values.

    Args:
        function (Function): A Function representing the loss function.
        centroid (np.ndarray): A numpy.ndarray representing the simplex centroid.
        verbosity (int): An int representing the level of verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.
    """
    if verbosity == 1:
        print(f"c = {np.around(centroid, decimal_precision)}")
    elif verbosity > 1:
        result = function(centroid, dont_count=True)
        result = (
            np.around(result, 3)
            if isinstance(result, np.ndarray)
            else f"{result:.0{decimal_precision}f}"
        )

        print(f"F(c = {np.around(centroid, decimal_precision)}) = {result}")


def nelder_mead_simplex_search(
    function: Function,
    start: np.ndarray,
    stride: Union[float, int] = 1,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    sigma: float = 0.5,
    use_jakobovic_expand: bool = False,
    epsilon: float = 1e-6,
    max_iterations: int = 100000,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
) -> np.ndarray:
    """
    Uses Nelder Mead Simplex Search to find an n-D optimum of a function.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        search.
        stride (Union[float, int], optional): A float or int representing the stride for
        simplex generation. Defaults to 1.
        alpha (float, optional): A float used in point reflection. Defaults to 1.0.
        beta (float, optional): A float used in point contraction. Defaults to 0.5.
        gamma (float, optional): A float used in point expansion. Defaults to 2.0.
        sigma (float, optional): A float used when moving points to the optimum.
        Defaults to 0.5.
        use_jakobovic_expand (float, optional): A bool determining whether or not to use
        the __expand_jakobovic method instead of the __expand method for point
        expansion. Defaults to False.
        epsilon (float, optional): A float representing the error threshold. Defaults to
        1e-6.
        max_iterations (int, optional): An int representing the maximum number of
        iterations before the algorithm times out and returns the last found optimum.
        Defaults to 100000.
        verbosity (Optional[str], optional): A str representing the verbosity of the
        output during algorithm execution. Defaults to None (no output during algorithm
        execution).
        decimal_precision (int, optional): An int representing the number of decimal
        digits to round numbers outputted during algorithm execution. Defaults to 3.

    Returns:
        np.ndarray: A numpy.ndarray representing the last found optimum.
    """
    (
        function,
        alpha,
        beta,
        gamma,
        sigma,
        use_jakobovic_expand,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    ) = clean_nelder_mead_simplex_search_arguments(
        function=function,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sigma=sigma,
        use_jakobovic_expand=use_jakobovic_expand,
        epsilon=epsilon,
        max_iterations=max_iterations,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )
    start, stride = clean_get_simplex_points(start=start, stride=stride)

    simplex_points = __get_simplex_points(start=start, stride=stride)
    simplex_values = np.array([function(x) for x in simplex_points])

    expansion_method = __expand_jakobovic if use_jakobovic_expand else __expand

    for _ in range(max_iterations):
        minimum_index = np.argmin(simplex_values)
        maximum_index = np.argmax(simplex_values)
        centroid = np.mean(np.delete(simplex_points, maximum_index, axis=0), axis=0)

        __print_nmss_values(
            function=function,
            centroid=centroid,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

        reflected_point = __reflect(
            centroid=centroid, maximum_point=simplex_points[maximum_index], alpha=alpha
        )

        reflected_value = function(reflected_point)
        minimum_value = simplex_values[minimum_index]

        if reflected_value < minimum_value:
            expanded_point = expansion_method(
                centroid=centroid, reflected_point=reflected_point, gamma=gamma
            )
            expanded_value = function(expanded_point)

            if expanded_value < minimum_value:
                simplex_points[maximum_index] = expanded_point
                simplex_values[maximum_index] = expanded_value
            else:
                simplex_points[maximum_index] = reflected_point
                simplex_values[maximum_index] = reflected_value
        else:
            maximum_value = simplex_values[maximum_index]

            if all(np.delete(simplex_values, maximum_index, axis=0) < reflected_value):
                if reflected_value < maximum_value:
                    simplex_points[maximum_index] = reflected_point
                    simplex_values[maximum_index] = reflected_value

                    # We need this here since we're introducing a new point and value
                    minimum_index = np.argmin(simplex_values)
                    maximum_index = np.argmax(simplex_values)

                    # We need to do this since the maximum value has potentially changed
                    maximum_value = simplex_values[maximum_index]

                contracted_point = __contract(
                    centroid=centroid,
                    maximum_point=simplex_points[maximum_index],
                    beta=beta,
                )
                contracted_value = function(contracted_point)

                if contracted_value < maximum_value:
                    simplex_points[maximum_index] = contracted_point
                    simplex_values[maximum_index] = contracted_value
                else:
                    for i, simplex_point in enumerate(simplex_points):
                        if i == minimum_index:
                            continue

                        simplex_points[i] += (
                            simplex_points[minimum_index] - simplex_points[i]
                        ) * sigma
                        simplex_values[i] = function(simplex_points[i])
            else:
                simplex_points[maximum_index] = reflected_point
                simplex_values[maximum_index] = reflected_value

        if __time_to_stop(
            simplex_values=simplex_values,
            centroid_value=function(centroid),
            epsilon=epsilon,
        ):
            break

    # Do this to get a more precise result
    maximum_index = np.argmax(simplex_values)
    centroid = np.mean(np.delete(simplex_points, maximum_index, axis=0), axis=0)

    return centroid

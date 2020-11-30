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

import copy
import sys
from typing import List, Optional, Tuple

import numpy as np

from . import constants
from .function import Function


def clean_hooke_jeeves_search_arguments(
    function: Function,
    start: np.ndarray,
    stride: Optional[np.ndarray],
    epsilon: Optional[np.ndarray],
    max_iterations: int,
    verbosity: Optional[str],
    decimal_precision: int,
) -> Tuple[Function, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Checks the Hooke-Jeeves Search arguments and returns them prepared for work.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        search.
        stride (Optional[np.ndarray]): A numpy.ndarray representing the movement vector.
        epsilon (Optional[np.ndarray]): A numpy.ndarray representing the error
        threshold vector.
        max_iterations (int): An int representing the maximum number of iterations
        before the algorithm times out and returns the last found optimum.
        verbosity (Optional[str]): A str representing the verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.

    Raises:
        TypeError: Raised if argument function is not a Function.
        TypeError: Raised if argument start is not a numpy.ndarray.
        TypeError: Raised if argument stride is not a numpy.ndarray.
        TypeError: Raised if argument epsilon is not a numpy.ndarray.
        ValueError: Raised if argument epsilon contains at least one negative element.
        TypeError: Raised if argument max_iterations is not an int.
        ValueError: Raised if argument max_iterations is a negative number.
        TypeError: Raised if argument verbosity is not a str.
        KeyError: Raised if argument verbosity is an invalid key.
        TypeError: Raised if argument decimal_precision is not an int.
        ValueError: Raised if argument decimal_precision is a negative number.

    Returns:
        Tuple[Function, np.ndarray, np.ndarray, np.ndarray, int, int, int]: Cleaned
        arguments.
    """
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    if not isinstance(start, np.ndarray):
        raise TypeError(
            "Expected argument start to be a numpy.ndarray, instead it is "
            f"{type(start)}."
        )

    start = np.reshape(start, -1)

    if stride is None:
        stride = np.array([0.5] * len(start))

    if not isinstance(stride, np.ndarray):
        raise TypeError(
            "Expected argument stride to be a numpy.ndarray, instead it is "
            f"{type(stride)}."
        )

    if epsilon is None:
        epsilon = np.array([1e-6] * len(start))

    if not isinstance(epsilon, np.ndarray):
        raise TypeError(
            "Expected argument epsilon to be a numpy.ndarray, instead it is "
            f"{type(epsilon)}."
        )

    if any(epsilon < 0):
        raise ValueError(
            "Expected argument epsilon to be a vector of positive floats, instead it "
            f"is {epsilon}."
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

    if verbosity not in constants.HOOKE_JEEVES_VERBOSITY_DICT:
        verbosity_dict_length = len(constants.HOOKE_JEEVES_VERBOSITY_DICT)

        if verbosity_dict_length == 0:
            verbosity_string = "There are no keys available."
        elif verbosity_dict_length == 1:
            _key = list(constants.HOOKE_JEEVES_VERBOSITY_DICT.keys())[0]
            verbosity_string = f"The only available key is {_key}."
        else:
            _keys = list(sorted(constants.HOOKE_JEEVES_VERBOSITY_DICT.keys()))
            verbosity_string = "The available keys are "
            verbosity_string += ", ".join([str(x) for x in _keys[:-1]])
            verbosity_string += f" and {_keys[-1]}."

        raise KeyError(
            f'Verbosity key "{verbosity}" is not in the Hooke Jeeves Verbosity '
            f"dictionary. {verbosity_string}"
        )

    verbosity = constants.HOOKE_JEEVES_VERBOSITY_DICT[verbosity]

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
        start,
        stride,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    )


def get_survey_points(
    function: Function, start: np.ndarray, stride: np.ndarray
) -> List[np.ndarray]:
    """
    Returns the Hooke-Jeeves __survey points. DO NOT use this for the Hooke-Jeeves
    search as it doesn't count function calls!

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        survey.
        stride (np.ndarray): A numpy.ndarray representing the movement vector.

    Returns:
        List[np.ndarray]: A List[numpy.ndarray] representing the list of points the
        __survey method would explore.
    """
    # Copy this to avoid editing the original value
    point = copy.deepcopy(start)
    points = list()

    for i in range(start.shape[0]):
        # We use vectors as terms because directly editing
        # numpy.ndarray elements doesn't seem to work!
        stride_one_hot = np.zeros(stride.shape)
        stride_one_hot[i] = stride[i]

        point_value = function(point, dont_count=True)

        point = point + stride_one_hot
        point_value_plus = function(point, dont_count=True)
        points.append(copy.deepcopy(point))

        if point_value_plus > point_value:
            point = point - stride_one_hot * 2
            points.append(copy.deepcopy(point))

            if function(point, dont_count=True) > point_value:
                point = point + stride_one_hot

    return points


def __survey(function: Function, start: np.ndarray, stride: np.ndarray) -> np.ndarray:
    """
    Surveys the immediate neighbourhood of argument start and returns the local optimum
    based on argument function.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        survey.
        stride (np.ndarray): A numpy.ndarray representing the movement vector.

    Returns:
        np.ndarray: A numpy.ndarray representing the local optimum of the loss function.
    """
    # Copy this to avoid editing the original value
    point = copy.deepcopy(start)

    for i in range(start.shape[0]):
        # We use vectors as terms because directly editing
        # numpy.ndarray elements doesn't seem to work!
        stride_one_hot = np.zeros(stride.shape)
        stride_one_hot[i] = stride[i]

        point_value = function(point)

        point = point + stride_one_hot
        point_value_plus = function(point)

        if point_value_plus > point_value:
            point = point - stride_one_hot * 2

            if function(point) > point_value:
                point = point + stride_one_hot

    return point


def __time_to_stop(stride: np.ndarray, min_stride: np.ndarray) -> bool:
    """
    Checks if it's time to stop Hooke-Jeeves Search.

    Args:
        stride (np.ndarray): A numpy.ndarray representing the movement vector.
        min_stride (np.ndarray): A numpy.ndarray representing the minimum allowed
        movement vector. In practice equal to half the epsilon.

    Returns:
        bool: True if the stopping condition of Hooke-Jeeves Search has been met, False
        otherwise.
    """
    # Although not specified, I presume all is the correct
    # aggregator function here, otherwise one large epsilon
    # element breaks the whole thing.
    return all(stride <= min_stride)


def __print_hjs_values(
    function: Function,
    base_point: np.ndarray,
    search_start_point: np.ndarray,
    current_point: np.ndarray,
    verbosity: int,
    decimal_precision: int,
):
    """
    Prints the Hooke-Jeeves Search values.

    Args:
        function (Function): A Function representing the loss function.
        base_point (np.ndarray): A numpy.ndarray representing the base point of the
        Hooke-Jeeves Search (AKA x_b).
        search_start_point (np.ndarray): A numpy.ndarray representing the search start
        point of Hooke-Jeeves Search (AKA x_p).
        current_point (np.ndarray): A numpy.ndarray representing the current point of
        Hooke-Jeeves Search (AKA x_n).
        verbosity (int): An int representing the level of verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.
    """
    if verbosity < 1:
        return

    points = (base_point, search_start_point, current_point)
    point_identifiers = ("x_b", "x_p", "x_n")
    point_strings = [f"{np.around(x, decimal_precision)}" for x in points]

    if verbosity == 1:
        print(
            "\n".join(
                f"{identifier} = {point}"
                for identifier, point in zip(point_identifiers, point_strings)
            )
        )
    elif verbosity > 1:
        point_values = [function(point, dont_count=True) for point in points]

        print(
            "\n".join(
                f"F({identifier} = {point}) = {value}"
                for identifier, point, value in zip(
                    point_identifiers, point_strings, point_values
                )
            )
        )

    # Do this because multi-line outputs get cluttered easily.
    print()


def hooke_jeeves_search(
    function: Function,
    start: np.ndarray,
    stride: Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
    max_iterations: int = 100000,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
) -> np.ndarray:
    """
    Uses Hooke-Jeeves Search to find an n-D optimum of a function.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        search.
        stride (Optional[np.ndarray], optional): A numpy.ndarray representing the
        movement vector. Defaults to None (0.5 for all elements).
        epsilon (Optional[np.ndarray], optional): A numpy.ndarray representing the error
        threshold vector. Defaults to None (1e-6 for all elements).
        max_iterations (int, optional): An int representing the maximum number of
        iterations before the algorithm times out and returns the last found optimum.
        Defaults to 100000.
        verbosity (Optional[str], optional): A str representing the verbosity of the
        output during algorithm execution. Defaults to None (no output).
        decimal_precision (int, optional): An int representing the number of decimal
        digits to round numbers outputted during algorithm execution. Defaults to 3.

    Returns:
        np.ndarray: A numpy.ndarray representing the last found optimum.
    """
    (
        function,
        start,
        stride,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    ) = clean_hooke_jeeves_search_arguments(
        function=function,
        start=start,
        stride=stride,
        epsilon=epsilon,
        max_iterations=max_iterations,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    # Copy it since we'll change it during execution.
    stride = copy.deepcopy(stride)

    # Don't calculate them every time.
    min_stride = epsilon / 2

    # Copy them since we'll change them during execution.
    base_point = copy.deepcopy(start)
    search_start_point = copy.deepcopy(start)

    timed_out = True

    for _ in range(max_iterations):
        current_point = __survey(function, search_start_point, stride)

        __print_hjs_values(
            function=function,
            base_point=base_point,
            search_start_point=search_start_point,
            current_point=current_point,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

        # No need to cache this since we're not using the values after.
        if function(current_point) < function(base_point):
            search_start_point = current_point * 2 - base_point
            base_point = copy.deepcopy(current_point)
        else:
            stride /= 2
            search_start_point = copy.deepcopy(base_point)

        if __time_to_stop(stride, min_stride):
            timed_out = False
            break

    if timed_out:
        print(
            f"WARNING: Hooke-Jeeves Search timed out after {max_iterations} iterations "
            "- result might not be a minimum.",
            file=sys.stderr,
        )

    return base_point

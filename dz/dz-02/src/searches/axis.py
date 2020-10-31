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
from typing import Optional, Tuple

import numpy as np

from . import constants
from .function import Function
from .golden_section import golden_section_search


def clean_axis_search_arguments(
    function: Function,
    start: np.ndarray,
    epsilon: Optional[np.ndarray],
    max_iterations: int,
) -> Tuple[Function, np.ndarray, np.ndarray, int]:
    """
    Checks the Axis Search arguments and returns them prepared for work.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        search.
        epsilon (Optional[np.ndarray]): A numpy.ndarray representing the element-wise
        error tolerance.
        max_iterations (int): An int representing the maximum number of iterations
        before the algorithm times out and returns the last found optimum.

    Raises:
        TypeError: Raised if argument function is not a Function.
        TypeError: Raised if argument start is not a numpy.ndarray.
        TypeError: Raised if argument epsilon is not a numpy.ndarray.
        ValueError: Raised if argument epsilon contains at least one negative element.
        TypeError: Raised if argument max_iterations is not an int.
        ValueError: Raised if argument max_iterations is a negative number.

    Returns:
        Tuple[Function, np.ndarray, np.ndarray, int]: Cleaned arguments.
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

    return function, start, epsilon, max_iterations


def axis_search(
    function: Function,
    start: np.ndarray,
    epsilon: Optional[np.ndarray] = None,
    max_iterations: int = 100000,
    verbosity: Optional[str] = None,
    k_constant: float = constants.GOLDEN_SECTION_K_CONSTANT,
    decimal_precision: int = 3,
) -> np.ndarray:
    """
    Uses Axis Search to find a n-D optimum of a function with the Golden Section method
    as a backbone.

    Args:
        function (Function): A Function representing the loss function.
        start (np.ndarray): A numpy.ndarray representing the starting point of the
        search.
        epsilon (Optional[np.ndarray], optional): A numpy.ndarray representing the
        element-wise error tolerance. Defaults to None (1e-6 for all elements).
        max_iterations (int, optional): An int representing the maximum number of
        iterations before the algorithm times out and returns the last found optimum.
        Defaults to 100000.
        verbosity (Optional[str], optional): A str representing the verbosity of the
        output during algorithm execution. Defaults to None (no output).
        k_constant (float, optional): A float representing the constant for the Golden
        Section search. Defaults to constants.GOLDEN_SECTION_K_CONSTANT.
        decimal_precision (int, optional): An int representing the number of decimal
        digits to round numbers outputted during algorithm execution. Defaults to 3.

    Returns:
        np.ndarray: A numpy.ndarray representing the last found optimum.
    """
    function, start, epsilon, max_iterations = clean_axis_search_arguments(
        function=function, start=start, epsilon=epsilon, max_iterations=max_iterations
    )

    last_point = copy.deepcopy(start)
    timed_out = True

    # Base movement matrix is an eye matrix with an eigenvector of
    # epsilon.
    base_movement_matrix = np.zeros((epsilon.shape[0], epsilon.shape[0]))

    for i in range(epsilon.shape[0]):
        base_movement_matrix[i][i] = epsilon[i]

    for _ in range(max_iterations):
        current_point = copy.deepcopy(last_point)

        for i, x in enumerate(current_point):
            movement_vector = copy.deepcopy(base_movement_matrix[i])

            artificial_function = Function(
                lambda x: function(current_point + x * movement_vector)
            )

            x_min = golden_section_search(
                function=artificial_function,
                start=0.0,
                epsilon=epsilon[i],
                verbosity=verbosity,
                k_constant=k_constant,
                decimal_precision=decimal_precision,
            )

            current_point[i] += x_min * movement_vector[i]

        diff_smaller_than_epsilon = np.abs(current_point - last_point) < (epsilon)
        last_point = copy.deepcopy(current_point)

        if all(diff_smaller_than_epsilon):
            timed_out = False
            break

    if timed_out:
        print(
            f"WARNING: Axis Search timed out after {max_iterations} iterations - "
            "result might not be a minimum.",
            file=sys.stderr,
        )

    return last_point

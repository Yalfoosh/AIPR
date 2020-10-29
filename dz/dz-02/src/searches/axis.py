import copy
import sys
from typing import Optional, Tuple

import numpy as np

from .function import Function
from .golden_section import golden_section_search


def clean_axis_search_arguments(
    function: Function,
    start: np.ndarray,
    epsilon: Optional[np.ndarray],
    max_iterations: int,
) -> Tuple[Function, np.ndarray, np.ndarray, int]:
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
        raise TypeError(
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
    k_constant: float = None,
    decimal_precision: int = 3,
) -> np.ndarray:
    function, start, epsilon, max_iterations = clean_axis_search_arguments(
        function=function, start=start, epsilon=epsilon, max_iterations=max_iterations
    )

    last_point = copy.deepcopy(start)
    timed_out = True

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

        diff_smaller_than_epsilon = np.abs(current_point - last_point).less(epsilon)
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

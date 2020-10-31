import sys
from typing import Optional, Tuple

import numpy as np

from . import constants
from .function import Function


def clean_nelder_mead_simplex_search_arguments(
    function: Function,
    start: np.ndarray,
    stride: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma: float,
    epsilon: float,
    max_iterations: int,
    verbosity: Optional[str],
    decimal_precision: int,
) -> Tuple[Function, np.ndarray, float, int, float, float, float, float, int, int, int]:
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

    if isinstance(stride, int):
        stride = float(stride)

    if not isinstance(stride, float):
        raise TypeError(
            "Expected argument stride to be a float, instead it is " f"{type(stride)}."
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
        raise TypeError(
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
        start,
        stride,
        alpha,
        beta,
        gamma,
        sigma,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    )


def clean_get_simplex_points(
    start: np.ndarray, stride: float
) -> Tuple[np.ndarray, float]:
    if not isinstance(start, np.ndarray):
        raise TypeError(
            "Expected argument start to be a numpy.ndarray, instead it is "
            f"{type(start)}."
        )

    start = np.reshape(start, -1)

    if isinstance(stride, int):
        stride = float(stride)

    if not isinstance(stride, float):
        raise TypeError(
            "Expected argument stride to be a float, instead it is " f"{type(stride)}."
        )

    return start, stride


def __get_simplex_points(
    start: np.ndarray, stride: float, clean_arguments: bool = True
) -> np.ndarray:
    if clean_arguments:
        start, stride = clean_get_simplex_points(start=start, stride=stride)

    if start.shape[0] == 0:
        raise ValueError(
            "Expected argument starting point to be a vector with at least one "
            "element, instead it is empty."
        )

    points = np.tile(start, reps=(start.shape[0], 1))
    points = points + stride * np.eye(points.shape[0])

    return np.vstack([start, points])


def __reflect(
    centroid: np.ndarray, maximum_point: np.ndarray, alpha: float
) -> np.ndarray:
    return (1 + alpha) * centroid - alpha * maximum_point


def __contract(
    centroid: np.ndarray, maximum_point: np.ndarray, beta: float
) -> np.ndarray:
    return (1 - beta) * centroid + beta * maximum_point


def __expand(
    centroid: np.ndarray, reflected_point: np.ndarray, gamma: float
) -> np.ndarray:
    return (1 - gamma) * centroid + gamma * reflected_point


def __time_to_stop(
    simplex_values: np.ndarray, centroid_value: float, epsilon: float
) -> bool:
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
    if verbosity == 0:
        return
    elif verbosity == 1:
        print(f"c = {np.around(centroid, decimal_precision)}")
    elif verbosity > 1:
        result = function(centroid, dont_count=True)

        if isinstance(result, np.ndarray):
            result = np.around(result, 3)
        else:
            result = f"{result:.0{decimal_precision}f}"

        print(f"F(c = {np.around(centroid, decimal_precision)}) = {result}")


def nelder_mead_simplex_search(
    function: Function,
    start: np.ndarray,
    stride: int = 1,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    sigma: float = 0.5,
    epsilon: float = 1e-6,
    max_iterations: int = 100000,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
):
    (
        function,
        start,
        stride,
        alpha,
        beta,
        gamma,
        sigma,
        epsilon,
        max_iterations,
        verbosity,
        decimal_precision,
    ) = clean_nelder_mead_simplex_search_arguments(
        function=function,
        start=start,
        stride=stride,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sigma=sigma,
        epsilon=epsilon,
        max_iterations=max_iterations,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    simplex_points = __get_simplex_points(
        start=start, stride=stride, clean_arguments=False
    )
    simplex_values = np.array([function(x) for x in simplex_points])

    timed_out = True

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
            expanded_point = __expand(
                centroid=centroid, reflected_point=reflected_point, gamma=gamma
            )

            simplex_points[maximum_index] = (
                expanded_point
                if function(expanded_point) < minimum_value
                else reflected_point
            )
            simplex_values[maximum_index] = function(simplex_points[maximum_index])
        else:
            maximum_value = simplex_values[maximum_index]

            if all(np.delete(simplex_values, maximum_index, axis=0) < reflected_value):
                if reflected_value < maximum_value:
                    simplex_points[maximum_index] = reflected_point
                    simplex_values[maximum_index] = reflected_value

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
            timed_out = False
            break

    if timed_out:
        print(
            f"WARNING: Nelder Mead Simplex Search timed out after {max_iterations} "
            "iterations - result might not be a minimum.",
            file=sys.stderr,
        )

    maximum_index = np.argmax(simplex_values)
    centroid = np.mean(np.delete(simplex_points, maximum_index, axis=0), axis=0)

    return centroid

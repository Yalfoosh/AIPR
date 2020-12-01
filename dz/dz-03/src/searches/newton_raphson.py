import copy
import sys
from typing import Optional

import numpy as np

from . import constants
from .function import Function
from .utils import find_optimal_gradient, normalize


STRIDE_SCALING_FUNCTION_DICT = {
    "none": lambda x, **kwargs: x,
    "normalize": lambda x, **kwargs: normalize(x, epsilon=kwargs.get("epsilon", 1e-6)),
    "find optimal": lambda x, **kwargs: find_optimal_gradient(
        function=kwargs.get("function"),
        current_point=kwargs.get("current_point"),
        gradient=x,
    ),
}


def clean_newton_raphson_arguments(
    function: Function,
    start: np.ndarray,
    stride_scaling: Optional[str],
    epsilon: Optional[float],
    max_iterations_without_improvement: int,
    verbosity: Optional[str],
    decimal_precision: int,
):
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    if not isinstance(function.derivative, Function):
        raise ValueError(
            "Expected argument function to have a derivative, instead it is "
            f"{function.derivative}."
        )

    if not isinstance(start, np.ndarray):
        raise TypeError(
            "Expected argument start to be a numpy.ndarray, instead it is "
            f"{type(start)}."
        )

    start = np.reshape(start, -1)

    if not isinstance(stride_scaling, str):
        raise TypeError(
            "Expected argument stride_scaling to be a str, instead it is "
            f"{type(stride_scaling)}."
        )

    if stride_scaling not in STRIDE_SCALING_FUNCTION_DICT:
        stride_scaling_dict_length = len(STRIDE_SCALING_FUNCTION_DICT)

        if stride_scaling_dict_length == 0:
            stride_scaling_string = "There are no keys available."
        elif stride_scaling_dict_length == 1:
            _key = list(STRIDE_SCALING_FUNCTION_DICT.keys())[0]
            stride_scaling_string = f"The only available key is {_key}."
        else:
            _keys = list(sorted(STRIDE_SCALING_FUNCTION_DICT.keys()))
            stride_scaling_string = "The available keys are "
            stride_scaling_string += ", ".join([str(x) for x in _keys[:-1]])
            stride_scaling_string += f" and {_keys[-1]}."

        raise KeyError(
            f'Stride scaling key "{stride_scaling}" is not in the Stride Scaling '
            f"Function dictionary. {stride_scaling_string}"
        )

    stride_scaling = STRIDE_SCALING_FUNCTION_DICT[stride_scaling]

    if not isinstance(epsilon, float):
        raise TypeError(
            f"Expected argument epsilon to be a float, instead it is {type(epsilon)}."
        )

    if not isinstance(max_iterations_without_improvement, int):
        raise TypeError(
            "Expected argument max_iterations_without_improvement to be an int, "
            f"instead it is {type(max_iterations_without_improvement)}."
        )

    if max_iterations_without_improvement < 1:
        raise ValueError(
            "Expected argument max_iterations_without_improvement to be a positive "
            f"integer, instead it is {max_iterations_without_improvement}."
        )

    if verbosity is None:
        verbosity = "none"

    if not isinstance(verbosity, str):
        raise TypeError(
            f"Expected argument verbosity to be a str, instead it is {type(verbosity)}."
        )

    if verbosity not in constants.NEWTON_RAPHSON_VERBOSITY_DICT:
        verbosity_dict_length = len(constants.NEWTON_RAPHSON_VERBOSITY_DICT)

        if verbosity_dict_length == 0:
            verbosity_string = "There are no keys available."
        elif verbosity_dict_length == 1:
            _key = list(constants.NEWTON_RAPHSON_VERBOSITY_DICT.keys())[0]
            verbosity_string = f"The only available key is {_key}."
        else:
            _keys = list(sorted(constants.NEWTON_RAPHSON_VERBOSITY_DICT.keys()))
            verbosity_string = "The available keys are "
            verbosity_string += ", ".join([str(x) for x in _keys[:-1]])
            verbosity_string += f" and {_keys[-1]}."

        raise KeyError(
            f'Verbosity key "{verbosity}" is not in the Newton-Raphson Verbosity '
            f"dictionary. {verbosity_string}"
        )

    verbosity = constants.NEWTON_RAPHSON_VERBOSITY_DICT[verbosity]

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
        stride_scaling,
        epsilon,
        max_iterations_without_improvement,
        verbosity,
        decimal_precision,
    )


def __print_nrs_values(
    function: Function,
    point: np.ndarray,
    verbosity: int,
    decimal_precision: int,
):
    if verbosity < 1:
        return

    point_string = np.around(point, decimal_precision)

    if verbosity == 1:
        print(f"x' = {point_string}")
    elif verbosity > 1:
        point_value = function(point, dont_count=True)

        print(f"F(x' = {point_string}) = {point_value}")


def newton_raphson_search(
    function: Function,
    start: np.ndarray,
    stride_scaling: Optional[str] = None,
    epsilon: float = 1e-6,
    max_iterations_without_improvement: int = 100,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
):
    (
        function,
        start,
        stride_function,
        epsilon,
        max_iterations_without_improvement,
        verbosity,
        decimal_precision,
    ) = clean_newton_raphson_arguments(
        function=function,
        start=start,
        stride_scaling=stride_scaling,
        epsilon=epsilon,
        max_iterations_without_improvement=max_iterations_without_improvement,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    current_point = copy.deepcopy(start)
    current_value = function(current_point)
    best_value = copy.deepcopy(current_value)
    timed_out = True

    iterations_without_improvement = 0

    while iterations_without_improvement < max_iterations_without_improvement:
        gradient_in_current_point = function.derivative(current_point)
        hesse_in_current_point = function.derivative.derivative(current_point)

        hesse_inverse = np.linalg.inv(hesse_in_current_point)

        # The reason we're not doing H^-1 @ grad is because it
        # would give us a column vector, and we're working with
        # rows by default.
        stride = gradient_in_current_point @ hesse_inverse

        if np.linalg.norm(stride) < epsilon:
            timed_out = False
            break

        resolved_stride = stride_function(
            stride,
            function=function,
            current_point=current_point,
        )

        current_point = current_point - resolved_stride

        __print_nrs_values(
            function=function,
            point=current_point,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

        new_value = function(current_point)

        if new_value < best_value:
            best_value = copy.deepcopy(new_value)
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    if timed_out:
        print(
            "WARNING: Newton-Raphson timed out after "
            f"{max_iterations_without_improvement} iterations passed with no "
            "improvement - result might not be a minimum.",
            file=sys.stderr,
        )

    return current_point

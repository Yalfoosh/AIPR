import copy
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

from . import constants
from .constraint import Constraint
from .exceptions import ConstraintsUnsatisfiable
from .function import Function


def clean_box_search_arguments(
    function: Function,
    alpha: float,
    constraints: Optional[Union[List[Constraint], Tuple[Constraint]]],
    epsilon: float,
    max_iterations_without_improvement: int,
    verbosity: Optional[str],
    decimal_precision: int,
):
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

    if constraints is None:
        constraints = list()

    if not isinstance(constraints, (list, tuple)):
        raise TypeError(
            "Expected argument constraints to be a list or a tuple, instead it is "
            f"{type(constraints)}."
        )

    if not all([isinstance(constraint, Constraint) for constraint in constraints]):
        raise TypeError(
            "All elements of argument constraints must be of type Constraint!"
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

    if verbosity not in constants.BOX_VERBOSITY_DICT:
        verbosity_dict_length = len(constants.BOX_VERBOSITY_DICT)

        if verbosity_dict_length == 0:
            verbosity_string = "There are no keys available."
        elif verbosity_dict_length == 1:
            _key = list(constants.BOX_VERBOSITY_DICT.keys())[0]
            verbosity_string = f'The only available key is "{_key}".'
        else:
            _keys = list(sorted(constants.BOX_VERBOSITY_DICT.keys()))
            verbosity_string = "The available keys are "
            verbosity_string += ", ".join([str(f'"{x}"') for x in _keys[:-1]])
            verbosity_string += f' and "{_keys[-1]}"".'

        raise KeyError(
            f'Verbosity key "{verbosity}" is not in the bOX Verbosity dictionary. '
            f"{verbosity_string}"
        )

    verbosity = constants.BOX_VERBOSITY_DICT[verbosity]

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
        constraints,
        epsilon,
        max_iterations_without_improvement,
        verbosity,
        decimal_precision,
    )


def clean_generate_points(
    start: np.ndarray,
    value_range: Tuple[int, int],
    n: Optional[int] = None,
):
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

    if not isinstance(value_range, tuple):
        value_range = tuple(value_range)

    if len(value_range) < 2:
        raise ValueError(
            "Excpected argument value_range to have at least 2 elements, instead it "
            f"has {len(value_range)}"
        )

    for i in range(2):
        if not isinstance(value_range[i], (float, int)):
            raise ValueError(
                "Expected argument value_range to consist of only floats or ints, but "
                f"found a {type(value_range[i])}."
            )

    if value_range[0] > value_range[1]:
        value_range[0], value_range[1] = value_range[1], value_range[0]

    if n is None:
        n = 2 * len(start)

    if not isinstance(n, int):
        raise TypeError(f"Expected argument n to be an int, instead it is {type(n)}")

    if n < 1:
        raise TypeError(f"Expected argument n to be a positive int, instead it is {n}.")

    return start, value_range, n


def _check_constraints(
    point: np.ndarray, constraints: Union[List[Constraint], Tuple[Constraint]]
):
    return all([constraint.is_satisfied(point) for constraint in constraints])


def _generate_points(
    start: np.ndarray,
    value_range: Tuple[float, float],
    constraints: Union[List[Constraint], Tuple[Constraint]],
    n: int = None,
):
    if not _check_constraints(start, constraints):
        raise ConstraintsUnsatisfiable(
            "Argument start, which is the starting point, doesn't satisfy the "
            "constraints given."
        )

    centroid = copy.deepcopy(start)
    points = np.random.uniform(*value_range[:2], size=(n, len(start)))

    adjustments_failed = False

    for i in range(len(points)):
        adjustments_made = 0

        while not _check_constraints(point=points[i], constraints=constraints):
            points[i] = 0.5 * (points[i] + centroid)

            adjustments_made += 1

            if adjustments_made > 100:
                adjustments_failed = True
                break

        if adjustments_failed:
            break

        centroid = points[: i + 1].mean(axis=0)

    if adjustments_failed:
        raise RuntimeError()
        print(
            file=sys.stderr,
        )

    return points


def __print_bs_values(
    function: Function,
    centroid: np.ndarray,
    verbosity: int,
    decimal_precision: int,
):
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


def __time_to_stop(values: np.ndarray, centroid_value: float, epsilon: float) -> bool:
    difference_in_values = values - centroid_value
    squared_difference_in_values = np.square(difference_in_values)
    mean_squared_difference_in_values = np.mean(squared_difference_in_values)

    return np.sqrt(mean_squared_difference_in_values) <= epsilon


def box_search(
    function: Function,
    start: np.ndarray,
    value_range: Tuple[Union[float, int], Union[float, int]],
    constraints: Optional[Union[List[Constraint], Tuple[Constraint]]] = None,
    alpha: float = 1.3,
    epsilon: float = 1e-6,
    max_iterations_without_improvement: int = 100,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
) -> np.ndarray:
    (
        function,
        alpha,
        constraints,
        epsilon,
        max_iterations_without_improvement,
        verbosity,
        decimal_precision,
    ) = clean_box_search_arguments(
        function=function,
        alpha=alpha,
        constraints=constraints,
        epsilon=epsilon,
        max_iterations_without_improvement=max_iterations_without_improvement,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    start, value_range, n = clean_generate_points(start=start, value_range=value_range)

    try:
        points = _generate_points(
            start=start, value_range=value_range, constraints=constraints, n=n
        )
    except RuntimeError as re:
        print(str(re), file=sys.stderr)
        return start

    values = np.array([function(x) for x in points])
    best_value = function(np.mean(np.delete(points, np.argmax(values), axis=0), axis=0))
    timed_out = True
    failed_to_readjust = False

    iterations_without_improvement = 0

    while iterations_without_improvement < max_iterations_without_improvement:
        worst_index = np.argmax(values)
        worst_index_2 = np.argmax(np.delete(values, worst_index, axis=0))

        centroid = np.mean(np.delete(points, worst_index, axis=0), axis=0)
        new_value = function(centroid)

        __print_bs_values(
            function=function,
            centroid=centroid,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

        if new_value < best_value:
            best_value = new_value

            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        reflected_point = (1 + alpha) * centroid - alpha * points[worst_index]
        reflected_point = np.clip(reflected_point, *value_range[:2])

        reflection_adjustments = 0

        while not _check_constraints(point=reflected_point, constraints=constraints):
            reflected_point = 0.5 * (reflected_point + centroid)

            reflection_adjustments += 1

            if reflection_adjustments > max_iterations_without_improvement:
                failed_to_readjust = True
                timed_out = False
                break

        if failed_to_readjust:
            print(
                "WARNING: Readjusting a point to satisfly the constraints failed, "
                "so the algorithm ended prematurely - result might not be a minimum.",
                file=sys.stderr,
            )
            break

        reflected_value = function(reflected_point)

        if reflected_value > function(points[worst_index_2]):
            reflected_point = 0.5 * (reflected_point + centroid)
            reflected_value = function(reflected_point)

        points[worst_index] = copy.deepcopy(reflected_point)
        values[worst_index] = reflected_value

        if __time_to_stop(
            values=values,
            centroid_value=function(centroid),
            epsilon=epsilon,
        ):
            timed_out = False
            break

    if timed_out:
        print(
            "WARNING: Box Search timed out after "
            f"{max_iterations_without_improvement} iterations passed with no "
            "improvement - result might not be a minimum.",
            file=sys.stderr,
        )

    # Do this to get a more precise result
    maximum_index = np.argmax(values)
    centroid = np.mean(np.delete(points, maximum_index, axis=0), axis=0)

    return centroid

import copy
from typing import Any, Callable, List, Tuple, Union

from .function import Function
from .nelder_mead import nelder_mead_simplex_search


import numpy as np


class Constraint:
    def __init__(self, function_object: Callable[[Any], bool]):
        if not callable(function_object):
            raise TypeError("Argument function_object must be callable!")

        self._constraint_function = copy.deepcopy(function_object)

    def is_satisfied(self, x) -> bool:
        return self._constraint_function(x)


def constraints_to_function(
    eq_constraints: Union[List[Callable], Tuple[Callable]],
    geq_constraints: Union[List[Callable], Tuple[Callable]],
    epsilon: float = 1e-13,
):
    def _constraint_function(x: np.ndarray, r: Union[float, int]):

        eq_constraint_values = np.array(
            [eq_constraint(x) for eq_constraint in eq_constraints]
        )
        eq_constraint_values = np.clip(eq_constraint_values, 0, None)

        geq_constraint_values = np.array(
            [geq_constraint(x) for geq_constraint in geq_constraints]
        )
        geq_constraint_values = np.clip(geq_constraint_values, 0, None)

        eq_part = np.sum(np.square(eq_constraint_values)) / r
        geq_part = -r * np.sum(np.log(geq_constraint_values))

        return eq_part + geq_part

    return _constraint_function


def find_inner_point(
    start: np.ndarray, geq_constraints: Union[List[Callable], Tuple[Callable]]
):
    def _f(x: np.ndarray):
        geq_values = np.array([geq_constraint(x) for geq_constraint in geq_constraints])
        geq_multipliers = np.zeros(shape=geq_values.shape)
        geq_multipliers[geq_values < 0] = 1.0
        geq_multipliers += -geq_values

        return -np.sum(geq_multipliers * geq_values)

    function = Function(_f)
    optimal_point = nelder_mead_simplex_search(
        function=function, start=copy.deepcopy(start)
    )

    return optimal_point

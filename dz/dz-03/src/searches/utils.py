from typing import Callable, Optional

import numpy as np

from . import constants
from .function import Function
from .golden_section import golden_section_search


def normalize(x: np.ndarray, epsilon: float = 1e-6):
    norm = np.linalg.norm(x)

    if norm < epsilon:
        norm = epsilon

    return x / norm


def find_optimal_gradient(
    function: Function,
    current_point: np.ndarray,
    gradient: np.ndarray,
):
    # A new function; used to determine the x for which the new
    # value along the gradient gives the smallest value.
    new_function = Function(lambda x: function(current_point - x * gradient))

    optimal_x = golden_section_search(function=new_function, start=0)

    return optimal_x * gradient

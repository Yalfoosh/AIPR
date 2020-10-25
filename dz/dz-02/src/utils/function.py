from typing import Callable

import numpy as np


def get_f3(n: int) -> Callable:
    if not isinstance(n, int):
        raise TypeError(f"Expected argument n to be an int, instead it is {type(n)}.")

    if n < 1:
        raise ValueError(
            f"Expected argument n to be a positive int, instead it is {type(n)}."
        )

    return lambda x: np.sum([np.square(x[i] - i) for i in range(n)])


def get_f3_start(n: int) -> np.array:
    if not isinstance(n, int):
        raise TypeError(f"Expected argument n to be an int, instead it is {type(n)}.")

    if n < 1:
        raise ValueError(
            f"Expected argument n to be a positive int, instead it is {type(n)}."
        )

    return np.array([[0] * n])

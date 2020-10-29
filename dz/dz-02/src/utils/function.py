from typing import Callable, Union

import numpy as np


def get_f3(starting_number: Union[float, int] = 1):
    if not isinstance(starting_number, (float, int)):
        raise TypeError(
            "Expected argument starting_number to be a float or int, instead it is "
            f"{type(starting_number)}."
        )

    return lambda x: np.sum(
        [np.square(x[i] - (starting_number + i)) for i in range(len(x))]
    )

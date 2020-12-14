from typing import Tuple, Union

import numpy as np


def get_binary_dim_for_interval_and_decimals(
    interval: Tuple[Union[float, int], Union[float, int]], n_decimals: int
):
    interval_length = interval[1] - interval[0]
    places_per_integer = 10 ** n_decimals

    max_number = interval_length * places_per_integer

    return int(np.ceil(np.log2(max_number)))

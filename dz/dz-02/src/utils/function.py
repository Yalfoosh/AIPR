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

from typing import Callable, Union

import numpy as np


def get_f3(starting_number: Union[float, int] = 1) -> Callable:
    """
    Returns function 3 for a given starting number (the first
    element of the function optimum).

    Args:
        starting_number (Union[float, int], optional): An int
        representing the first element of the function optimum.
        Defaults to 1.

    Raises:
        TypeError: Raised if argument starting_number is not a
        float or int.

    Returns:
        Callable: A callable object representing function 3.
    """
    if not isinstance(starting_number, (float, int)):
        raise TypeError(
            "Expected argument starting_number to be a float or int, instead it is "
            f"{type(starting_number)}."
        )

    return lambda x: np.sum(
        [np.square(x[i] - (starting_number + i)) for i in range(len(x))]
    )

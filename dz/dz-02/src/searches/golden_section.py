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

from typing import List, Optional, Tuple, Union

from . import constants
from .function import Function


def clean_golden_section_search_arguments(
    function: Function,
    start: Union[float, int],
    end: Optional[Union[float, int]],
    k_constant: float,
    epsilon: float,
    verbosity: Optional[str],
    decimal_precision: int,
) -> Tuple[Function, Tuple[float, float], float, float, int, int]:
    """
    Checks the Golden Section Search arguments and returns them prepared for work.

    Args:
        function (Function): A Function representing the loss function.
        start (Union[float, int]): AA float or int representing the starting point of
        the search.
        end (Optional[Union[float, int]]): A float or int representing the ending point
        of the search.
        k_constant (float): A float representing the constant for the Golden Section
        Search.
        epsilon (float): A float representing the error threshold.
        verbosity (Optional[str]): A str representing the verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.

    Raises:
        TypeError: Raised if argument function is not a Function.
        TypeError: Raised if argument start is not a float or int.
        TypeError: Raised if argument end is passed, but is not a float or int.
        TypeError: Raised if argument k_constant is not a float.
        TypeError: Raised if argument epsilon is not a float.
        ValueError: Raised if argument epsilon is a negative number.
        TypeError: Raised if argument verbosity is not a string.
        KeyError: Raised if argument verbosity is an invalid key.
        TypeError: Raised if argument decimal_precision is not an int.
        ValueError: Raised if argument decimal_precision is negative.

    Returns:
        Tuple[Function, Tuple[float, float], float, float, int, int]: Cleaned arguments.
    """
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    if not isinstance(start, (float, int)):
        raise TypeError(
            "Expected argument start to be a float or int, instead it is "
            f"{type(start)}"
        )

    interval = (float(start),)

    if end is not None:
        if not isinstance(end, (float, int)):
            raise TypeError(
                "Expected argument end to be a float or int, instead it is "
                f"{type(end)}"
            )

        interval = interval + (float(end),)

    if not isinstance(k_constant, float):
        raise TypeError(
            "Expected argument k_constant to be a float, instead it is "
            f"{type(k_constant)}."
        )

    if not isinstance(epsilon, float):
        raise TypeError(
            f"Expected argument epsilon to be a float, instead it is {type(epsilon)}."
        )

    if epsilon < 0:
        raise ValueError(
            "Expected argument epsilon to be a positive float, instead it is {epsilon}."
        )

    if verbosity is None:
        verbosity = "none"

    if not isinstance(verbosity, str):
        raise TypeError(
            f"Expected argument verbosity to be a str, instead it is {type(verbosity)}."
        )

    if verbosity not in constants.GOLDEN_SECTION_VERBOSITY_DICT:
        verbosity_dict_length = len(constants.GOLDEN_SECTION_VERBOSITY_DICT)

        if verbosity_dict_length == 0:
            verbosity_string = "There are no keys available."
        elif verbosity_dict_length == 1:
            _key = list(constants.GOLDEN_SECTION_VERBOSITY_DICT.keys())[0]
            verbosity_string = f'The only available key is "{_key}".'
        else:
            _keys = list(sorted(constants.GOLDEN_SECTION_VERBOSITY_DICT.keys()))
            verbosity_string = "The available keys are "
            verbosity_string += ", ".join([str(f'"{x}"') for x in _keys[:-1]])
            verbosity_string += f' and "{_keys[-1]}"".'

        raise KeyError(
            f'Verbosity key "{verbosity}" is not in the Golden Section Verbosity '
            f"dictionary. {verbosity_string}"
        )

    verbosity = constants.GOLDEN_SECTION_VERBOSITY_DICT[verbosity]

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

    return function, interval, k_constant, epsilon, verbosity, decimal_precision


def clean_find_unimodality_interval_arguments(
    function: Function,
    start: float,
    stride: int,
) -> Tuple[Function, float, int]:
    """
    Checks the Find unimodality interval arguments and returns them prepared for work.

    Args:
        function (Function): A Function representing the loss function.
        start (float): A float representing the starting point of the search.
        stride (int): An int representing the stride of the alrgorithm.

    Raises:
        TypeError: Raised if argument function is not a Function.
        TypeError: Raised if argument start is not a float.
        TypeError: Raised if argument stride is not an int.
        ValueError: Raised if argument string is a negative number.

    Returns:
        Tuple[Function, float, int]: Cleaned arguments.
    """
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    start = float(start)

    if not isinstance(start, float):
        raise TypeError(
            f"Expected argument start to be a float, instead it is {type(start)}."
        )

    if not isinstance(stride, int):
        raise TypeError(
            f"Expected argument stride to be an int, instead it is {type(stride)}."
        )

    if stride < 1:
        raise ValueError(
            f"Expected argument stride to be a positive int, instead it is {stride}."
        )

    return function, start, stride


def __print_gss_values(
    function: Function,
    a: float,
    b: float,
    c: float,
    d: float,
    verbosity: int,
    decimal_precision: int,
):
    """
    Prints the Golden Section Search values.

    Args:
        function (Function): A Function representing the loss function.
        a (float): A float representing the a variable of Golden Section Search.
        b (float): A float representing the b variable of Golden Section Search.
        c (float): A float representing the c variable of Golden Section Search.
        d (float): A float representing the d variable of Golden Section Search.
        verbosity (int): An int representing the level of verbosity of the output during
        algorithm execution.
        decimal_precision (int): An int representing the number of decimal digits to
        round numbers outputted during algorithm execution.
    """
    if verbosity == 0:
        return
    elif verbosity == 1:
        value_string = ", ".join([f"{x:.0{decimal_precision}f}" for x in (a, b, c, d)])
        print(f"(a, b, c, d) = {value_string}")
    elif verbosity > 1:
        # We don't count function calls only for c and d, as f(a)
        # and f(b) are never calculated.
        value_string = "  ".join(
            [
                f"f({x} = {y}) = "
                f"{function(y, dont_count=(x in ('c', 'd'))):.0{decimal_precision}f}"
                for x, y in zip(("a", "b", "c", "d"), (a, b, c, d))
            ]
        )


def find_unimodality_interval(
    function: Function,
    start: float,
    stride: int = 1,
) -> Tuple[float, float]:
    """
    Finds the nearest unimodality interval starting from some point.

    Args:
        function (Function): A Function representing the loss function.
        start (float): A float representing the starting point of the search.
        stride (int, optional): An int representing the stride of the alrgorithm.
        Defaults to 1.

    Returns:
        Tuple[float, float]: A pair of floats representing the bounds of the unimodality
        interval.
    """

    function, start, stride = clean_find_unimodality_interval_arguments(
        function=function, start=start, stride=stride
    )

    left, mid, right = start - stride, start, start + stride
    left_value, mid_value, right_value = (function(x) for x in (left, mid, right))

    step = 1

    if not left_value > mid_value < right_value:
        if mid_value > right_value:
            while mid_value > right_value:
                step *= 2
                left, mid, right = mid, right, start + stride * step
                mid_value, right_value = right_value, function(right)
        else:
            while mid_value > left_value:
                step *= 2
                left, mid, right = start - stride * step, left, mid
                left_value, mid_value = function(left), left_value

    return left, right


def golden_section_search(
    function: Function,
    start: Union[float, int],
    end: Optional[Union[float, int]] = None,
    k_constant: float = constants.GOLDEN_SECTION_K_CONSTANT,
    epsilon: float = 1e-6,
    verbosity: Optional[str] = None,
    decimal_precision: int = 3,
) -> float:
    """
    Uses Golden Section Search to find a 1D optimum of a function.

    Args:
        function (Function): A Function representing the loss function.
        start (Union[float, int]): A float or int representing the starting point of the
        search.
        end (Optional[Union[float, int]], optional): A float or int representing the
        right bound of the starting interval. Defaults to None (finds the unimodality
        interval based on argument start).
        k_constant (float, optional): A float representing the constant for the Golden
        Section Search. Defaults to constants.GOLDEN_SECTION_K_CONSTANT.
        epsilon (float, optional): A float representing the error threshold. Defaults to
        1e-6.
        verbosity (Optional[str], optional): A str representing the verbosity of the
        output during algorithm execution. Defaults to None (no output during algorithm
        execution).
        decimal_precision (int, optional): An int representing the number of decimal
        digits to round numbers outputted during algorithm execution. Defaults to 3.

    Returns:
        float: A float representing the last found optimum.
    """
    (
        function,
        interval,
        k_constant,
        epsilon,
        verbosity,
        decimal_precision,
    ) = clean_golden_section_search_arguments(
        function=function,
        start=start,
        end=end,
        k_constant=k_constant,
        epsilon=epsilon,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    if len(interval) == 1:
        interval = find_unimodality_interval(function=function, start=start)

    a, b = interval
    c, d = b - k_constant * (b - a), a + k_constant * (b - a)
    c_value, d_value = function(c), function(d)

    while (b - a) > epsilon:
        __print_gss_values(
            a=a,
            b=b,
            c=c,
            d=d,
            function=function,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

        if c_value < d_value:
            b, d = d, c
            c = b - k_constant * (b - a)

            c_value, d_value = function(c), c_value
        else:
            a, c = c, d
            d = a + k_constant * (b - a)

            c_value, d_value = d_value, function(d)

    return (a + b) / 2

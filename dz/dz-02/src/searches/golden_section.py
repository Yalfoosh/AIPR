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
    epsilon: float,
    verbosity: Optional[str],
    k_constant: float,
    decimal_precision: int,
) -> Tuple[Function, Tuple[float, float], float, int, float, int]:
    if not isinstance(function, Function):
        raise TypeError(
            "Expected argument function to be a Function, instead it is "
            f"{type(function)}."
        )

    if start is None:
        raise ValueError("Argument start mustn't be None!")
    else:
        interval = (float(start),)

        if end is not None:
            interval = interval + (float(end),)

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

    if not isinstance(k_constant, float):
        raise TypeError(
            "Expected argument k_constant to be a float, instead it is "
            f"{type(k_constant)}."
        )

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

    return function, interval, epsilon, verbosity, k_constant, decimal_precision


def clean_find_unimodality_interval_arguments(
    function: Function,
    start: float,
    stride: int,
) -> Tuple[Function, float, int]:
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
    if verbosity == 0:
        return
    elif verbosity == 1:
        value_string = ", ".join([f"{x:.0{decimal_precision}f}" for x in (a, b, c, d)])
        print(f"(a, b, c, d) = {value_string}")
    elif verbosity > 1:
        value_string = "  ".join(
            [
                f"f({x} = {y}) = {function(y, dont_count=True):.0{decimal_precision}f}"
                for x, y in zip(("a", "b", "c", "d"), (a, b, c, d))
            ]
        )


def find_unimodality_interval(
    function: Function,
    start: float,
    stride: int = 1,
) -> Tuple[float, float]:

    function, start, stride = clean_find_unimodality_interval_arguments(
        function=function, start=start, stride=stride
    )

    left, mid, right = start - stride, start, start + stride
    f_left, f_mid, f_right = (function(x) for x in (left, mid, right))

    step = 1

    if not f_left > f_mid < f_right:
        if f_mid > f_right:
            while f_mid > f_right:
                step *= 2
                left, mid, right = mid, right, start + stride * step
                f_mid, f_right = f_right, function(right)
        else:
            while f_mid > f_left:
                step *= 2
                left, mid, right = start - stride * step, left, mid
                f_left, f_mid = function(left), f_left

    return left, right


def golden_section_search(
    function: Function,
    start: Union[float, int],
    end: Optional[Union[float, int]] = None,
    epsilon: float = 1e-6,
    verbosity: Optional[str] = None,
    k_constant: float = constants.GOLDEN_SECTION_K_CONSTANT,
    decimal_precision: int = 3,
) -> float:
    (
        function,
        interval,
        epsilon,
        verbosity,
        k_constant,
        decimal_precision,
    ) = clean_golden_section_search_arguments(
        function=function,
        start=start,
        end=end,
        epsilon=epsilon,
        verbosity=verbosity,
        k_constant=k_constant,
        decimal_precision=decimal_precision,
    )

    if len(interval) == 1:
        interval = find_unimodality_interval(function=function, start=start)

    a, b = interval
    c, d = b - k_constant * (b - a), a + k_constant * (b - a)
    f_c, f_d = function(c), function(d)

    __print_gss_values(
        function=function,
        a=a,
        b=b,
        c=c,
        d=d,
        verbosity=verbosity,
        decimal_precision=decimal_precision,
    )

    while (b - a) > epsilon:
        if f_c < f_d:
            b, d = d, c
            c = b - k_constant * (b - a)

            f_c, f_d = function(c), f_c
        else:
            a, c = c, d
            d = a + k_constant * (b - a)

            f_c, f_d = f_d, function(d)

        __print_gss_values(
            a=a,
            b=b,
            c=c,
            d=d,
            function=function,
            verbosity=verbosity,
            decimal_precision=decimal_precision,
        )

    return (a + b) / 2

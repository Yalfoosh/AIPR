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

import copy
from typing import Any, Callable, Optional

import numpy as np


class Function:
    """
    A class wrapper for callable objects, which can count calls.
    """

    def __init__(self, function_object: Callable):
        """
        The Function constructor.

        Args:
            function_object (Callable): A callable object representing some function.

        Raises:
            ValueError: Raised if argument function_object isn't callable.
        """
        if not callable(function_object):
            raise ValueError("Argument function_object must be callable!")

        self._function_object = copy.deepcopy(function_object)
        self._call_count = 0

        self._derivative: Optional[Function] = None

    @property
    def function_object(self) -> Callable:
        """
        The function_object property.

        Returns:
            Callable: A deep copy of the internal function object.
        """
        return copy.deepcopy(self._function_object)

    @property
    def call_count(self) -> int:
        """
        The call_count property.

        Returns:
            int: An int representing the number of times the internal function object
            has been called (excluding the times it was called with dont_count set to
            True).
        """
        return self._call_count

    @property
    def n(self) -> int:
        """
        An alias for the call_count property.

        Returns:
            int: Whatever call_count returns.
        """
        return self.call_count

    @property
    def derivative(self) -> "Function":
        """
        The derivative property.

        Returns:
            Function: A Function representing the derivative of the function object.
        """
        return self._derivative

    @derivative.setter
    def derivative(self, value: "Function"):
        """
        A derivative setter. Setting a value successfully will get the deep copy and
        reset its counter.

        Args:
            value (Function): A Function representing the derivative you wish to assign
            to this function.
        """
        if isinstance(value, Function):
            self._derivative = value.get_new()

    def call(self, *args, **kwargs) -> Any:
        """
        Calls itself.

        Returns:
            Any: Whatever calling itself returns.
        """
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Calls the function_object within with args and kwargs, and if dont_count isn't
        set to true counts the function call.

        Returns:
            Any: Whatever the internal function object returns.
        """
        dont_count = kwargs.get("dont_count", None)

        if dont_count is None:
            dont_count = False
        else:
            # Do this so the actual function call isn't littered
            # by the dont_count kwarg.
            kwargs.pop("dont_count")

        call_result = self._function_object(*args, **kwargs)

        if not dont_count:
            self._call_count += 1

        return call_result

    def reset(self):
        """
        Resets the internal function object call counter to 0.
        """

        self._call_count = 0

        if self._derivative is not None:
            self._derivative.reset()

    def get_deepcopy(self) -> "Function":
        """
        Returns the deep copy of this class instance.

        Returns:
            Function: A Function representing the deep copy of
            this class instance.
        """
        return copy.deepcopy(self)

    def get_new(self) -> "Function":
        """
        Returns a deep copy of itself with the function object
        call counter reset to 0.

        Returns:
            Function: A Function representing the deep copy of
            this class instance with the function object call
            counter reset to 0.
        """
        to_return = self.get_deepcopy()
        to_return.reset()

        return to_return

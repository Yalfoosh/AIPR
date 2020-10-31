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
from typing import Callable, Optional


class Function:
    def __init__(self, function_object: Callable):
        if not callable(function_object):
            raise ValueError("Argument function_object must be callable!")

        self._function_object = copy.deepcopy(function_object)
        self._call_count = 0

    @property
    def function_object(self) -> Callable:
        return copy.deepcopy(self._function_object)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def n(self) -> int:
        return self.call_count

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        dont_count = kwargs.get("dont_count", None)

        if dont_count is None:
            dont_count = False
        else:
            kwargs.pop("dont_count")

        call_result = self._function_object(*args, **kwargs)

        if not dont_count:
            self._call_count += 1

        return call_result

    def reset(self, function_object: Optional[Callable] = None):
        if function_object is not None:
            if not callable(function_object):
                raise ValueError("Argument function_object must be callable!")

            self._function_object = function_object

        self._call_count = 0

    def get_deepcopy(self) -> "Function":
        return copy.deepcopy(self)

    def get_new(self) -> "Function":
        to_return = self.get_deepcopy()
        to_return.reset()

        return to_return

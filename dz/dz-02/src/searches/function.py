import copy
from typing import Callable, Optional


class Function:
    def __init__(self, function_object: Callable):
        if not callable(function_object):
            raise ValueError("Argument function_object must be callable!")

        self._function_object = copy.deepcopy(function_object)
        self._call_count = 0

    @property
    def call_count(self):
        return self._call_count

    @property
    def n(self):
        return self.call_count

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        call_result = self._function_object(*args, **kwargs)

        if not kwargs.get("dont_count", False):
            self._call_count += 1

        return call_result

    def reset(self, function_object: Optional[Callable] = None):
        if function_object is not None:
            if not callable(function_object):
                raise ValueError("Argument function_object must be callable!")

            self._function_object = function_object

        self._call_count = 0

    def get_deepcopy(self):
        return copy.deepcopy(self)

    def get_new(self):
        to_return = self.get_deepcopy()
        to_return.reset()

        return to_return

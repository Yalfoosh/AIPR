import copy
from typing import Any, Callable, Dict, Tuple

import numpy as np

from .integrator import Integrator


class ExplicitIntegrator(Integrator):
    pass


class EulerIntegrator(ExplicitIntegrator):
    @staticmethod
    def __check_generate_function_arguments(**kwargs):
        if "a" not in kwargs:
            raise KeyError('Argument kwargs must contain an entry with key "a"!')

        if "b" not in kwargs:
            raise KeyError('Argument kwargs must contain an entry with key "b"!')

        a, b = kwargs["a"], kwargs["b"]

        if not isinstance(a, np.ndarray):
            raise TypeError(
                f'Expected kwargs["a"] to be a np.ndarray, instead it is {type(a)}.'
            )

        if not isinstance(b, np.ndarray):
            raise TypeError(
                f'Expected kwargs["b"] to be a np.ndarray, instead it is {type(b)}.'
            )

        return copy.deepcopy(a), copy.deepcopy(b)

    def generate_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_function_arguments(**kwargs)
        r = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])

        m = np.eye(a.shape[0]) + step * a
        n = step * b

        def __f(x: np.ndarray, t: float):
            return m @ x + n @ r(t)

        return __f


class RungeKutta4Integrator(ExplicitIntegrator):
    @staticmethod
    def __check_generate_function_arguments(**kwargs):
        if "a" not in kwargs:
            raise KeyError('Argument kwargs must contain an entry with key "a"!')

        if "b" not in kwargs:
            raise KeyError('Argument kwargs must contain an entry with key "b"!')

        a, b = kwargs["a"], kwargs["b"]

        if not isinstance(a, np.ndarray):
            raise TypeError(
                f'Expected kwargs["a"] to be a np.ndarray, instead it is {type(a)}.'
            )

        if not isinstance(b, np.ndarray):
            raise TypeError(
                f'Expected kwargs["b"] to be a np.ndarray, instead it is {type(b)}.'
            )

        return copy.deepcopy(a), copy.deepcopy(b)

    def generate_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_function_arguments(**kwargs)
        r = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])
        half_step = step / 2.0
        sixth_step = half_step / 3.0

        def __f(x: np.ndarray, t: float):
            _1 = a @ x + b @ r(t)
            _2 = a @ (x + half_step * _1) + b @ r(t + half_step)
            _3 = a @ (x + half_step * _2) + b @ r(t + half_step)
            _4 = a @ (x + step * _3) + b @ r(t + step)

            return x + sixth_step * (_1 + 2 * (_2 + _3) + _4)

        return __f

import copy
from typing import Callable

import numpy as np

from .explicit import ExplicitIntegrator
from .integrator import Integrator


class ImplicitIntegrator(Integrator):
    def generate_correct_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        raise NotImplementedError


class InverseEulerIntegrator(ImplicitIntegrator):
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

    @staticmethod
    def __check_generate_correct_function_arguments(**kwargs):
        return InverseEulerIntegrator.__check_generate_function_arguments(**kwargs)

    def generate_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_function_arguments(**kwargs)
        r = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])

        p = np.linalg.inv(np.eye(a.shape[0]) - step * a)
        q = p @ (step * b)

        def __f(x: np.ndarray, t: float):
            return p @ x + q @ r(t + step)

        return __f

    def generate_correct_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_correct_function_arguments(**kwargs)
        r = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])

        def __f(x: np.ndarray, prediction: np.ndarray, t: float):
            return x + step * (a @ prediction + b @ r(t))

        return __f


class TrapezoidalIntegrator(ImplicitIntegrator):
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

    @staticmethod
    def __check_generate_correct_function_arguments(**kwargs):
        return TrapezoidalIntegrator.__check_generate_function_arguments(**kwargs)

    def generate_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_function_arguments(**kwargs)
        r_function = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])
        half_step = step / 2.0

        _inv = np.linalg.inv(np.eye(a.shape[0]) - half_step * a)
        _ninv = np.eye(a.shape[0]) + half_step * a

        r = _inv @ _ninv
        s = _inv @ (half_step * b)

        def __f(x: np.ndarray, t: float):
            return r @ x + s @ (r_function(t) + r_function(t + step))

        return __f

    def generate_correct_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        a, b = self.__check_generate_correct_function_arguments(**kwargs)
        r = copy.deepcopy(kwargs["time_function"])

        step = copy.deepcopy(kwargs["step"])
        half_step = step / 2.0

        def __f(x: np.ndarray, prediction: np.ndarray, t: float):
            return x + half_step * (a @ (x + prediction) + b @ (r(t) + r(t + step)))

        return __f

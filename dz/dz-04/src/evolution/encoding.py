import copy
from typing import Tuple, Union

import numpy as np


from .module import Module
from .utils import bin2dec_vector, dec2bin_vector


class BinaryEncoder(Module):
    @staticmethod
    def __check_init_args(
        dim: int, interval: Union[Tuple[Union[float, int], Union[float, int]]]
    ) -> Tuple[int, np.ndarray]:
        if not isinstance(dim, int):
            raise TypeError(
                f"Expected argument dim to be an int, instead it is {type(dim)}."
            )
        if dim < 1:
            raise ValueError(
                f"Expected argument dim to be a positive integer, instead it is {dim}."
            )

        if not isinstance(interval, tuple):
            try:
                interval = tuple(interval)
            except Exception:
                raise TypeError(
                    "Expected argument interval to be a tuple, instead it is "
                    f"{type(interval)}"
                )

        if len(interval) < 1:
            raise ValueError("Expected argument interval to be non-empty.")

        if len(interval) < 2:
            interval = (interval[0], 0) if interval < 0 else (0, interval[0])

        interval = interval[:2]

        if interval[0] > interval[1]:
            raise ValueError(
                "Expected argument interval to have a first item smaller than the "
                f"second one, instead it is {interval}."
            )

        interval = np.array(interval, dtype=np.float)

        return dim, interval

    def __init__(
        self, dim: int, interval: Union[Tuple[Union[float, int], Union[float, int]]]
    ):
        self.__dim, self.__interval = self.__check_init_args(dim=dim, interval=interval)

        max_int = int(2 ** dim)
        self.__quantum = (interval[1] - interval[0]) / max_int
        self.__max_value = max_int - 1

    @property
    def dim(self):
        return self.__dim

    @property
    def interval(self):
        return self.__interval

    @property
    def quantum(self):
        return self.__quantum

    @property
    def max_value(self):
        return self.__max_value

    def _input_to_int(self, x) -> int:
        return np.maximum(
            0, np.minimum(self.max_value, (x - self.interval[0]) / self.quantum)
        ).astype(np.int)

    def apply(self, x):
        x_int = self._input_to_int(x)

        return dec2bin_vector(x_int, self.dim)

    def __str__(self):
        return f"BinaryEncoder({self.dim} bit, {self.interval})"


class BinaryDecoder(Module):
    @staticmethod
    def __check_init_args(
        dim: int, interval: Union[Tuple[Union[float, int], Union[float, int]]]
    ) -> Tuple[int, np.ndarray]:
        if not isinstance(dim, int):
            raise TypeError(
                f"Expected argument dim to be an int, instead it is {type(dim)}."
            )
        if dim < 1:
            raise ValueError(
                f"Expected argument dim to be a positive integer, instead it is {dim}."
            )

        if not isinstance(interval, tuple):
            try:
                interval = tuple(interval)
            except Exception:
                raise TypeError(
                    "Expected argument interval to be a tuple, instead it is "
                    f"{type(interval)}"
                )

        if len(interval) < 1:
            raise ValueError("Expected argument interval to be non-empty.")

        if len(interval) < 2:
            interval = (interval[0], 0) if interval < 0 else (0, interval[0])

        interval = interval[:2]

        if interval[0] > interval[1]:
            raise ValueError(
                "Expected argument interval to have a first item smaller than the "
                f"second one, instead it is {interval}."
            )

        interval = np.array(interval, dtype=np.float)

        return dim, interval

    def __init__(
        self, dim: int, interval: Union[Tuple[Union[float, int], Union[float, int]]]
    ):
        self.__dim, self.__interval = self.__check_init_args(dim=dim, interval=interval)
        self.__coefficients = np.array(
            [int(2 ** i) for i in range(self.dim)], dtype=np.int32
        )

        max_int = int(2 ** dim)
        self.__quantum = (interval[1] - interval[0]) / max_int
        self.__max_value = max_int - 1

    @property
    def dim(self):
        return self.__dim

    @property
    def interval(self):
        return self.__interval

    @property
    def quantum(self):
        return self.__quantum

    @property
    def max_value(self):
        return self.__max_value

    def apply(self, x):
        x_int = bin2dec_vector(x)

        return (x_int * self.quantum) + self.interval[0]

    def __str__(self):
        return f"BinaryDecoder({self.dim} bit, {self.interval})"

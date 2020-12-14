from typing import Optional, Tuple, Union

import numpy as np

from .module import Module
from .population import Specimen


class Crossover(Module):
    pass


class CrossoverInsideInterval(Crossover):
    @staticmethod
    def __check_init_args(
        interval: Optional[Tuple[Union[float, int], Union[float, int]]]
    ):
        if interval is not None:
            interval = interval[:2]

            for value in interval:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        "Expected argument interval to contain only float or int "
                        f"values, instead they contain {type(value)}."
                    )

            if len(interval) == 0:
                interval = None
            elif len(interval) == 1:
                interval = (interval[0], None)
            else:
                if interval[1] < interval[0]:
                    raise ValueError(
                        "Expected argument interval to be sorted in ascending order, "
                        f"instead it is {interval}."
                    )

        return interval

    def __init__(self, interval: Optional[Tuple[Union[float, int], Union[float, int]]]):
        self._interval = self.__check_init_args(interval=interval)

    @property
    def interval(self):
        return self._interval

    def __str__(self):
        return "CrossoverInsideInterval"


class AveragingCrossover(CrossoverInsideInterval):
    def __init__(
        self, interval: Optional[Tuple[Union[float, int], Union[float, int]]] = None
    ):
        super().__init__(interval)

    def apply(self, parents: np.ndarray) -> np.ndarray:
        child = np.mean(parents, axis=0)

        if self.interval is not None:
            child = np.clip(child, *self.interval)

        return np.array([child])

    def __str__(self):
        return "AveragingCrossover operator"


class ChooseOneCrossover(CrossoverInsideInterval):
    def __init__(
        self, interval: Optional[Tuple[Union[float, int], Union[float, int]]] = None
    ):
        super().__init__(interval)

    def apply(self, parents: np.ndarray) -> np.ndarray:
        indices = np.random.choice(len(parents), len(parents[0]))

        child = []
        for i, index in enumerate(indices):
            child.append(parents[index][i])
        child = np.array(child)

        if self.interval is not None:
            child = np.clip(child, *self.interval)

        return np.array([child])

    def __str__(self):
        return "ChooseOneCrossover operator"


class ANDCrossover(Crossover):
    def apply(self, parents: np.ndarray):
        return np.array([np.product(parents, axis=0)])

    def __str__(self):
        return "ANDCrossover operator"


class XORCrossover(Crossover):
    def apply(self, parents: np.ndarray):
        return np.array([np.sum(parents, axis=0) % 2])

    def __str__(self):
        return "XORCrossover operator"


class XORTransformedCrossover(Crossover):
    @staticmethod
    def __check_init_args(weight: float, bias: float):
        if not isinstance(weight, float):
            raise TypeError(
                f"Expected argument weight to be a float, instead it is {type(weight)}."
            )

        if not isinstance(bias, float):
            raise TypeError(
                f"Expected argument bias to be a float, instead it is {type(bias)}."
            )

        return weight, bias

    def __init__(self, weight: float = 3.0, bias: float = 1.0):
        weight, bias = self.__check_init_args(weight=weight, bias=bias)

        self._weight = weight
        self._bias = bias
        self._function = np.vectorize(lambda x: int(x * self.weight + self.bias) % 2)
        self._crossover = XORCrossover()

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    def apply(self, parents: np.ndarray):
        return self._crossover(self._function(parents))

    def __str__(self):
        return "XORTransformedCrossover operator"

from typing import Optional, Tuple, Union

import numpy as np

from .module import Module


class Mutation(Module):
    @staticmethod
    def check_mutation_probability(mutation_probability: float):
        if not isinstance(mutation_probability, float):
            raise TypeError(
                "Expected argument mutation_probability to be a float, instead it is "
                f"{type(mutation_probability)}."
            )

        if not (0.0 <= mutation_probability <= 1.0):
            raise ValueError(
                "Expected argument mutation_probability to be in range [0, 1], instead "
                f"it is {mutation_probability}."
            )

        return mutation_probability

    @staticmethod
    def __check_init_args(mutation_probability: float):
        mutation_probability = Mutation.check_mutation_probability(
            mutation_probability=mutation_probability
        )

        return mutation_probability

    def __init__(self, mutation_probability: float = 1e-2):
        self._mutation_probability = self.__check_init_args(
            mutation_probability=mutation_probability
        )

    @property
    def mutation_probability(self):
        return self._mutation_probability

    @property
    def p(self):
        return self.mutation_probability


class MutationInsideInterval(Mutation):
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

    def __init__(
        self,
        mutation_probability: float,
        interval: Optional[Tuple[Union[float, int], Union[float, int]]],
    ):
        super().__init__(mutation_probability=mutation_probability)

        self._interval = self.__check_init_args(interval=interval)

    @property
    def interval(self):
        return self._interval

    def __str__(self):
        return "CrossoverInsideInterval"


class GaussianMutation(MutationInsideInterval):
    @staticmethod
    def __check_init_args(mean: float, scale: float):
        if not isinstance(mean, float):
            raise TypeError(
                f"Expected argument mean to be a float, instead it is {type(mean)}."
            )

        if not isinstance(scale, float):
            raise TypeError(
                f"Expected argument scale to be a float, instead it is {type(scale)}."
            )

        if scale <= 0.0:
            raise ValueError(
                "Expected argument scale to be a non-zero positive float, instead it "
                f"is {scale}."
            )

        return mean, scale

    def __init__(
        self,
        mutation_probability: float = 1e-2,
        mean: float = 0.0,
        scale: float = 1.0,
        interval: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
    ):
        super().__init__(mutation_probability=mutation_probability, interval=interval)

        mean, scale = self.__check_init_args(mean, scale)

        self._mean = mean
        self._scale = scale

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._scale

    def _correct_args(self, mean: Optional[float], scale: Optional[float]):
        if mean is None:
            mean = self.mean

        if scale is None:
            scale = self.scale

        return self.__check_init_args(mean, scale)

    def apply(
        self,
        specimen: np.ndarray,
        mutation_probability: Optional[float] = None,
        mean: Optional[float] = None,
        scale: Optional[float] = None,
    ):
        if mutation_probability is None:
            mutation_probability = self.mutation_probability
        else:
            mutation_probability = super().check_mutation_probability(
                mutation_probability=mutation_probability
            )

        mean, scale = self._correct_args(mean, scale)

        corruption = np.random.choice(
            np.array([0.0, 1.0]),
            size=specimen.size,
            p=[1.0 - mutation_probability, mutation_probability],
        )
        corruption = corruption * np.random.normal(
            loc=mean, scale=scale, size=specimen.size
        )

        new_specimen = specimen + corruption

        if self.interval is not None:
            new_specimen = np.clip(new_specimen, *self.interval)

        return new_specimen

    def __str__(self):
        return "GaussianMutation operator"


class StochasticCorruptionMutation(Mutation):
    @staticmethod
    def __check_init_args(mutation_probability: float):
        mutation_probability = Mutation.check_mutation_probability(
            mutation_probability=mutation_probability
        )

        return mutation_probability

    def __init__(self, mutation_probability: float = 1e-2):
        self._mutation_probability = self.__check_init_args(
            mutation_probability=mutation_probability
        )

    def apply(self, specimen: np.ndarray, mutation_probability: Optional[float] = None):
        if mutation_probability is None:
            mutation_probability = self.mutation_probability
        else:
            mutation_probability = super().check_mutation_probability(
                mutation_probability=mutation_probability
            )

        corruption = np.random.choice(
            np.array([0, 1]),
            size=specimen.shape,
            p=[1.0 - mutation_probability, mutation_probability],
        )

        return (specimen + corruption) % 2

    def __str__(self):
        return "StochasticCorruptionMutation operator"

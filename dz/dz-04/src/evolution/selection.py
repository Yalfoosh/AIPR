import copy
from typing import Tuple

import numpy as np

from .function import Function
from .module import Module
from .population import Population


class Selection(Module):
    pass


class TournamentSelection(Selection):
    @staticmethod
    def __check_init_args(tournament_size: int, n_winners: int):
        if not isinstance(tournament_size, int):
            raise TypeError(
                "Expected argument tournament_size to be an int, instead it is "
                f"{tournament_size}."
            )

        if tournament_size < 1:
            raise ValueError(
                "Expected argument tournament_size to be a positive integer, instead "
                f"it is {tournament_size}."
            )

        if not isinstance(n_winners, int):
            raise TypeError(
                "Expected argument n_winners to be an int, instead it is "
                f"{n_winners}."
            )

        if n_winners < 1:
            raise ValueError(
                "Expected argument n_winners to be a positive integer, instead "
                f"it is {n_winners}."
            )

        if n_winners > tournament_size:
            raise ValueError(
                "Expected argument n_winners to be less or equal to the tournament "
                f"size ({tournament_size}), instead it is {n_winners}."
            )

        return tournament_size, n_winners

    def __init__(self, tournament_size: int = 3, n_winners: int = 2):
        tournament_size, n_winners = self.__check_init_args(tournament_size, n_winners)

        self._tournament_size = tournament_size
        self._n_winners = n_winners

    @property
    def tournament_size(self):
        return self._tournament_size

    @property
    def n_winners(self):
        return self._n_winners

    def apply(self, population: Population) -> Tuple[np.ndarray, np.ndarray]:
        if len(population) == 0:
            return np.array([]), np.array([])

        participants = np.random.choice(
            len(population), self.tournament_size, replace=False
        )
        participants = np.sort(participants)

        if self.tournament_size == self.n_winners:
            winners = copy.deepcopy(participants)
        else:
            winners = participants[: self.n_winners]

        return participants, winners

    def __str__(self):
        return (
            f"TournamentSelection ({self.tournament_size}-tournament, {self.n_winners} "
            "win)"
        )

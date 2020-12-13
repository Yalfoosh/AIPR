import copy

import numpy as np

from .function import Function
from .module import Module
from .population import Population


class TournamentSelection(Module):
    @staticmethod
    def __check_init_args(tournament_size: int):
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

        return tournament_size

    def __init__(self, tournament_size: int = 3):
        tournament_size = self.__check_init_args(tournament_size)

        self._tournament_size = tournament_size

    @property
    def tournament_size(self):
        return self._tournament_size

    def apply(self, population: Population, n_winners: int = 2):
        if len(population) < self.tournament_size:
            raise RuntimeError(
                "Expected argument population to be at least as big as the "
                f"tournament size ({self.tournament_size}), instead it is of length "
                f"{len(population)}."
            )

        if self.tournament_size < n_winners:
            raise RuntimeError(
                "Expected argument n_winners to be less or equal to tournament_size "
                f"({self.tournament_size}), instead it is {n_winners}."
            )

        if len(population) == 0:
            return np.array([]), np.array([])

        participants = np.random.choice(
            len(population), self.tournament_size, replace=False
        )

        if self.tournament_size == n_winners:
            winners = copy.deepcopy(participants)
        else:
            winners = participants[np.argpartition(participants, n_winners)[:n_winners]]

        return participants, winners

    def __str__(self):
        return f"TournamentSelection ({self.tournament_size}-tournament)"

import copy
from typing import Optional, Tuple, Union

import numpy as np

from .evolution.crossover import Crossover
from .evolution.function import Function
from .evolution.module import Module
from .evolution.mutation import Mutation
from .evolution.population import Population
from .evolution.selection import TournamentSelection


class EvolutionaryTournament(Module):
    @staticmethod
    def __check_init_args(
        selection: TournamentSelection, crossover: Crossover, mutation: Mutation
    ):
        if not isinstance(selection, TournamentSelection):
            raise TypeError(
                "Expected argument selection to be a TournamentSelection, instead it "
                f"is {type(selection)}."
            )

        if not isinstance(crossover, Crossover):
            raise TypeError(
                "Expected argument crossover to be a Crossover, instead it is "
                f"{type(crossover)}."
            )

        if not isinstance(mutation, Mutation):
            raise TypeError(
                "Expected argument mutation to be a Mutation, instead it is "
                f"{type(mutation)}."
            )

        return (
            copy.deepcopy(selection),
            copy.deepcopy(crossover),
            copy.deepcopy(mutation),
        )

    def __init__(
        self,
        selection: TournamentSelection,
        crossover: Crossover,
        mutation: Mutation,
    ):
        selection, crossover, mutation = self.__check_init_args(
            selection=selection, crossover=crossover, mutation=mutation
        )

        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation

    @property
    def selection(self) -> TournamentSelection:
        return self._selection

    @property
    def crossover(self) -> Crossover:
        return self._crossover

    @property
    def mutation(self) -> Mutation:
        return self._mutation

    def apply(
        self,
        population: Population,
        mutation_probability: Optional[float] = None,
    ) -> np.ndarray:
        participants, winners = self.selection(population=population)
        parents = np.array([x.element for x in population[winners]])

        population.pop(int(participants[-1]))

        new_specimen = self.crossover(parents=parents)[0]
        new_specimen = self.mutation(
            new_specimen, mutation_probability=mutation_probability
        )

        population.add(new_specimen)

        return new_specimen

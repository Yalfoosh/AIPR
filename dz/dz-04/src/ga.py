import copy
import sys
from typing import Callable, Tuple, Union

import numpy as np
from tqdm import tqdm

from .evolutionary_tournament import EvolutionaryTournament
from .evolution.function import Function
from .evolution.population import Population


def simple_ga(
    wellness_function: Function,
    specimen_length: int,
    interval: Tuple[Union[float, int], Union[float, int]],
    evolutionary_tournament: EvolutionaryTournament,
    max_retries: int = 3,
    max_iterations: int = 100,
    population_capacity: int = 20,
    elitism: int = 1,
    epsilon: float = 1e-6,
    stop_prematurely: bool = True,
    mutation_scheduler: Callable = lambda x: 1e-2,
    encoding_function: Callable = lambda x: x,
    decoding_function: Callable = lambda x: x,
):
    specimina = list()
    stop_execution = False

    for _ in range(max_retries):
        population = Population(
            wellness_function=wellness_function,
            capacity=population_capacity,
            elitism=elitism,
        )

        population.invade(
            encoding_function(
                np.random.uniform(*interval, (population_capacity, specimen_length))
            )
        )

        iterator = tqdm(range(max_iterations), file=sys.stdout)
        best = None

        for n_iter in iterator:
            mutation_probability = mutation_scheduler(n_iter)

            evolutionary_tournament(
                population=population, mutation_probability=mutation_probability
            )

            new_point = population[0]

            if best is None or best.wellness < new_point.wellness:
                best = copy.deepcopy(new_point)
                iterator.set_description(
                    f"[{n_iter}] {decoding_function(best.element)} "
                    f"({best.wellness:.02f})"
                )

            if abs(best.wellness) < epsilon:
                stop_execution = True
                break

        specimina.append(best)

        if stop_prematurely and stop_execution:
            break

    return specimina

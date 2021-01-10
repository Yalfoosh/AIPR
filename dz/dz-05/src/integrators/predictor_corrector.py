import copy
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .implicit import ExplicitIntegrator, ImplicitIntegrator
from .integrator import Integrator


class PredictorCorrectorIntegrator(Integrator):
    @staticmethod
    def __check_init_args(predictor: ExplicitIntegrator, corrector: ImplicitIntegrator):
        if not isinstance(predictor, ExplicitIntegrator):
            raise TypeError(
                "Expected argument predictor to be a ExplicitIntegrator, instead it is "
                f"{type(predictor)}."
            )

        if not isinstance(corrector, ImplicitIntegrator):
            raise TypeError(
                "Expected argument corrector to be a ImplicitIntegrator, instead it is "
                f"{type(corrector)}."
            )

        return copy.deepcopy(predictor), copy.deepcopy(corrector)

    @staticmethod
    def __check_integrate_arguments(*args, **kwargs):
        if len(args) > 0:
            kwargs["n_corrector_repeats"] = args[0]

        n_corrector_repeats = kwargs.get("n_corrector_repeats", 1)

        if not isinstance(n_corrector_repeats, int):
            raise TypeError(
                "Expected argument n_corrector_repeats to be an int, instead it is "
                f"{type(n_corrector_repeats)}."
            )

        if n_corrector_repeats < 1:
            raise ValueError(
                "Expected argument n_corrector_repeats to be a positive int, instead "
                f"it is {n_corrector_repeats}."
            )

        return n_corrector_repeats

    def __init__(self, predictor: ExplicitIntegrator, corrector: ImplicitIntegrator):
        predictor, corrector = self.__check_init_args(
            predictor=predictor, corrector=corrector
        )

        self._predictor = predictor
        self._corrector = corrector

    @property
    def predictor(self):
        return self._predictor

    @property
    def corrector(self):
        return self._corrector

    def _integrate(
        self,
        step: float,
        interval: Tuple[float, float],
        initial_state: np.ndarray,
        time_function: Optional[Callable[[Tuple, Dict], np.ndarray]],
        steps_to_print: int,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        to_return = dict()

        (
            step,
            interval,
            initial_state,
            time_function,
            steps_to_print,
        ) = Integrator._check_integrate_arguments(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            steps_to_print=steps_to_print,
        )

        n_corrector_repeats = self.__check_integrate_arguments(*args, **kwargs)

        current_state = copy.deepcopy(initial_state)

        predictor_function = self.predictor.generate_function(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            *args,
            **kwargs,
        )
        corrector_function = self.corrector.generate_correct_function(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            *args,
            **kwargs,
        )

        a = kwargs["a"]
        b = kwargs["b"]

        to_return["states"] = [
            {
                "x": copy.deepcopy(current_state),
                "t": None,
                "i": None,
            }
        ]

        iterator = enumerate(np.arange(interval[0], interval[1] + step, step))

        if steps_to_print > 0:
            iterator = tqdm(
                iterator,
                desc=f"x_0 = {current_state.T}" if steps_to_print != 0 else "",
                file=sys.stdout,
            )

        for i, t in iterator:
            if steps_to_print > 0 and i % steps_to_print == 0:
                iterator.set_description(f"x_{i}(t = {t}) = {current_state.T}")

            prediction = predictor_function(current_state, t)

            for _ in range(n_corrector_repeats):
                prediction = corrector_function(current_state, prediction, t)

            current_state = prediction

            to_return["states"].append(
                {
                    "x": copy.deepcopy(current_state),
                    "t": copy.deepcopy(t),
                    "i": copy.deepcopy(i),
                }
            )

        return to_return

    @property
    def name(self):
        return f"Predictor {self.predictor.name}, Corrector {self.corrector.name}"

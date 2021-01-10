import copy
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


class Integrator:
    @staticmethod
    def __get_default_time_function(initial_state: np.ndarray):
        time_constant = np.zeros(shape=initial_state.shape)

        def __f(*args, **kwargs):
            return time_constant

        return __f

    @staticmethod
    def _check_integrate_arguments(
        step: float,
        interval: Tuple[float, float],
        initial_state: np.ndarray,
        time_function: Optional[Callable[[Tuple, Dict], np.ndarray]],
        steps_to_print: int,
    ):
        if time_function is None:
            time_function = Integrator.__get_default_time_function(
                initial_state=initial_state
            )

        if not isinstance(step, float):
            raise TypeError(
                f"Expected argument step to be a float, instead it is {type(step)}."
            )

        if not isinstance(interval, tuple):
            raise TypeError(
                "Expected argument interval to be a tuple, instead it is "
                f"{type(interval)}."
            )

        if not isinstance(initial_state, np.ndarray):
            raise TypeError(
                "Expected argument initial_state to be a numpy.ndarray, instead it is "
                f"{type(initial_state)}."
            )

        if not callable(time_function):
            raise TypeError(
                "Expected argument time_function to be callable, instead it is "
                f"{type(time_function)}."
            )

        if not isinstance(steps_to_print, int):
            raise TypeError(
                "Expected argument steps_to_print to be an int, instead it is "
                f"{type(steps_to_print)}."
            )

        if len(interval) != 2:
            raise ValueError(
                "Expected argument interval to have length of 2, instead it has a "
                f"length of {len(interval)}."
            )

        if step <= 0:
            raise ValueError(
                f"Expected argument step to be a positive float, instead it is {step}."
            )

        if interval[0] > interval[1]:
            raise ValueError(
                "Expected argument interval to have a first item smaller than the "
                f"seconds, instead found {interval[0]} and {interval[1]}."
            )

        if steps_to_print < 0:
            raise ValueError(
                "Expected argument steps_to_print to be a non-negative integer, "
                f"instead it is {steps_to_print}."
            )

        return step, interval, initial_state, time_function, steps_to_print

    @property
    def name(self):
        return self.__class__.__name__

    def generate_function(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        raise NotImplementedError

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
        ) = self._check_integrate_arguments(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            steps_to_print=steps_to_print,
        )

        current_state = copy.deepcopy(initial_state)
        function = self.generate_function(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            *args,
            **kwargs,
        )
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

            current_state = function(current_state, t)
            to_return["states"].append(
                {
                    "x": copy.deepcopy(current_state),
                    "t": copy.deepcopy(t),
                    "i": copy.deepcopy(i),
                }
            )

        return to_return

    def __call__(
        self,
        step: float,
        interval: Tuple[float, float],
        initial_state: np.ndarray,
        time_function: Optional[Callable[[Tuple, Dict], np.ndarray]] = None,
        steps_to_print: int = 0,
        *args,
        **kwargs,
    ):
        return self._integrate(
            step=step,
            interval=interval,
            initial_state=initial_state,
            time_function=time_function,
            steps_to_print=steps_to_print,
            *args,
            **kwargs,
        )
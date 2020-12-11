import copy
from typing import Any, Iterable, List, Optional, Tuple, Union

from sortedcontainers import SortedList

from .function import Function


class Specimen:
    def __init__(self, element: Any, wellness: Union[float, int]):
        self._element = copy.deepcopy(element)
        self._wellness = copy.deepcopy(wellness)

    @property
    def element(self) -> Any:
        return self._element

    @property
    def wellness(self) -> Union[float, int]:
        return self._wellness


class SpecimenGenerator:
    @staticmethod
    def __check_init_args(wellness_function: Function) -> Function:
        if not isinstance(wellness_function, Function):
            raise TypeError(
                "Expected argument wellness_function to be a Function, instead it is "
                f"{type(wellness_function)}."
            )

        return wellness_function

    @property
    def wellness_function(self):
        return self._wellness_function

    def __init__(self, wellness_function: Function):
        wellness_function = self.__check_init_args(wellness_function=wellness_function)

        self._wellness_function = wellness_function

    def get(self, element: Any) -> Specimen:
        return Specimen(element=element, wellness=self._wellness_function(element))

    def __call__(self, element) -> Specimen:
        return self.get(element)


class Population:
    @staticmethod
    def __check_init_args(
        wellness_function: Function,
        iterable: Optional[Iterable[Union[Any, Specimen]]],
        capacity: int,
    ) -> Tuple[SpecimenGenerator, List[Specimen], int]:
        if not isinstance(wellness_function, Function):
            raise TypeError(
                "Expected argument wellness_function to be a Function, instead it is "
                f"{type(wellness_function)}."
            )

        specimen_generator = SpecimenGenerator(wellness_function=wellness_function)

        # We do this before the iterable so we don't call the
        # wellness function in case our max_size argument is
        # incorrectly passed.
        if not isinstance(capacity, int):
            raise TypeError(
                "Expected argument capacity to be an int, instead it is "
                f"{type(capacity)}."
            )

        if capacity < 1:
            raise ValueError(
                "Expected argument capacity to be a positive integer, instead it is "
                f"{capacity}."
            )

        if iterable is not None:
            try:
                iter(iterable)
            except TypeError:
                raise TypeError(
                    "Expected argument iterable to be an iterable or None, instead it "
                    f"is {type(iterable)}."
                )

            iterable = [
                x if isinstance(x, Specimen) else specimen_generator(x)
                for x in iterable
            ]

            if len(iterable) > capacity:
                iterable.sort(key=lambda x: -x.wellness)
                iterable = iterable[:capacity]

        return wellness_function, specimen_generator, iterable, capacity

    def __init__(
        self,
        wellness_function: Function,
        iterable: Optional[Iterable[Union[Any, Specimen]]] = None,
        capacity: int = 10,
    ):
        specimen_generator, iterable, capacity = self.__check_init_args(
            wellness_function=wellness_function, iterable=iterable, capacity=capacity
        )

        self._specimen_generator = specimen_generator
        self._capacity = capacity

        self.__content = SortedList(iterable=iterable, key=lambda x: -x.wellness)

    @property
    def wellness_function(self):
        return self._specimen_generator.wellness_function

    @property
    def specimen_generator(self):
        return self._specimen_generator

    @property
    def capacity(self):
        return self._capacity

    def ban(self, x: Union[Any, Specimen]):
        if x is not None:
            self.__content.discard(
                x if isinstance(x, Specimen) else self.specimen_generator(x)
            )

    def cull(self, n_additional: int = 0):
        if not isinstance(n_additional, int):
            raise TypeError(
                "Expected argument n_additional to be an int, instead it is "
                f"{type(n_additional)}."
            )

        if n_additional < 0:
            raise ValueError(
                "Expected argument n_additional to be a non-negative integer, instead "
                f"it is {n_additional}."
            )

        n_to_remove = len(self.__content) - self.capacity + n_additional

        if n_to_remove < len(self.__content):
            for _ in range(n_to_remove):
                self.__content.pop()
        else:
            self.__content.clear()

    def pop(self, index: int = -1):
        if not isinstance(index, int):
            raise TypeError(
                f"Expected argument index to be an int, instead it is {type(index)}."
            )

        return self.__content.pop(index)

    def append(self, specimen: Specimen):
        if not isinstance(specimen, Specimen):
            raise TypeError(
                "Expected argument speciment to be a Specimen, instead it is "
                f"{type(specimen)}."
            )

        self.__content.add(specimen)

    def add(self, specimen: Specimen, remove_before: bool = True):
        if not isinstance(remove_before, bool):
            raise TypeError(
                "Expected argument remove_before to be a bool, instead it is "
                f"{type(remove_before)}."
            )

        if remove_before:
            self.cull(n_additional=1)

        self.append(specimen=specimen)

        self.cull()

    def assimilate(self, specimina: Iterable[Specimen]):
        try:
            iter(specimina)
        except TypeError:
            raise TypeError(
                "Expected argument specimina to be an iterable, instead it is"
                f"{type(specimina)}."
            )

        for x in specimina:
            self.append(x if isinstance(x, Specimen) else self.specimen_generator(x))

    def invade(self, specimina: Iterable[Specimen], remove_before: bool = True):
        if not isinstance(remove_before, bool):
            raise TypeError(
                "Expected argument remove_before to be a bool, instead it is "
                f"{type(remove_before)}."
            )

        if remove_before:
            self.cull(len(specimina))

        self.assimilate(specimina)

        self.cull()

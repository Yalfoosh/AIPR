from textwrap import dedent
from typing import List, Tuple, Union


def sign(x: Union[float, int], zero_is_positive: bool = False) -> int:
    """
    Gets the sign of a number.

    Args:
        x (Union[float, int]): A float or int representing the
        value you want to check the sign of.
        zero_is_positive (bool): A boolean determining whether or
        not to think of 0 as a positive number.

    Raises:
        TypeError: Raised if x is None.
        TypeError: Raised if x is not a float or int.

    Returns:
        int: -1 if x < 0
              0 if x == 0
              1 if x > 0 (or x >= 0 if zero_is_positive is True)
    """
    if x is None:
        raise TypeError("Argument x mustn't be None!")

    if not isinstance(x, (float, int)):
        raise TypeError(
            f"Expected argument x to be a float or int, instead it is {type(x)}."
        )

    return -1 if x < 0 else 1 if x > 0 else 1 if zero_is_positive else 0


def count_swaps_in_row_order(row_order: Union[List[int], Tuple[int]]) -> int:
    """
    Counts the number of swaps in a row order.

    Args:
        row_order (Union[List[int], Tuple[int]]): A list or tuple
        of ints representing the order of rows.

    Returns:
        int: The minimum number of swaps it takes for a
        range(len(row_order)) to reach row_order.
    """
    count = 0

    for i in range(len(row_order)):
        if row_order[i] != i:
            row_order[row_order[i]], row_order[i] = (
                row_order[i],
                row_order[row_order[i]],
            )
            count += 1

    return count
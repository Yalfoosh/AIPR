import copy
from textwrap import dedent
from typing import Callable, List, Optional, Tuple, Union

from .exceptions import MatrixNotBroadcastableException, MatrixShapeMismatchException


class MatrixConfig:
    def __init__(self, height: int, width: int, dtype: Callable):
        if height is None:
            raise TypeError("Argument height mustn't be None!")

        if width is None:
            raise TypeError("Argument width mustn't be None!")

        if dtype is None:
            raise TypeError("Argument dtype mustn't be None!")

        if not isinstance(height, int):
            raise TypeError(
                f"Expected argument height to be an int, instead it is {type(height)}."
            )

        if not isinstance(width, int):
            raise TypeError(
                f"Expected argument width to be an int, instead it is {type(width)}."
            )

        if not callable(dtype):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument dtype to be a callable, instead it is \
                    {type(dtype)}.\
                    """
                )
            )

        if height < 1:
            raise ValueError(
                f"Expected argument height to be 1 or greater, instead it is {height}."
            )

        if width < 1:
            raise ValueError(
                f"Expected argument width to be 1 or greater, instead it is {width}."
            )

        self.shape = (height, width)
        self.dtype = dtype


class Matrix:
    def __init__(
        self,
        height: int,
        width: int,
        dtype: Callable = float,
        config: Optional[MatrixConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = MatrixConfig(height=height, width=width, dtype=dtype)

        self._config = copy.deepcopy(config)
        self._data = list()

        for _ in range(self.height):
            current_list = list()

            for _ in range(self.width):
                current_list.append(None)

            self._data.append(current_list)

    # region Properties
    @property
    def shape(self):
        return self._config.shape

    @property
    def height(self):
        return self._config.shape[0]

    @property
    def width(self):
        return self._config.shape[1]

    @property
    def dtype(self):
        return self._config.dtype

    @property
    def epsilon(self):
        return 1e-13

    @property
    def EPS(self):
        return self.epsilon

    # endregion

    # region Factory Methods
    @staticmethod
    def full(
        height: int,
        width: int,
        fill_value: Union[float, int],
        dtype: Optional[Callable] = None,
    ):
        if dtype is None:
            dtype = type(fill_value)

        to_return = Matrix(height=height, width=width, dtype=dtype)
        to_return.fill(fill_value)

        return to_return

    @staticmethod
    def zeros(height: int, width: int, dtype: Callable = float):
        to_return = Matrix(height=height, width=width, dtype=dtype)
        to_return.fill(to_return.dtype(0.0))

        return to_return

    @staticmethod
    def eye(height: int, width: int, dtype: Callable = float):
        to_return = Matrix.zeros(height=height, width=width, dtype=dtype)

        for i in range(min(to_return.height, to_return.width)):
            to_return[i][i] = to_return.dtype(1.0)

        return to_return

    @staticmethod
    def from_array(array: Union[List, Tuple]):
        if array is None:
            raise TypeError("Argument array mustn't be None!")

        if not isinstance(array, (list, tuple)):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument array to be a list or a tuple, instead it is \
                    {type(array)}.\
                    """
                )
            )

        if len(array) == 0:
            raise ValueError("Expected argument array to have at least one element.")

        for i in range(len(array)):
            if isinstance(array[i], (list, tuple)):
                for j in range(len(array[i])):
                    if isinstance(array[i][j], (list, tuple)):
                        raise TypeError(
                            dedent(
                                """\
                                Expected argument array to at most be of depth 2, yet \
                                it is deeper.\
                                """
                            )
                        )
            else:
                array[i] = [array[i]]

        height = len(array[i])
        width = max([0] + [len(x) for x in array])

        if min(height, width) == 0:
            raise ValueError("Expected argument array to be non-empty.")

        dtype = type(array[0][0])

        to_return = Matrix(height=height, width=width, dtype=dtype)

        for i in range(len(to_return)):
            for j in range(len(to_return[i])):
                value_to_insert = to_return.dtype(
                    array[i][j] if (i < len(array) and j < len(array[i])) else 0.0
                )

                to_return[i][j] = value_to_insert

        return to_return

    # endregion

    # region Submatrix Methods
    def diagonal(self) -> "Matrix":
        to_return = Matrix(
            height=1, width=min(self.height, self.width), dtype=self.dtype
        )

        for j in range(len(to_return[0])):
            to_return[0][j] = self[j][j]

        return to_return

    def reverse_diagonal(self) -> "Matrix":
        to_return = Matrix(
            height=1, width=min(self.height, self.width), dtype=self.dtype
        )

        offset = len(to_return[0]) - 1

        for j in range(len(to_return[0])):
            to_return[0][j] = self[j][offset - j]

        return to_return

    def row(self, index: str) -> "Matrix":
        if index is None:
            raise TypeError("Argument index mustn't be None!")

        if not isinstance(index, int):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument index to be an int, instead it is {type(index)}.\
                    """
                )
            )

        if not 0 <= index < self.height:
            raise ValueError(
                dedent(
                    f"""\
                    Expected argument index to be in range [0, {self.height}>, instead \
                    it is {index}.\
                    """
                )
            )

        to_return = Matrix(height=1, width=self.width, dtype=self.dtype)

        for j in range(to_return.width):
            to_return[0][j] = self[index][j]

        return to_return

    def column(self, index: int):
        if index is None:
            raise TypeError("Argument index mustn't be None!")

        if not isinstance(index, int):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument index to be an int, instead it is {type(index)}.\
                    """
                )
            )

        if not 0 <= index < self.width:
            raise ValueError(
                dedent(
                    f"""\
                    Expected argument index to be in range [0, {self.width}>, instead \
                    it is {index}.\
                    """
                )
            )

        to_return = Matrix(height=self.height, width=1, dtype=self.dtype)

        for i in range(to_return.height):
            to_return[i][0] = self[i][index]

        return to_return

    # endregion

    def fill(self, value: Union[float, int]):
        if value is None:
            raise TypeError("Argument value mustn't be None!")

        if not (isinstance(value, float) or isinstance(value, int)):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument value to be a float or int, instead it is \
                    {type(value)}.\
                    """
                )
            )

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] = self.dtype(value)

    # region Matrix Casting Methods
    def int(self):
        self._config.dtype = int

        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j])

    def float(self):
        self._config.dtype = float

        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j])

    # endregion

    # region Arithmetic Methods
    def add(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, (float, int)):
            to_swap = Matrix(height=self.height, width=self.width, dtype=type(other))
            to_swap.fill(other)

            other = to_swap

        if not isinstance(other, Matrix):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument other to eventually be a Matrix, instead it is \
                    {type(other)}.\
                    """
                )
            )

        if self.shape != other.shape:
            raise MatrixShapeMismatchException(
                dedent(
                    f"""\
                    Failed to do matrix addition: shapes {self.shape} & {other.shape} \
                    do not match.\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] += other[i][j]

    def sub(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, (float, int)):
            to_swap = Matrix(height=self.height, width=self.width, dtype=type(other))
            to_swap.fill(other)

            other = to_swap

        if not isinstance(other, Matrix):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument other to eventually be a Matrix, instead it is \
                    {type(other)}.\
                    """
                )
            )

        if self.shape != other.shape:
            raise MatrixShapeMismatchException(
                dedent(
                    f"""\
                    Failed to do matrix subtraction: shapes {self.shape} & \
                    {other.shape} do not match.\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] -= other[i][j]

    def mul(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    dedent(
                        f"""\
                        Failed to do Hadamard multiplication: shapes {self.shape} & \
                        {other.shape} do not match.
                        """
                    )
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument other to be a float or an int, instead it is \
                    {type(other)}.\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] *= other[i][j]

    def truediv(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    dedent(
                        f"""\
                        Failed to do Hadamard division: shapes {self.shape} & \
                        {other.shape} do not match.
                        """
                    )
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument other to be a float or an int, instead it is \
                    {type(other)}.\
                    """
                )
            )

        self.float()

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] /= other[i][j]

    # endregion

    # region Print Methods
    def pretty_print(
        self,
        left_space: str = "",
        tab: str = "  ",
        newline_char="\n",
        decimal_precision: int = 3,
    ):
        string_matrix = list()

        for i in range(len(self)):
            string_row = list()

            for j in range(len(self[i])):
                if self.dtype == float:
                    string_row.append(f"{self[i][j]:.0{decimal_precision}f}")
                elif self.dtype == int:
                    string_row.append(f"{self[i][j]:d}")
                else:
                    string_row.append(f"{self[i][j]}")

            string_matrix.append(string_row)

        max_str_len = max(
            [max([len(element) for element in row]) for row in string_matrix]
        )

        string_rows = [
            ", ".join(
                [" " * (max_str_len - len(element)) + f"{element}" for element in row]
            )
            for row in string_matrix
        ]
        translated_string_rows = [f"{left_space}{tab}[{x}]" for x in string_rows]
        string_block = f",{newline_char}".join(translated_string_rows)

        to_return = (
            f"{left_space}[{newline_char}{string_block}{newline_char}{left_space}]"
        )

        return to_return

    # endregion

    # region Dunder methods

    # region Data Access Dunder Methods
    def __getitem__(self, key):
        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j])

        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, item):
        self._data[key] = item

        for j in range(len(self._data[key])):
            if self._data[key][j] is not None:
                self._data[key][j] = self.dtype(self._data[key][j])

    # endregion

    # region Mathematical Dunder Methods
    def __add__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.add(other)

        return to_return

    def __truediv__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.truediv(other)

        return to_return

    def __mul__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.mul(other)

        return to_return

    def __sub__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.sub(other)

        return to_return

    # endregion

    # region Output Dunder Methods
    def __repr__(self):
        row_strings = [f"[{','.join([str(x) for x in row])}]" for row in self]
        row_string = ",".join(row_strings)

        return f"[{row_string}]"

    def __str__(self):
        return self.pretty_print(
            left_space="",
            tab="" if self.height == 1 else "  ",
            newline_char="" if self.height == 1 else "\n",
            decimal_precision=3,
        )

    # endregion

    # endregion

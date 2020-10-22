import copy
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .constants import WHITESPACE_REGEX
from .exceptions import (
    MatrixIsSingular,
    MatrixNotBroadcastableException,
    MatrixShapeMismatchException,
    NotSolvable,
)
from .utils import count_swaps_in_row_order, sign


class MatrixConfig:
    def __init__(
        self, height: int, width: int, dtype: Callable, epsilon: Union[float, int]
    ):
        if height is None:
            raise TypeError("Argument height mustn't be None!")

        if width is None:
            raise TypeError("Argument width mustn't be None!")

        if dtype is None:
            raise TypeError("Argument dtype mustn't be None!")

        if epsilon is None:
            raise TypeError("Argument epsilon mustn't be None!")

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
                f"Expected argument dtype to be a callable, instead it is {type(dtype)}"
                "."
            )

        if not isinstance(epsilon, (float, int)):
            raise TypeError(
                "Expected argument epsilon to be a float or int, instead it is "
                f"{type(epsilon)}."
            )

        if height < 1:
            raise ValueError(
                f"Expected argument height to be 1 or greater, instead it is {height}."
            )

        if width < 1:
            raise ValueError(
                f"Expected argument width to be 1 or greater, instead it is {width}."
            )

        if epsilon < 0:
            raise ValueError(
                "Expected argument epsilon to be 0 or greater, instead it is {epsilon}."
            )

        self.shape = (height, width)
        self.dtype = dtype
        self.epsilon = float(epsilon)


class Matrix:
    __matrix_environment_to_latex_environment = {
        "plain": "matrix",
        "square_bracket": "bmatrix",
        "brace": "Bmatrix",
        "bracket": "pmatrix",
        "vertical_bar": "vmatrix",
        "double_vertical_bar": "Vmatrix",
        "small": "smallmatrix",
    }

    def __init__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dtype: Callable = float,
        epsilon: float = 1e-13,
        config: Optional[MatrixConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = MatrixConfig(
                height=height, width=width, dtype=dtype, epsilon=epsilon
            )

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
        return self._config.epsilon

    @epsilon.setter
    def epsilon(self, value: Union[float, int]):
        if value is None:
            raise TypeError("Argument value mustn't be None!")

        if not isinstance(value, (float, int)):
            raise TypeError(
                "Expected argument other to be a float or int, instead it is "
                f"{type(value)}."
            )

        self._config.epsilon = float(value)

    @property
    def EPS(self):
        return self.epsilon

    @property
    def T(self):
        return self.transposed()

    @property
    def sum(self):
        to_return = self.dtype(0)

        for i in range(self.height):
            for j in range(self.width):
                to_return += self[i][j]

        return to_return

    @property
    def product(self):
        to_return = self.dtype(1)

        for i in range(self.height):
            for j in range(self.width):
                to_return *= self[i][j]

        return to_return

    @property
    def determinant(self):
        try:
            lu_matrix = self.lu()
        except NotSolvable:
            try:
                lu_matrix, p_matrix = self.lup()

                swap_count = count_swaps_in_row_order(
                    Matrix.permutation_matrix_to_row_order(p_matrix)
                )
                p_determinant = 1 if swap_count % 2 == 0 else -1
                lu_matrix *= p_determinant
            except NotSolvable:
                # If even LUP fails, the matrix is singular; if a matrix
                # is singular, it's determinant is 0.
                return 0

        lu_determinant = lu_matrix.diagonal.product

        if abs(lu_determinant) < self.epsilon:
            # Probably a singular matrix that didn't raise an
            # exception.
            return 0

        if isinstance(lu_determinant, float) and lu_determinant.is_integer():
            lu_determinant = int(lu_determinant)

        return lu_determinant

    @property
    def inverse(self):
        to_return = copy.deepcopy(self)
        to_return.invert()

        return to_return

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
                "Expected argument array to be a list or a tuple, instead it is "
                f"{type(array)}."
            )

        if len(array) == 0:
            raise ValueError("Expected argument array to have at least one element.")

        for i in range(len(array)):
            if isinstance(array[i], (list, tuple)):
                for j in range(len(array[i])):
                    if isinstance(array[i][j], (list, tuple)):
                        raise TypeError(
                            "Expected argument array to at most be of depth 2, yet it "
                            "is deeper."
                        )
            else:
                array[i] = [array[i]]

        height = len(array)
        width = max([0] + [len(x) for x in array])

        if min(height, width) == 0:
            raise ValueError("Expected argument array to be non-empty.")

        dtype = int

        for i in range(len(array)):
            if dtype == float:
                break

            for j in range(len(array[i])):
                if isinstance(array[i][j], float):
                    dtype = float
                    break

        to_return = Matrix(height=height, width=width, dtype=dtype)

        for i in range(len(to_return)):
            for j in range(len(to_return[i])):
                value_to_insert = to_return.dtype(
                    array[i][j] if (i < len(array) and j < len(array[i])) else 0.0
                )

                to_return[i][j] = value_to_insert

        return to_return

    @staticmethod
    def from_text(text: str):
        if text is None:
            raise TypeError("Argument text mustn't be None!")

        if not isinstance(text, str):
            raise TypeError(
                f"Expected argument text to be a str, instead it is {type(text)}"
            )

        data = list()

        for line in [x for x in text.splitlines() if (x is not None and len(x) != 0)]:
            current_row = list()

            for element in [
                x
                for x in WHITESPACE_REGEX.split(line)
                if (x is not None and len(x) != 0)
            ]:
                element = float(element)

                if element.is_integer():
                    element = int(element)

                current_row.append(element)

            data.append(current_row)

        return Matrix.from_array(data)

    @staticmethod
    def from_file(file_path: Union[Path, str]):
        if file_path is None:
            raise TypeError("Argument file_path mustn't be None!")

        if not isinstance(file_path, (Path, str)):
            raise TypeError(
                "Expected argument file_path to be a Path or str, instead it is "
                f"{type(file_path)}."
            )

        with open(file_path, mode="r", encoding="utf8") as file:
            return Matrix.from_text(file.read())

    # endregion

    # region Submatrix Methods
    @property
    def diagonal(self) -> "Matrix":
        to_return = Matrix(
            height=1, width=min(self.height, self.width), dtype=self.dtype
        )

        for j in range(len(to_return[0])):
            to_return[0][j] = self[j][j]

        return to_return

    @property
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
                f"Expected argument index to be an int, instead it is {type(index)}."
            )

        if not 0 <= index < self.height:
            raise ValueError(
                f"Expected argument index to be in range [0, {self.height}>, instead "
                f"it is {index}."
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
                f"Expected argument index to be an int, instead it is {type(index)}."
            )

        if not 0 <= index < self.width:
            raise ValueError(
                f"Expected argument index to be in range [0, {self.width}>, instead it "
                f"it is {index}."
            )

        to_return = Matrix(height=self.height, width=1, dtype=self.dtype)

        for i in range(to_return.height):
            to_return[i][0] = self[i][index]

        return to_return

    # endregion

    # region Helper Methods
    def save(self, file_path: Union[Path, str]):
        if file_path is None:
            raise TypeError("Argument file_path mustn't be None!")

        if not isinstance(file_path, (Path, str)):
            raise TypeError(
                "Expected argument file_path to be a Path or a str, instead it is "
                f"{type(file_path)}."
            )

        with open(file_path, mode="w+", encoding="utf8") as file:
            file.write("\n".join([" ".join([str(x) for x in row]) for row in self]))

    def fill(self, value: Union[float, int]):
        if value is None:
            raise TypeError("Argument value mustn't be None!")

        if not (isinstance(value, float) or isinstance(value, int)):
            raise TypeError(
                "Expected argument value to be a float or int, instead it is "
                f"{type(value)}."
            )

        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] = self.dtype(value)

    @staticmethod
    def row_order_to_permutation_matrix(
        row_order: Union[List[int], Tuple[int]]
    ) -> "Matrix":
        to_return = Matrix.zeros(len(row_order), len(row_order), int)

        for i, row_index in enumerate(row_order):
            to_return[i][row_index] = 1

        return to_return

    @staticmethod
    def permutation_matrix_to_row_order(permutation_matrix: "Matrix") -> List[int]:
        permutation_matrix = copy.deepcopy(permutation_matrix)

        if permutation_matrix.dtype != int:
            permutation_matrix.int()

        to_return = [None] * min(permutation_matrix.height, permutation_matrix.width)

        for i in range(permutation_matrix.height):
            for j in range(permutation_matrix.width):
                if permutation_matrix[i][j] == 1:
                    to_return[i] = j
                    break

        return to_return

    @staticmethod
    def split_lu_matrix(lu_matrix: "Matrix"):
        if lu_matrix is None:
            raise TypeError("Argument lu_matrix mustn't be None!")

        if not isinstance(lu_matrix, Matrix):
            raise TypeError(
                "Expected argument lu_matrix to be a Matrix, instead it is "
                f"{type(lu_matrix)}."
            )

        if lu_matrix.height != lu_matrix.width:
            raise MatrixShapeMismatchException(
                "Failed to split LU matrix: it's defined only for square matrices, but "
                f"a matrix of shape {lu_matrix.shape} is not square."
            )

        l_matrix = Matrix.eye(
            height=lu_matrix.height, width=lu_matrix.width, dtype=lu_matrix.dtype
        )
        u_matrix = Matrix.zeros(
            height=lu_matrix.height, width=lu_matrix.width, dtype=lu_matrix.dtype
        )

        for i in range(lu_matrix.height):
            for j in range(lu_matrix.width):
                if i > j:
                    l_matrix[i][j] = lu_matrix[i][j]
                else:
                    u_matrix[i][j] = lu_matrix[i][j]

        return l_matrix, u_matrix

    # endregion

    # region Matrix Casting Methods
    def int(self):
        self._config.dtype = int

        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j] + 0.5)

        return self

    def float(self):
        self._config.dtype = float

        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j])

                    if abs(self._data[i][j]) < self.epsilon:
                        self._data[i][j] = 0.0

        return self

    # endregion

    # region Arithmetic Methods
    def abs(self):
        for i in range(self.height):
            for j in range(self.width):
                self[i][j] = abs(self[i][j])

    def add(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix addition: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] += other[i][j]

    def floordiv(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do Hadamard whole division: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] //= other[i][j]

    def invert(self):
        if self.height != self.width:
            raise MatrixShapeMismatchException(
                "Failed to invert matrix: it's defined only for square matrices, but a "
                f"matrix of shape {self.shape} is not square."
            )

        try:
            lu_matrix, p_matrix = self.lup()
        except NotSolvable:
            raise MatrixIsSingular(self)

        temp_matrix = Matrix.zeros(
            height=self.height, width=self.width, dtype=self.dtype
        )
        identity_matrix = Matrix.eye(
            height=self.height, width=self.width, dtype=self.dtype
        )
        identity_matrix = p_matrix @ identity_matrix

        for j in range(self.width):
            current_column = lu_matrix.backward_substitute(
                lu_matrix.forward_substitute(identity_matrix.row(j))
            )

            for i in range(self.height):
                if temp_matrix.dtype == int and current_column.dtype == float:
                    temp_matrix.float()

                temp_matrix[i][j] = current_column[0][i]

        self._config = copy.deepcopy(temp_matrix._config)
        self._data = copy.deepcopy(temp_matrix._data)

    def matmul(self, other: "Matrix"):
        if other is None:
            raise TypeError("Argument other mustn't be None!")

        if not isinstance(other, Matrix):
            raise TypeError(
                f"Expected argument other to be a Matrix, instead it is {type(other)}."
            )

        if self.width != other.height:
            raise MatrixNotBroadcastableException(
                "Failed to do matrix multiplication: the height of the right matrix "
                f"({other.height}) should be equal to the width of the left matrix "
                f"({self.width})"
            )

        if other.dtype == float:
            self.float()

        to_return = Matrix.zeros(
            height=self.height, width=other.width, dtype=self.dtype
        )

        for i in range(to_return.height):
            for j in range(to_return.width):
                current_value = 0

                for k in range(self.width):
                    current_value += self[i][k] * other[k][j]

                if to_return.dtype == int and isinstance(current_value, float):
                    to_return.float()

                to_return[i][j] = to_return.dtype(current_value)

        self._config = copy.deepcopy(to_return._config)
        self._data = copy.deepcopy(to_return._data)

    def mod(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix modulus: shapes {self.shape} & {other.shape} "
                    "do not match"
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] %= other[i][j]

    def mul(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do Hadamard multiplication: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] *= other[i][j]

    def negate(self):
        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j] = -self[i][j]

    def pow(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do Hadamard power: shapes {self.shape} & {other.shape} "
                    "do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] **= other[i][j]

        if self.dtype == int and other.dtype == int:
            self.int()

    def sub(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix subtraction: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] -= other[i][j]

    def transpose(self):
        _new_data = list()

        for j in range(self.width):
            current_row = list()

            for i in range(self.height):
                current_row.append(self._data[i][j])

            _new_data.append(current_row)

        self._config.shape = self.width, self.height
        self._data = _new_data

    def transposed(self):
        to_return = copy.deepcopy(self)
        to_return.transpose()

        return to_return

    def truediv(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do Hadamard division: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] /= other[i][j]

    # endregion

    # region Comparison Methods
    def __ordinary_equals(self, first, second):
        return first == second

    def __float_equals(self, first, second):
        return abs(first - second) < self.epsilon

    def equals(self, other: "Matrix"):
        if other is None or not (
            isinstance(other, Matrix) and self.shape == other.shape
        ):
            return False

        comparison_function = self.__ordinary_equals

        if self.dtype == float or other.dtype == float:
            comparison_function = self.__float_equals

        for i in range(self.height):
            for j in range(self.width):
                if not comparison_function(self[i][j], other[i][j]):
                    return False

        return True

    def ge(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix comparison: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        for i in range(self.height):
            for j in range(self.width):
                if self[i][j] < other[i][j]:
                    return False

        return True

    def gt(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix comparison: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        for i in range(self.height):
            for j in range(self.width):
                if self[i][j] <= other[i][j]:
                    return False

        return True

    def le(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix comparison: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        for i in range(self.height):
            for j in range(self.width):
                if self[i][j] > other[i][j]:
                    return False

        return True

    def lt(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    f"Failed to do matrix comparison: shapes {self.shape} & "
                    f"{other.shape} do not match."
                )
        elif isinstance(other, (float, int)):
            other = Matrix.full(self.height, self.width, fill_value=other)
        else:
            raise TypeError(
                "Expected argument other to be a float, int, or Matrix, instead it is "
                f"{type(other)}."
            )

        for i in range(self.height):
            for j in range(self.width):
                if self[i][j] >= other[i][j]:
                    return False

        return True

    # endregion

    # region Problem Solving Methods
    def forward_substitute(self, row: "Matrix"):
        if row is None:
            raise TypeError("Argument row mustn't be None!")

        if not isinstance(row, Matrix):
            raise TypeError(
                f"Expected argument row to be a Matrix, instead it is {type(row)}."
            )

        if row.width < self.height:
            raise ValueError(
                f"Expected argument row to have a width of at least {self.height}, "
                f"instead it has width of {row.width}."
            )

        row = copy.deepcopy(row)

        for i in range(self.height - 1):
            for j in range(i + 1, self.height):
                row[0][j] -= self[j][i] * row[0][i]

        return row

    def backward_substitute(self, row: "Matrix", salvage_zero_divison: bool = False):
        if row is None:
            raise TypeError("Argument row mustn't be None!")

        if not isinstance(row, Matrix):
            raise TypeError(
                f"Expected argument row to be a Matrix, instead it is {type(row)}."
            )

        if row.width < self.height:
            raise ValueError(
                f"Expected argument row to have a width of at least {self.height}, "
                f"instead it has width of {row.width}."
            )

        row = copy.deepcopy(row)

        for i in reversed(range(self.height)):
            pivot = copy.deepcopy(self[i][i])

            if abs(pivot) < self.epsilon:
                if salvage_zero_divison:
                    pivot = self.epsilon * sign(pivot, zero_is_positive=True)
                else:
                    raise NotSolvable(
                        "Tried to divide by (approximately) zero in method "
                        f"Matrix.backward_substitute: Matrix[{i}][{i}] is the culprit "
                        f"(in {self})."
                    )

            if row.dtype == int and (
                isinstance(pivot, float) or row[0][i] % pivot != 0
            ):
                row.float()

            row[0][i] /= pivot

            for j in range(i):
                row[0][j] -= self[j][i] * row[0][i]

        return row

    def lu(self, salvage_zero_pivot: bool = False):
        if self.height != self.width:
            raise MatrixShapeMismatchException(
                "Failed to do LU decomposition: it's defined only for square matrices, "
                f"but a matrix of shape {self.shape} is not square."
            )

        to_return = copy.deepcopy(self)

        for i in range(to_return.height - 1):
            for j in range(i + 1, to_return.height):
                pivot = copy.deepcopy(to_return[i][i])

                if abs(pivot) < self.epsilon:
                    if salvage_zero_pivot:
                        pivot = self.epsilon * sign(pivot, zero_is_positive=True)
                    else:
                        raise NotSolvable(
                            "Tried to divide by (approximately) zero in method "
                            f"Matrix.lu: Matrix[{i}][{i}] is the culprit (in "
                            f"{to_return})."
                        )

                if to_return.dtype == int and (
                    isinstance(pivot, float) or to_return[j][i] % pivot != 0
                ):
                    to_return.float()

                to_return[j][i] /= pivot

                for k in range(i + 1, to_return.height):
                    to_return[j][k] -= to_return[j][i] * to_return[i][k]

        if abs(to_return[-1][-1]) < self.epsilon:
            raise NotSolvable(
                "Encountered a zero pivot in method Matrix.lu: "
                f"Matrix[{to_return.height - 1}][{to_return.height - 1}] is the "
                f"culprit (in {to_return})."
            )

        return to_return

    def lup(self, salvage_zero_pivot: bool = False):
        if self.height != self.width:
            raise MatrixShapeMismatchException(
                "Failed to do LUP decomposition: it's defined only for square "
                f"matrices, but a matrix of shape {self.shape} is not square."
            )

        row_order = list(range(self.height))
        to_return = copy.deepcopy(self)

        for i in range(to_return.height - 1):
            current_column = i

            for j in range(i + 1, to_return.height):
                if abs(to_return[row_order[j]][i]) > abs(
                    to_return[row_order[current_column]][i]
                ):
                    current_column = j

            if current_column != i:
                row_order[i], row_order[current_column] = (
                    row_order[current_column],
                    row_order[i],
                )

            for j in range(i + 1, to_return.height):
                pivot = copy.deepcopy(to_return[row_order[i]][i])

                if abs(pivot) < self.epsilon:
                    if salvage_zero_pivot:
                        pivot = self.epsilon * sign(pivot, zero_is_positive=True)
                    else:
                        raise NotSolvable(
                            "Tried to divide by (approximately) zero in method "
                            f"Matrix.lup: Matrix[{i}][{i}] is the culprit (in "
                            f"{to_return})."
                        )

                if to_return.dtype == int and (
                    isinstance(pivot, float) or to_return[row_order[j]][i] % pivot != 0
                ):
                    to_return.float()

                to_return[row_order[j]][i] /= pivot

                for k in range(i + 1, to_return.height):
                    to_return[row_order[j]][k] -= (
                        to_return[row_order[j]][i] * to_return[row_order[i]][k]
                    )

        permutation_matrix = self.row_order_to_permutation_matrix(row_order)
        to_return = permutation_matrix @ to_return

        if abs(to_return[-1][-1]) < self.epsilon:
            raise NotSolvable(
                "Encountered a zero pivot in method Matrix.lup: "
                f"Matrix[{to_return.height - 1}][{to_return.height - 1}] is the "
                f"culprit (in {to_return})."
            )

        return to_return, permutation_matrix

    def solve(self, values: "Matrix"):
        if values is None:
            raise TypeError("Argument values mustn't be None!")

        if not isinstance(values, Matrix):
            raise TypeError(
                "Expected argument values to be a Matrix, instead it is "
                f"{type(values)}."
            )

        values = copy.deepcopy(values)

        if values.width == 1 and values.height != 1:
            values.transpose()

        if self.height != values.width:
            raise MatrixShapeMismatchException(
                "Failed to solve matrix: argument values should be a row or column "
                f"matrix with {self.height} elements, instead it has {values.width}."
            )

        return self.backward_substitute(self.forward_substitute(values))

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
            " ".join(
                [" " * (max_str_len - len(element)) + f"{element}" for element in row]
            )
            for row in string_matrix
        ]
        translated_string_rows = [f"{left_space}{tab}[{x}]" for x in string_rows]
        string_block = f"{newline_char}".join(translated_string_rows)

        to_return = (
            f"{left_space}[{newline_char}{string_block}{newline_char}{left_space}]"
        )

        return to_return

    def latex(self, environment: str = "plain", decimal_precision: int = 3):
        if environment is None:
            raise TypeError("Argument environment mustn't be None!")

        if decimal_precision is None:
            raise TypeError("Argument decimal_precision mustn't be None!")

        if not isinstance(environment, str):
            raise TypeError(
                "Expected argument environment to be a string, instead it is "
                f"{type(environment)}."
            )

        if not isinstance(decimal_precision, int):
            raise TypeError(
                "Expected argument decimal_precision to be an int, instead it is "
                f"{type(decimal_precision)}."
            )

        matrix_environments = sorted(
            self.__matrix_environment_to_latex_environment.keys()
        )

        if environment not in matrix_environments:
            raise ValueError(
                "Expected argument environment to be "
                f"{', '.join(matrix_environments[:-1])} or {matrix_environments[-1]}, "
                f"instead it is {environment}."
            )

        if decimal_precision < 0:
            raise ValueError(
                "Expected argument decimal_precision to be greater or equal to 0, "
                f"instead it is {type(decimal_precision)}."
            )

        environment = self.__matrix_environment_to_latex_environment[environment]

        to_return = "\\begin{" + environment + "}\n"

        for i in range(self.height):
            to_return += "\t"
            row = copy.deepcopy(self[i])

            for j in range(len(row)):
                if isinstance(row[j], float):
                    if row[j].is_integer():
                        row[j] = f"{int(row[j]):d}"
                    else:
                        row[j] = f"{row[j]:.{decimal_precision}f}"
                elif isinstance(row[j], int):
                    row[j] = f"{row[j]:d}"
                else:
                    row[j] = str(row[j])

            to_return += " & ".join(row)
            to_return += r" \\"
            to_return += "\n"

        to_return += "\\end{" + environment + "}"

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
    def __abs__(self):
        to_return = copy.deepcopy(self)
        to_return.abs()

        return to_return

    def __add__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.add(other)

        return to_return

    def __float__(self):
        if self.height == 1 and self.width == 1:
            return float(self[0][0])
        else:
            raise TypeError(
                "Cannot implicitly convert a multielement Matrix to a float."
            )

    def __floordiv__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.floordiv(other)

        return to_return

    def __int__(self):
        if self.height == 1 and self.width == 1:
            return int(0.5 + self[0][0])
        else:
            raise TypeError(
                "Cannot implicitly convert a multielement Matrix to an int."
            )

    def __invert__(self):
        return self.inverted()

    def __matmul__(self, other: "Matrix"):
        to_return = copy.deepcopy(self)
        to_return.matmul(other)

        return to_return

    def __mod__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.mod(other)

        return to_return

    def __mul__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.mul(other)

        return to_return

    def __neg__(self):
        to_return = copy.deepcopy(self)
        to_return.negate()

        return to_return

    def __pow__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.pow(other)

        return to_return

    def __sub__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.sub(other)

        return to_return

    def __truediv__(self, other: Union[float, int, "Matrix"]):
        to_return = copy.deepcopy(self)
        to_return.truediv(other)

        return to_return

    # endregion

    # region Extended Mathematical Dunder Methods
    def __iadd__(self, other: Union[float, int, "Matrix"]):
        self.add(other)

        return self

    def __ifloordiv__(self, other: Union[float, int, "Matrix"]):
        self.floordiv(other)

        return self

    def __imatmul__(self, other: "Matrix"):
        self.matmul(other)

        return self

    def __imod__(self, other: Union[float, int, "Matrix"]):
        self.mod(other)

        return self

    def __imul__(self, other: Union[float, int, "Matrix"]):
        self.mul(other)

        return self

    def __ipow__(self, other: Union[float, int, "Matrix"]):
        self.pow(other)

        return self

    def __isub__(self, other: Union[float, int, "Matrix"]):
        self.sub(other)

        return self

    def __itruediv__(self, other: Union[float, int, "Matrix"]):
        self.truediv(other)

        return self

    # endregion

    # region Comparison Dunder Methods
    def __eq__(self, other: Union[float, int, "Matrix"]):
        return self.equals(other)

    def __ge__(self, other: Union[float, int, "Matrix"]):
        return self.ge(other)

    def __gt__(self, other: Union[float, int, "Matrix"]):
        return self.gt(other)

    def __le__(self, other: Union[float, int, "Matrix"]):
        return self.le(other)

    def __lt__(self, other: Union[float, int, "Matrix"]):
        return self.lt(other)

    def __ne__(self, other: Union[float, int, "Matrix"]):
        return not self.equals(other)

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

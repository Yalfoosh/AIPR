import copy
from textwrap import dedent
from typing import Callable, List, Optional, Tuple, Union

from .exceptions import (
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
                dedent(
                    f"""\
                    Expected argument dtype to be a callable, instead it is \
                    {type(dtype)}.\
                    """
                )
            )

        if not isinstance(epsilon, (float, int)):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument epsilon to be a float or int, instead it is \
                    {type(epsilon)}.\
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

        if epsilon < 0:
            raise ValueError(
                dedent(
                    f"""\
                    Expected argument epsilon to be 0 or greater, instead it is \
                    {epsilon}.\
                    """
                )
            )

        self.shape = (height, width)
        self.dtype = dtype
        self.epsilon = float(epsilon)


class Matrix:
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
                dedent(
                    f"""\
                    Expected argument other to be a float or int, instead it is \
                    {type(value)}.\
                    """
                )
            )

        self._config.epsilon = float(value)

    @property
    def EPS(self):
        return self.epsilon

    @property
    def T(self):
        return self.transposed()

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

        height = len(array)
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

    # region Helper Methods
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

    @staticmethod
    def split_lu_matrix(self, lu_matrix: "Matrix"):
        if lu_matrix is None:
            raise TypeError("Argument lu_matrix mustn't be None!")

        if not isinstance(lu_matrix, Matrix):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument lu_matrix to be a Matrix, instead it is \
                    {type(lu_matrix)}.\
                    """
                )
            )

        if lu_matrix.height != lu_matrix.width:
            raise MatrixShapeMismatchException(
                dedent(
                    f"""\
                    Failed to split LU matrix: it's defined only for square matrices, \
                    but a matrix of shape {lu_matrix.shape} is not square.\
                    """
                )
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

    def float(self):
        self._config.dtype = float

        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                if self._data[i][j] is not None:
                    self._data[i][j] = self.dtype(self._data[i][j])

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
                    dedent(
                        f"""\
                        Failed to do matrix addition: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] += other[i][j]

    def determinant(self):
        lu_matrix, p_matrix = self.lup()
        l_matrix, u_matrix = Matrix.split_lu_matrix(lu_matrix)

        swap_count = count_swaps_in_row_order(
            Matrix.permutation_matrix_to_row_order(p_matrix)
        )
        p_determinant = 1 if swap_count % 2 == 0 else -1

        l_diagonal = l_matrix.diagonal()
        u_diagonal = u_matrix.diagonal()

        return p_determinant * (l_diagonal @ u_diagonal.T)[0][0]

    def floordiv(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    dedent(
                        f"""\
                        Failed to do Hadamard whole division: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
            )

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] //= other[i][j]

    def invert(self):
        if self.height != self.width:
            raise MatrixShapeMismatchException(
                dedent(
                    f"""\
                    Failed to invert matrix: it's defined only for square matrices, \
                    but a matrix of shape {self.shape} is not square.\
                    """
                )
            )

        lu_matrix, p_matrix = self.lup()
        to_return = Matrix.zeros(height=self.height, width=self.width, dtype=self.dtype)
        identity_matrix = Matrix.eye(
            height=self.height, width=self.width, dtype=self.dtype
        )

        for j in range(self.width):
            current_column = lu_matrix.backward_substitute(
                lu_matrix.forward_substitute(identity_matrix.row(j))
            )

            for i in range(self.height):
                to_return[i][j] = current_column[0][i]

        return p @ to_return

    def inverse(self):
        to_return = copy.deepcopy(self)
        to_return.invert()

        return to_return

    def matmul(self, other: "Matrix"):
        if other is None:
            raise TypeError("Argument other mustn't be None!")

        if not isinstance(other, Matrix):
            raise TypeError(
                dedent(
                    f"""\
                    Expected argument other to be a Matrix, instead it is \
                    {type(other)}.\
                    """
                )
            )

        if self.width != other.height:
            raise MatrixNotBroadcastableException(
                dedent(
                    f"""\
                    Failed to do matrix multiplication: the height of the right matrix \
                    ({other.height}) should be equal to the width of the left matrix \
                    ({self.width}).\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        to_return = Matrix.zeros(
            height=self.height, width=other.width, dtype=self.dtype
        )

        for i in range(to_return.height):
            for j in range(to_return.width):
                current_value = 0.0

                for k in range(self.width):
                    current_value += self[i][k] * other[k][j]

                to_return[i][j] = to_return.dtype(current_value)

        return to_return

    def mod(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    dedent(
                        f"""\
                        Failed to do matrix modulus: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    dedent(
                        f"""\
                        Failed to do Hadamard power: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
            )

        if other.dtype == float:
            self.float()

        for i in range(self.height):
            for j in range(self.width):
                self[i][j] **= other[i][j]

        if self.dtype == int and other.dtype == int:
            # Account for rounding errors
            self += 0.25
            self.int()

    def sub(self, other: Union[float, int, "Matrix"]):
        if isinstance(other, Matrix):
            if not self.shape == other.shape:
                raise MatrixShapeMismatchException(
                    dedent(
                        f"""\
                        Failed to do matrix subtraction: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    dedent(
                        f"""\
                        Failed to do matrix comparison: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    dedent(
                        f"""\
                        Failed to do matrix comparison: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    dedent(
                        f"""\
                        Failed to do matrix comparison: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                    dedent(
                        f"""\
                        Failed to do matrix comparison: shapes {self.shape} & \
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
                    Expected argument other to be a float, int, or Matrix, instead it \
                    is {type(other)}.\
                    """
                )
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
                dedent(
                    f"""\
                    Expected argument row to be a Matrix, instead it is {type(row)}.\
                    """
                )
            )

        if row.width < self.height:
            raise ValueError(
                dedent(
                    f"""\
                    Expected argument row to have a width of at least {self.height}, \
                    instead it has width of {row.width}.\
                    """
                )
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
                dedent(
                    f"""\
                    Expected argument row to be a Matrix, instead it is {type(row)}.\
                    """
                )
            )

        if row.width < self.height:
            raise ValueError(
                dedent(
                    f"""\
                    Expected argument row to have a width of at least {self.height}, \
                    instead it has width of {row.width}.\
                    """
                )
            )

        row = copy.deepcopy(row)

        for i in range(self.height)[::-1]:
            if abs(self[i][i]) < self.epsilon:
                if salvage_zero_divison:
                    self[i][i] = self.epsilon * sign(self[i][i], zero_is_positive=True)
                else:
                    raise NotSolvable(
                        dedent(
                            f"""\
                            Tried to divide by (approximately) zero in method \
                            Matrix.backward_substitute: Matrix[{i}][{i}] is the \
                            culprit (in {self}).\
                            """
                        )
                    )

                row[0][i] /= self[i][i]

                for j in range(i):
                    row[0][j] -= self[j][i] * row[0][i]

        return row

    def lu(self, salvage_zero_pivot: bool = False):
        if self.height != self.width:
            raise MatrixShapeMismatchException(
                dedent(
                    f"""\
                    Failed to do LU decomposition: it's defined only for square \
                    matrices, but a matrix of shape {self.shape} is not square.\
                    """
                )
            )

        to_return = copy.deepcopy(self)

        for i in range(to_return.height - 1):
            for j in range(i + 1, to_return.height):
                pivot = to_return[i][i]

                if abs(pivot) < self.epsilon:
                    if salvage_zero_pivot:
                        pivot = self.epsilon * sign(pivot, zero_is_positive=True)
                    else:
                        raise NotSolvable(
                            dedent(
                                f"""\
                                Tried to divide by (approximately) zero in method \
                                Matrix.lu: Matrix[{i}][{i}] is the culprit (in \
                                {to_return}).\
                                """
                            )
                        )

                to_return[j][i] /= pivot

                for k in range(i + 1, to_return.height):
                    to_return[j][k] -= to_return[j][i] * to_return[i][k]

        return to_return

    def lup(self, salvage_zero_pivot: bool = False):
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

            for j in range(i + i, to_return.height):
                pivot = to_return[row_order[i]][i]

                if abs(pivot) < self.epsilon:
                    if salvage_zero_pivot:
                        pivot = self.epsilon * sign(pivot, zero_is_positive=True)
                    else:
                        raise NotSolvable(
                            dedent(
                                f"""\
                                Tried to divide by (approximately) zero in method \
                                Matrix.lup: Matrix[{i}][{i}] is the culprit (in \
                                {to_return}).\
                                """
                            )
                        )

                to_return[row_order[j]][i] /= pivot

                for k in range(i + 1, to_return.height):
                    to_return[row_order[j]][k] -= (
                        to_return[row_order[j]][i] * to_return[row_order[i]][k]
                    )

        if abs(to_return[-1][-1]) < self.epsilon:
            raise NotSolvable(
                dedent(
                    f"""\
                    Encountered a zero pivot in method Matrix.lup: \
                    Matrix[{to_return.height - 1}][{to_return.height - 1}] is the \
                    culprit (in {to_return}).\
                    """
                )
            )

        permutation_matrix = self.row_order_to_permutation_matrix(row_order)

        return permutation_matrix @ to_return, permutation_matrix

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
        return self.matmul(other)

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
        result = self.matmul(other)

        self._data = result._data
        self._config = copy.deepcopy(result._config)

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

class MatrixShapeMismatchException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MatrixNotBroadcastableException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotSolvable(Exception):
    def __init__(self, message):
        super().__init__(message)


class MatrixIsSingular(Exception):
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = " "
        else:
            matrix = f" {matrix} "

        super().__init__(f"Matrix{matrix}is singular.")

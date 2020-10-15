class MatrixShapeMismatchException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MatrixNotBroadcastableException(Exception):
    def __init__(self, message):
        super().__init__(message)
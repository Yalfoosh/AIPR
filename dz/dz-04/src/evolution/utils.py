import numpy as np


bin2dec_vector = np.vectorize(
    lambda x: x.dot(1 << np.arange(x.size)[::-1]), signature="(n)->()"
)

dec2bin_vector = np.vectorize(
    lambda x, y: np.array(list(np.binary_repr(x).zfill(y))).astype(np.int32),
    signature="(),()->(n)",
)

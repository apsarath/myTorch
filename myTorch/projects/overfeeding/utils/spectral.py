import numpy as np
import numpy.linalg as linalg
from collections import namedtuple

spectral_properties = namedtuple("spectral_properties", [
    "condition_number",
    "spectral_norm",
    "s",
    "eig_vals",
    "spectral_radius"
])


def compute_spectral_properties(A):
    '''Method to compute the different spectral properties for a given matrix.
    Following properties are computed:
        * condition number - ratio of smallest to largest singular value of  SVD of a matrix
        * spectral norm - largest singular value of A or square root of largest eigen value
        corresponding to (conjugate_transpose(A))A
        * Spectral radius - largest eign value
        * gain of a matrix - spectral norm
        *
    '''
    condition_number = linalg.cond(A)
    spectral_norm = linalg.norm(A, 2)
    u, s, v = linalg.svd(A)
    eig_vals = linalg.eigvals(np.matmul(A.conj().T, A))
    spectral_radius = eig_vals[0]
    return spectral_properties(condition_number, spectral_norm, s, eig_vals, spectral_radius)
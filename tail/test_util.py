'''This is not used in actual pipeline
but simulation for unit test only
'''

import numpy as np
from numba import jit
from scipy import stats


def sim_guassian_2d_sq(n, N):
    '''simulate a guassian fluctuation
    with ``N`` masses.
    Return 2d-array of shape (n, n)
    '''
    return (
        stats.gaussian_kde(np.random.randint(0, n, size=(2, N)))
        .evaluate(
            np.mgrid[0:n, 0:n]
            .reshape(2, n * n)
        ).reshape((n, n))
    )


@jit(nopython=True, nogil=True)
def sim_matrix(n):
    '''simulate a random matrix,
    diagonal from [10, 11),
    everywhere else from [0, 1).
    e.g. used for testing a crude mode-coupling matrix.
    '''

    M = np.random.rand(n, n)
    np.fill_diagonal(M, 10.)
    return M

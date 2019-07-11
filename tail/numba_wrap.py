import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def fftfreq(n, d=1.0):
    '''Numba wrap of numpy.fft.fftfreq
    source modified from
    https://github.com/numpy/numpy/blob/250861059b106371cb232456eeccd6d9e97d8f00/numpy/fft/helper.py#L168-L177
    '''
    N = (n - 1) // 2 + 1

    results = np.empty(n)

    results[:N] = np.arange(0, N)
    results[N:] = np.arange(-(n // 2), 0)

    results /= n * d

    return results

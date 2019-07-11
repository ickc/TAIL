import numpy as np
from numba import jit, prange, stencil
from scipy import signal
from scipy.fftpack import next_fast_len

from dautil.util import zero_padding
from tail.util import normalize_max

# for zero-padding
EXTEND_RATIO = 2
# for _smooth_window
M = 32
STD = 3.3013338110314874
# for tappering, 30 corresponds to 1 degree in 2 arcmin. pixel
WIDTH = 30


@jit(nopython=True, nogil=True)
def _planck_taper(x):
    '''Planck tapering mapping x in (0, 1) to y in (0, 1)
    '''
    return 1. / (1. + np.exp((1. - 2. * x) / (x * (1. - x))))


@jit(nopython=True, nogil=True, parallel=True)
def _get_boundary(mask):

    @stencil
    def _get_boundary_stencil(mask):
        return (mask[0, 0] & ~(mask[-1, 0] & mask[1, 0] & mask[0, -1] & mask[0, 1]))

    return _get_boundary_stencil(mask)


@jit(nopython=True, nogil=True, parallel=True)
def _get_taper(mask, width):
    '''``mask``: 2d-boolean array

    ``width``: width of the taper, ``int``.

    return a mask tapered by a width of ``width`` with ``taper_func``.

    Note: this function is a bit slow, for array of width 1700,
    on a 10-core machine tooks ~1s, because it exhaustively
    goes through all pixels to search for minimum. A better
    algorithm will be using a boundary detection algorithm
    which only eat into the boundary ``m`` no. of times.
    '''
    idxs, idys = np.nonzero(_get_boundary(mask))
    width_sq = width * width

    m, n = mask.shape
    l = idxs.size

    result = np.zeros((m, n), dtype=np.float64)

    for i in prange(m):
        for j in range(n):
            if mask[i, j]:
                # get distance squared from the boundary
                d_sq = np.inf
                for k in range(l):
                    dx = i - idxs[k]
                    dy = j - idys[k]
                    temp = dx * dx + dy * dy
                    if temp < d_sq:
                        d_sq = temp
                # taper according to distance comparing to width
                result[i, j] = 1. if d_sq >= width_sq else 0. if d_sq == 0. else _planck_taper(np.sqrt(d_sq) / width)
    return result


def zero_pad_ratio(mask, extend_ratio=EXTEND_RATIO):
    '''Zero-padding optimized for fftw-size
    '''
    n_x = next_fast_len(mask.shape[0] * extend_ratio)
    return zero_padding(mask, (n_x, n_x))


def _smooth_window(mask, M=M, std=STD, mode='full'):
    '''convolve with a 2d gaussian kernel with parameter ``M`` and ``std``
    these default value is chosen to closely match that defined in AnalysisBackend,
    which used a hamming window with 16 pixel wide instead.
    The convolve mode is "full" instead of "same", because it is going
    to be zero-padded later anyway, and "full" is more accurate (in case some non-zero
    mask is near the boundary.)
    16 pixel wide corresponds to 0.53 degree scale, used in IPP/SPP.
    '''
    @jit(nopython=True, nogil=True)
    def ker_1d_to_2d(ker1):
        '''1d kernel to 2d kernel, normalized
        '''
        ker2 = np.outer(ker1, ker1)
        ker2 /= ker2.sum()
        return ker2

    ker1 = signal.windows.gaussian(M, std)
    ker2 = ker_1d_to_2d(ker1)
    del ker1
    result = signal.fftconvolve(mask, ker2, mode=mode)
    return result


def standard_mask(mask, extend_ratio=EXTEND_RATIO, smooth=True, taper=True):
    '''``mask`` modified in-place
    '''
    # _smooth_window might change mask.shape if mode == 'full'
    n_x = next_fast_len(mask.shape[0] * EXTEND_RATIO)

    # tapering need to be done before smoothing, because the smoothing will change the boundary
    if taper:
        mask *= _get_taper(mask > 0., WIDTH)

    if smooth:
        if taper:
            # if taperring, then smoothing beyond the boundary doesn't make sense
            mask_boolean = mask > 0.
            temp = np.zeros_like(mask)
            temp[mask_boolean] = _smooth_window(mask, mode='same')[mask_boolean]
            mask = temp
            del mask_boolean, temp
        else:
            mask = _smooth_window(mask, mode='full')

    # this normalization doesn't matter
    # mask is used in 2 places, mode-coupling, and pseudo-spectra
    # in mode-coupling, norm_fft is used
    # in pseudo-spectra as a weight, the absolute scale doesn't matter
    # normalize_max(mask)

    return zero_padding(mask, (n_x, n_x))

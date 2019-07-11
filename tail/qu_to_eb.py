'''This code treat columns as x-coordinate
and rows as y-coordinate
'''

import numpy as np
from numba import jit, prange, stencil
from scipy.fftpack import fft2

from tail.numba_wrap import fftfreq


@jit(nopython=True, nogil=True, parallel=True)
def _get_l_2d(n, pixel_size):
    '''given dimension ``n``, pixel width ``pixel_size``,
    return l in 2d, norm and arg (angle)
    '''
    freq = fftfreq(n, pixel_size)

    pi_2 = np.pi * 2.
    result = np.empty((n, n, 2))
    for i in prange(n):
        freq_i = freq[i]
        for j in range(n):
            freq_j = freq[j]
            result[i, j, 0] = pi_2 * np.sqrt(freq_i * freq_i + freq_j * freq_j)
            result[i, j, 1] = np.arctan2(freq_i, freq_j)
    return result[:, :, 0], result[:, :, 1]


@stencil
def _d_dx(w, d):
    '''``w``: 2d-array
    ``d``: pixel width
    '''
    return (w[0, 1] - w[0, -1]) / (2. * d)


@stencil
def _d_dy(w, d):
    '''``w``: 2d-array
    ``d``: pixel width
    '''
    return (w[1, 0] - w[-1, 0]) / (2. * d)


@stencil
def _d2_dx2(w, d):
    '''``w``: 2d-array
    ``d``: pixel width
    '''
    return (w[0, 1] - 2 * w[0, 0] + w[0, -1]) / (d * d)


@stencil
def _d2_dy2(w, d):
    '''``w``: 2d-array
    ``d``: pixel width
    '''
    return (w[1, 0] - 2 * w[0, 0] + w[-1, 0]) / (d * d)


@stencil
def _d2_dxdy(w, d):
    '''``w``: 2d-array
    ``d``: pixel width
    '''
    return (w[1, 1] + w[-1, -1] - w[1, -1] - w[-1, 1]) / (2. * d)


@jit(nopython=True, nogil=True, parallel=True)
def _qu2eb_rotation(QU_array, chi):
    '''QU to EB rotation, inplace.
    Written to minimized memory use and parallelize.
    (2 * n_threads + 2) * n**2 * 64 bit memory intermediate memory needed.
    '''
    # sin_2chi is 2_chi here, for memory reason
    sin_2chi = 2. * chi
    cos_2chi = np.cos(sin_2chi)
    sin_2chi = np.sin(sin_2chi)
    for i in prange(QU_array.shape[1]):
        Q_fft = QU_array[0, i]
        U_fft = QU_array[1, i]
        # E
        temp = Q_fft * cos_2chi + U_fft * sin_2chi
        # B
        QU_array[1, i] = U_fft * cos_2chi - Q_fft * sin_2chi
        QU_array[0, i] = temp


def _counter_term(QU_array, mask, pixel_size, ls, chi):
    '''ls[0, 0] is modified to np.inf
    temp1_f = (Q_dwdx_f + U_dwdy_f) * np.sin(chi) + (Q_dwdy_f - U_dwdx_f) * np.cos(chi)
    temp2_f = fft2(
        QU_array[0] * (2. * _d2_dxdy(mask, pixel_size)) + QU_array[1] * \
        (_d2_dy2(mask, pixel_size) - _d2_dx2(mask, pixel_size))
    )
    return (temp2_f / ls - 2.j * temp1_f) / ls
    written to reduce memory use
    requires (5 * n_obs + 2) * n**2 * 64 bit memory
    result needs 2 * n_obs * n**2 * 64 bit memory
    i.e. (3 * n_obs + 2) * n**2 * 64 bit memory intermediate memory needed.
    '''
    # dwdx, dwdy: (n, n), float
    dwdx = _d_dx(mask, pixel_size)
    dwdy = _d_dy(mask, pixel_size)

    # temp: (n_obs, n, n), float
    temp = QU_array[0] * dwdx
    # temp1_f: (n_obs, n, n), complex
    # Q_dwdx_f
    temp1_f = fft2(temp)
    temp = QU_array[1] * dwdy
    # U_dwdy_f
    temp1_f += fft2(temp)
    temp1_f *= np.sin(chi)

    temp = QU_array[0] * dwdy
    # temp2_f: (n_obs, n, n), complex
    # Q_dwdy_f
    temp2_f = fft2(temp)
    temp = QU_array[1] * dwdx
    # U_dwdx_f
    temp2_f -= fft2(temp)
    temp2_f *= np.cos(chi)
    temp1_f += temp2_f

    temp = QU_array[0] * (2. * _d2_dxdy(mask, pixel_size)) + \
        QU_array[1] * (_d2_dy2(mask, pixel_size) - _d2_dx2(mask, pixel_size))
    temp2_f = fft2(temp)

    # s.t. when divided yield 0
    ls[0, 0] = np.inf
    # dwdx holds 1/ls from now on
    dwdx = np.reciprocal(ls)
    temp2_f *= dwdx
    temp2_f -= 2.j * temp1_f
    temp2_f *= dwdx
    return temp2_f


def iqu_to_teb(iqu_array, tmask, pmask, tnorm, pnorm, pixel_size, n_x):
    '''``iqu_array``: masked IQU array, of shape (3, , n_obs, n_x, n_x)
    '''
    iqu_array[0, :, :, :] *= tmask
    iqu_array[1:, :, :, :] *= pmask

    ls, chi = _get_l_2d(n_x, pixel_size)

    # calculate counter_term first because it requires more intermediate memory
    counter_term = _counter_term(iqu_array[1:], pmask, pixel_size, ls, chi)
    # which can be released before teb_array is allocated

    teb_array = fft2(iqu_array)
    _qu2eb_rotation(teb_array[1:], chi)

    teb_array[2] += counter_term

    teb_array[0] *= tnorm
    teb_array[1:] *= pnorm
    return teb_array

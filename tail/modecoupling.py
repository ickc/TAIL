import numpy as np
from numba import jit, prange
from scipy.fftpack import fft2, next_fast_len

from dautil.util import zero_padding
from tail.numba_wrap import fftfreq
from tail.util import fill_nan, norm_fft, normalize_row


@jit(nopython=True, nogil=True, parallel=True)
def _bin_psd2(pixel_size, l_max, mask):
    '''identical to ``_bin_psd2_cross`` except
    that mask1 == mask2
    '''
    N = mask.shape[0]
    freq = fftfreq(N, pixel_size)

    n = l_max + 1
    psd_1d = np.zeros(n)
    hit = np.zeros(n, dtype=np.int64)

    pi_2 = np.pi * 2.
    for i in prange(N):
        freq_i = freq[i]
        for j in range(N):
            freq_j = freq[j]
            l = int(round(pi_2 * np.sqrt(freq_i * freq_i + freq_j * freq_j)))
            idx = l if l < l_max else l_max

            hit[idx] += 1

            # psd_2d
            mask_ij = mask[i, j]
            real = mask_ij.real
            imag = mask_ij.imag
            psd_1d[idx] += real * real + imag * imag

    psd_1d = psd_1d[:-1]
    hit = hit[:-1]

    for i in range(l_max):
        hit_ = hit[i]
        psd_1d[i] = psd_1d[i] / hit_ if hit_ > 0 else np.nan
    fill_nan(psd_1d)
    return psd_1d


@jit(nopython=True, nogil=True, parallel=True)
def _bin_psd2_cross(pixel_size, l_max, mask1, mask2):
    '''bins 2d fft to 1d integers
    '''
    N = mask1.shape[0]
    freq = fftfreq(N, pixel_size)

    n = l_max + 1
    psd_1d = np.zeros(n)
    hit = np.zeros(n, dtype=np.int64)

    pi_2 = np.pi * 2.
    for i in prange(N):
        freq_i = freq[i]
        for j in range(N):
            freq_j = freq[j]
            l = int(round(pi_2 * np.sqrt(freq_i * freq_i + freq_j * freq_j)))
            idx = l if l < l_max else l_max

            hit[idx] += 1

            # psd_2d
            mask1_ij = mask1[i, j]
            mask2_ij = mask2[i, j]
            psd_1d[idx] += mask1_ij.real * mask2_ij.real + mask1_ij.imag * mask2_ij.imag

    psd_1d = psd_1d[:-1]
    hit = hit[:-1]

    for i in range(l_max):
        hit_ = hit[i]
        psd_1d[i] = psd_1d[i] / hit_ if hit_ > 0 else np.nan
    fill_nan(psd_1d)
    return psd_1d


def _get_W(l_max, pixel_size, mask1, mask2=None, l_min=1):
    '''if ``mask2 is None``, get auto-psd of ``mask1``,
    else cross-psd of ``mask1`` and ``mask2``.
    return the 1d-spectrum, binned to integers up to (but not include) ``l_max``
    '''
    def _get_fft(mask, n_x):
        mask = zero_padding(mask, (n_x, n_x))
        return fft2(mask) * norm_fft(mask)

    n_x = max(int(round(np.pi / (pixel_size * l_min))), mask1.shape[0])
    n_x = next_fast_len(n_x)

    mask1_fft = _get_fft(mask1, n_x)
    mask2_fft = None if mask2 is None else _get_fft(mask2, n_x)

    W = _bin_psd2(pixel_size, l_max, mask1_fft) if mask2_fft is None else \
        _bin_psd2_cross(pixel_size, l_max, mask1_fft, mask2_fft)

    return W


@jit(nopython=True, nogil=True)
def _J_t(k1, k2, k3):
    '''See Eq. A10 from MASTER paper
    it actually returns J_t * pi / 2 because overall scale doesn't matter
    '''
    k1_2 = k1 * k1
    k2_2 = k2 * k2
    k3_2 = k3 * k3
    temp = 2 * (k1_2 * k2_2 + k2_2 * k3_2 + k3_2 * k1_2) - k1_2 * k1_2 - k2_2 * k2_2 - k3_2 * k3_2
    # factor of 2 / pi ignored
    # return 2. / (np.pi * np.sqrt(temp)) if temp > 0 else 0.
    return 1. / np.sqrt(temp) if temp > 0 else 0.


@jit(nopython=True, nogil=True)
def _get_alpha(k1, k2, k3):
    '''return the angle in [0, pi], corresponds to k1
    made in the triangle of k1, k2, k3
    essentially just cosine rule
    '''
    return np.arccos((k2 * k2 + k3 * k3 - k1 * k1) / (2 * k2 * k3))


def _get_J_p(Mtype, pure='hybrid'):
    '''supported cases:
    ('EEEE', 'hybrid'),
    ('BBBB', 'hybrid'),
    ('TETE', 'hybrid'),
    ('TBTB', 'hybrid'),
    ('EBEB', 'hybrid'),
    ('EBEB', 'pseudo')
    To include other cases, port them from commit 70fba3c.
    '''

    @jit(nopython=True, nogil=True)
    def tete(k1, k2, k3):
        alpha3 = _get_alpha(k3, k1, k2)
        return np.cos(2. * alpha3)

    @jit(nopython=True, nogil=True)
    def eeee(k1, k2, k3):
        alpha3 = _get_alpha(k3, k1, k2)
        temp = np.cos(2. * alpha3)
        return temp * temp

    @jit(nopython=True, nogil=True)
    def ebeb_pseudo(k1, k2, k3):
        alpha3 = _get_alpha(k3, k1, k2)
        return np.cos(4. * alpha3)

    @jit(nopython=True, nogil=True)
    def tbtb(k1, k2, k3):
        alpha1 = _get_alpha(k1, k2, k3)
        alpha3 = _get_alpha(k3, k1, k2)
        k3_k1 = k3 / k1
        temp = np.cos(2. * alpha3) + 2. * k3_k1 * np.cos(alpha3 - alpha1) + k3_k1 * k3_k1 * np.cos(2. * alpha1)
        return temp

    @jit(nopython=True, nogil=True)
    def bbbb(k1, k2, k3):
        alpha1 = _get_alpha(k1, k2, k3)
        alpha3 = _get_alpha(k3, k1, k2)
        k3_k1 = k3 / k1
        temp = np.cos(2. * alpha3) + 2. * k3_k1 * np.cos(alpha3 - alpha1) + k3_k1 * k3_k1 * np.cos(2. * alpha1)
        return temp * temp

    @jit(nopython=True, nogil=True)
    def ebeb(k1, k2, k3):
        alpha1 = _get_alpha(k1, k2, k3)
        alpha3 = _get_alpha(k3, k1, k2)

        alpha31 = alpha3 - alpha1
        alpha1 *= 2.
        alpha3 *= 2.

        k3_k1 = k3 / k1

        k3_k1_2 = k3_k1 * k3_k1
        k3_k1 *= 2.

        temp = np.cos(alpha3)
        temp *= temp + k3_k1 * np.cos(alpha31) + k3_k1_2 * np.cos(alpha1)
        temp2 = np.sin(alpha3)
        temp2 *= temp2 + k3_k1 * np.sin(alpha31) - k3_k1_2 * np.sin(alpha1)
        return temp - temp2

    if Mtype == 'EEEE':
        return eeee
    elif Mtype == 'BBBB':
        return bbbb
    elif Mtype == 'TETE':
        return tete
    elif Mtype == 'TBTB':
        return tbtb
    elif Mtype == 'EBEB':
        if pure == 'hybrid':
            return ebeb
        else:
            return ebeb_pseudo


def _get_M_gen(Mtype, pure='hybrid'):

    if Mtype == 'TTTT':
        _J = _J_t
    else:
        _J_p = _get_J_p(Mtype, pure='hybrid')

        @jit(nopython=True, nogil=True)
        def _J(k1, k2, k3):
            return _J_t(k1, k2, k3) * _J_p(k1, k2, k3)

    @jit(nopython=True, nogil=True)
    def simps(W, k1, k2):
        '''integrate W * J * k3 for k3 in (k3_min, k3_max)
        using Simpson's rule.
        1st term of Simpson's rule put at k3_min,
        hence the first non-zero terms are 4, then 2, ...
        which equals to 2 * (2 - i % 2)
        '''
        k3_min = np.abs(k1 - k2)
        k3_max = k1 + k2
        result = 0.
        for i, k3 in enumerate(range(k3_min + 1, k3_max)):
            result += (2 - i % 2) * _J(k1, k2, k3) * W[k3] * k3
        # factor of 2 / 3 ignored
        # return result / 1.5
        return result

    @jit(nopython=True, nogil=True, parallel=True)
    def _get_M(W, l_max, dl):
        '''Note that the middle of the l-bin is biased by 0.5.
        e.g. dl = 10. first bin is [0, 10), middle is chosen as 5,
        but it should be 4.5 instead.
        '''
        bin_width = dl // 2
        n = l_max // dl
        M = np.empty((n, n))
        for i in prange(n):
            k1 = bin_width + dl * i
            for j in range(n):
                k2 = bin_width + dl * j
                # factor of 2 pi ignored
                # M[i, j] = 2. * np.pi * k2 * simps(W, k1, k2)
                M[i, j] = k2 * simps(W, k1, k2)
        # from all the factors ignored above, it should return this instead
        # return M * (8. / 3.)
        return M

    return _get_M


def calc_M(mask1, mask2, Mtype, pure, pixel_size=0.0005817764173314432, l_max=3000, dl=10, normalize=True):
    '''assume ``l_max // dl == 0``, any excess will be included. e.g. if l_max=3001, dl=10, then
    the last bin is [3000, 3010)
    For no binning, set ``dl = 1``.
    '''
    # k3 < k1_max + k2_max = 2 * l_max - dl - dl % 2
    W = _get_W(2 * l_max - dl - dl % 2, pixel_size, mask1, mask2=mask2)

    get_M = _get_M_gen(Mtype, pure=pure)
    M = get_M(W, l_max, dl)
    if normalize:
        normalize_row(M)
    return M

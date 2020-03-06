import numpy as np
from numba import jit, prange

from dautil.util import abs2
from tail.numba_wrap import fftfreq
from tail.util import fill_nan

# nu ###########################################################################


@jit(nopython=True, nogil=True, parallel=True)
def compute_nu4(pixel_size, lmax, mask_i, mask_j, mask_k, mask_l):
    mask_ij = mask_i * mask_j
    mask_kl = mask_k * mask_l
    w2_ij = mask_ij.sum()
    w2_kl = mask_kl.sum()
    w4 = (mask_ij * mask_kl).sum()

    two_l_plus_1 = np.arange(1, 2 * lmax + 1, 2)
    return 0.25 * pixel_size * pixel_size * w2_ij * w2_kl / (np.pi * w4) * two_l_plus_1


@jit(nopython=True, nogil=True, parallel=True)
def compute_nu2(pixel_size, lmax, mask_i, mask_j):
    '''identical to compute_nu4(pixel_size, lmax, mask_i, mask_j, mask_i, mask_j)
    '''
    mask_ij = mask_i * mask_j
    w2_ij = mask_ij.sum() * pixel_size
    w4 = (np.square(mask_ij)).sum()

    two_l_plus_1 = np.arange(1, 2 * lmax + 1, 2)
    return 0.25 * w2_ij * w2_ij / (np.pi * w4) * two_l_plus_1


@jit(nopython=True, nogil=True, parallel=True)
def compute_nu(pixel_size, lmax, mask):
    '''
    Estimate number of degrees of freedom in a map
    assume dl == 1
    identical to compute_nu2(pixel_size, lmax, mask, mask)
    '''
    temp = np.square(mask)
    w2_pixel_size = temp.sum() * pixel_size
    w4 = np.square(temp).sum()
    two_l_plus_1 = np.arange(1, 2 * lmax + 1, 2)
    return (w2_pixel_size * w2_pixel_size * 0.25 / (np.pi * w4)) * two_l_plus_1

# util #########################################################################


@jit(nopython=True, nogil=True, parallel=True)
def _bin_psd(pixel_size, l_max, psd):
    '''similar to ``_bin_psd2`` but psd is already
    sqaured and ``l`` is already calculated
    '''
    N = psd.shape[0]
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

            psd_1d[idx] += psd[i, j]

    psd_1d = psd_1d[:-1]
    hit = hit[:-1]

    ak_to_dl_scale = N * pixel_size
    ak_to_dl_scale *= 0.5 * ak_to_dl_scale / np.pi

    for i in range(l_max):
        hit_ = hit[i]
        psd_1d[i] = (psd_1d[i] * ak_to_dl_scale / hit_) * (i * (i + 1)) if hit_ > 0 else np.nan
    # explicitly leave nan at the boundaries to show that
    # we have no information there
    fill_nan(psd_1d, fill_boundary=False)
    return psd_1d

# get pseudo-spectra ###########################################################


@jit(nopython=True, nogil=True, parallel=True)
def _form_1d_spectra(pixel_size, lmax, mapfs, weights):
    '''get auto, cross spectra
    '''
    n_obs = weights.size
    # auto holds sum over i, i
    auto = _bin_psd(
        pixel_size,
        lmax,
        (abs2(mapfs) * np.square(weights).reshape(n_obs, 1, 1)).sum(axis=0)
    )
    auto_weight = np.square(weights).sum()

    # cross holds sum over i, j, i != j
    # .reshape(-1, 1, 1) is the same as [:, None, None]
    cross = _bin_psd(
        pixel_size,
        lmax,
        abs2((mapfs * weights.reshape(n_obs, 1, 1)).sum(axis=0))
    ) - auto
    cross_weight = weights.sum()
    cross_weight *= cross_weight
    cross_weight -= auto_weight

    cross /= cross_weight
    auto /= auto_weight
    return auto, cross


@jit(nopython=True, nogil=True, parallel=True)
def _form_1d_cross_spectra_nodiag(pixel_size, lmax, mapX, weightX, mapY, weightY, n_auto=-1):
    '''
    Form cross spectra between e.g. T and E
    ``n_auto``: the first no. of ``n_auto`` elements of each array are the auto-map,
    i.e. per i-th element they comes from the same observation
    '''
    if n_auto < 1:
        n_auto = weightX.size
    n_obsX = weightX.size
    n_obsY = weightY.size

    # auto holds sum over i, i
    autoXY = _bin_psd(
        pixel_size,
        lmax,
        ((mapX[:n_auto] * mapY[:n_auto].conjugate()).real * (weightX[:n_auto] * weightY[:n_auto]).reshape(n_auto, 1, 1)).sum(axis=0)
    )
    auto_weight_XY = (weightX[:n_auto] * weightY[:n_auto]).sum()

    crossXY = _bin_psd(pixel_size, lmax, (
        (mapX * weightX.reshape(n_obsX, 1, 1)).sum(axis=0) *
        (mapY * weightY.reshape(n_obsY, 1, 1)).sum(axis=0).conjugate()
    ).real) - autoXY
    cross_weight_XY = weightX.sum() * weightY.sum() - auto_weight_XY

    crossXY /= cross_weight_XY
    autoXY /= auto_weight_XY
    return autoXY, crossXY


@jit(nopython=True, nogil=True, parallel=True)
def _form_1d_cross_spectra_cross(
    pixel_size, lmax,
    map0A, weight0A,
    map0B, weight0B,
    map1A, weight1A,
    map1B, weight1B,
    n_auto=-1
):
    '''
    return auto, cross

    if len(maps) != len(weights),
    then it will still calculate, but terminates at shortest one first.
    '''
    # TODO: use nodiag only for same observation
    auto01, cross01 = _form_1d_cross_spectra_nodiag(pixel_size, lmax, map0A, weight0A, map1B, weight1B, n_auto=n_auto)
    auto10, cross10 = _form_1d_cross_spectra_nodiag(pixel_size, lmax, map1A, weight1A, map0B, weight0B, n_auto=n_auto)

    weight01 = weight0A.sum() * weight1B.sum()
    weight10 = weight1A.sum() * weight0B.sum()

    # multiply by weight
    auto01 *= weight01
    auto10 *= weight10
    cross01 *= weight01
    cross10 *= weight10

    # these now holds the total
    auto01 += auto10
    cross01 += cross10
    weight01 += weight10

    # devide by total weight
    auto01 /= weight01
    cross01 /= weight01

    return auto01, cross01

# main functions ###############################################################


@jit(nopython=True, nogil=True, parallel=True)
def pseudospectra_auto(
    pixel_size, lmax,
    mapfs, weights  # , mask
):
    '''``mapfs``: map in Fourier domain, stacked in first axis
    l range: [0, lmax)
    '''

    def estimate_neff(weights):
        ''' Correct average power spectrum estimates for effective number of maps '''
        w = weights.sum()
        return w * w / np.square(weights).sum()

    # cross is the signal only
    auto, cross = _form_1d_spectra(pixel_size, lmax, mapfs, weights)

    # noise as auto - cross, corrected by n_eff
    auto -= cross
    auto /= estimate_neff(weights)

    # nu = compute_nu(pixel_size, lmax, mask)
    return cross, auto  # , nu


@jit(nopython=True, nogil=True, parallel=True)
def pseudospectra_cross(
    pixel_size, lmax,
    tfs, tweights,  # tmasks,
    efs, pweights,  # pmasks
    n_auto=-1
):
    '''TODO: in the case of auto-spectra in null-cross-spectra,
    add a number that defines up to that no. of maps the observations are
    identical, i.e. only up to that no. of maps auto-spectra should be calculated.
    '''
    # cross is the signal only
    auto, cross = _form_1d_cross_spectra_nodiag(pixel_size, lmax, tfs, tweights, efs, pweights, n_auto=n_auto)

    # noise as auto - cross
    auto -= cross

    # nu = compute_nu2(pixel_size, lmax, tmasks, pmasks)
    return cross, auto  # , nu


@jit(nopython=True, nogil=True, parallel=True)
def pseudospectra_cross_cross(
    pixel_size, lmax,
    tf0s, tweight0s,  # tmask0s,
    ef0s, pweight0s,  # pmask0s,
    tf1s, tweight1s,  # tmask1s,
    ef1s, pweight1s,  # pmask1s
    n_auto
):

    # cross is the signal only
    auto, cross = _form_1d_cross_spectra_cross(
        pixel_size, lmax,
        tf0s, tweight0s,
        ef0s, pweight0s,
        tf1s, tweight1s,
        ef1s, pweight1s,
        n_auto=n_auto
    )
    # noise as auto - cross
    auto -= cross
    # nu = compute_nu4(pixel_size, lmax, tmask0s, pmask1s, tmask1s, pmask0s)
    return cross, auto  # , nu

# all ##########################################################################


@jit(nopython=True, nogil=True, parallel=True)
def pseudospectra_auto_all(pixel_size, lmax, teb_array, tpweights):
    t_s, t_n = pseudospectra_auto(pixel_size, lmax, teb_array[0], tpweights[0])
    e_s, e_n = pseudospectra_auto(pixel_size, lmax, teb_array[1], tpweights[1])
    b_s, b_n = pseudospectra_auto(pixel_size, lmax, teb_array[2], tpweights[1])
    te_s, te_n = pseudospectra_cross(pixel_size, lmax, teb_array[0], tpweights[0], teb_array[1], tpweights[1])
    tb_s, tb_n = pseudospectra_cross(pixel_size, lmax, teb_array[0], tpweights[0], teb_array[2], tpweights[1])
    eb_s, eb_n = pseudospectra_cross(pixel_size, lmax, teb_array[1], tpweights[1], teb_array[2], tpweights[1])
    return t_s, t_n, e_s, e_n, b_s, b_n, te_s, te_n, tb_s, tb_n, eb_s, eb_n


@jit(nopython=True, nogil=True, parallel=True)
def pseudospectra_cross_all(pixel_size, lmax, teb_array0, tpweights0, teb_array1, tpweights1, n_auto):
    t_s, t_n = pseudospectra_cross(pixel_size, lmax, teb_array0[0], tpweights0[0], teb_array1[0], tpweights1[0], n_auto=n_auto)
    e_s, e_n = pseudospectra_cross(pixel_size, lmax, teb_array0[1], tpweights0[1], teb_array1[1], tpweights1[1], n_auto=n_auto)
    b_s, b_n = pseudospectra_cross(pixel_size, lmax, teb_array0[2], tpweights0[1], teb_array1[2], tpweights1[1], n_auto=n_auto)
    te_s, te_n = pseudospectra_cross_cross(
        pixel_size, lmax,
        teb_array0[0], tpweights0[0],
        teb_array0[1], tpweights0[1],
        teb_array1[0], tpweights1[0],
        teb_array1[1], tpweights1[1],
        n_auto
    )
    tb_s, tb_n = pseudospectra_cross_cross(
        pixel_size, lmax,
        teb_array0[0], tpweights0[0],
        teb_array0[2], tpweights0[1],
        teb_array1[0], tpweights1[0],
        teb_array1[2], tpweights1[1],
        n_auto
    )
    eb_s, eb_n = pseudospectra_cross_cross(
        pixel_size, lmax,
        teb_array0[1], tpweights0[1],
        teb_array0[2], tpweights0[1],
        teb_array1[1], tpweights1[1],
        teb_array1[2], tpweights1[1],
        n_auto
    )
    return t_s, t_n, e_s, e_n, b_s, b_n, te_s, te_n, tb_s, tb_n, eb_s, eb_n

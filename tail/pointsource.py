import numpy as np
import pandas as pd
import quaternion
from numba import generated_jit, jit, types, prange

# 10.5 arcmin in radian
FWHM = 0.0030543261909900766


@jit(nopython=True, nogil=True)
def _projector(ra_dec):
    '''OLEA projector
    Note that this convention is different from AnalysisBackend
    AnalysisBackend output (ys, xs), where here the order is xs first
    '''
    result = np.empty_like(ra_dec)
    cos_dec = np.cos(ra_dec[1, :])
    # result[1] temporarily holds k
    result[0, :] = np.sqrt(2. / (1. + np.cos(ra_dec[0, :]) * cos_dec))
    result[1, :] = result[0] * np.sin(ra_dec[0, :]) * cos_dec
    result[0, :] *= np.sin(ra_dec[1, :])
    return result


@generated_jit(nopython=True, nogil=True, parallel=True)
def _euler_array(alpha, axis):

    def not_array(alpha, axis):
        result = np.array([
            np.cos(alpha * 0.5),
            0.,
            0.,
            0.,
        ])
        result[axis] = np.sin(alpha * 0.5)
        return result

    def is_array(alpha, axis):
        result = np.zeros((alpha.size, 4), dtype=alpha.dtype)
        for i in range(alpha.size):
            result[i, 0] = np.cos(alpha[i] * 0.5)
            result[i, axis] = np.sin(alpha[i] * 0.5)
        return result
    if isinstance(alpha, types.Array):
        return is_array
    else:
        return not_array


def _euler(alpha, axis):
    return quaternion.as_quat_array(_euler_array(alpha, axis))


# @jit(nopython=True, nogil=True, parallel=True)
# def _quat_to_decpa(seq):
#     q0 = seq[:, 0]
#     q1 = seq[:, 1]
#     q2 = seq[:, 2]
#     q3 = seq[:, 3]
#     # result holds theta, psi
#     result = np.empty((q0.size, 2))
#     result[0] = np.arcsin(2 * (q0 * q2 - q3 * q1))
#     result[1] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
#     return result


@jit(nopython=True, nogil=True, parallel=True)
def _quat_for_offset(seq):
    q0 = seq[:, 0]
    q1 = seq[:, 1]
    q2 = seq[:, 2]
    q3 = seq[:, 3]
    result = np.empty((2, q0.size))
    result[0] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    result[1] = -np.arcsin(2 * (q0 * q2 - q3 * q1))
    return result


def _offset(ra, dec, ra0, dec0):
    q = _euler(dec0, 2) * _euler(ra - ra0, 3) * _euler(-dec, 2)
    return _quat_for_offset(quaternion.as_float_array(q))


# these are more general but not used
# @jit(nopython=True, nogil=True, parallel=True)
# def _quat_to_radecpa(seq):
#     q0 = seq[:, 0]
#     q1 = seq[:, 1]
#     q2 = seq[:, 2]
#     q3 = seq[:, 3]
#     phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
#     theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
#     psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
#     return phi, theta, psi


# def offset_radecpa_makequat(ra, dec, pa, racen, deccen):
#     return _euler_y(deccen) * _euler_z(-racen) * _euler_z(ra) * _euler_y(-dec) * _euler_x(-pa)


# def offset_radecpa_applyquat(q, azd, eld):
#     seq = q * _euler_z(-azd) * _euler_y(-eld)

#     phi, theta, psi = _quat_to_radecpa(quaternion.as_float_array(seq))

#     return psi, -theta, -phi


def _return_fixed_source_pos(racen, deccen, source_dict):
    '''Return the position of the fixed sources given a center ra/dec'''
    ra = source_dict.loc['ra'].values
    dec = source_dict.loc['dec'].values
    return _offset(ra, dec, racen, deccen)


@jit(nopython=True, nogil=True)
def inverted_gaussian(x):
    '''inverted Gaussian 0 at center and 1 at infinity,
    with FWHM at x = 1
    '''
    return 1. - np.power(2., -(x * x))


@jit(nopython=True, nogil=True, parallel=True)
def _punch_mask(xmin, xmax, ymin, ymax, shape, fs, fwhm=FWHM):
    '''
    Construct a point source mask by punching holes in a map
    '''
    fs_in_box = fs[:, (xmin <= fs[0]) & (fs[0] <= xmax) & (ymin <= fs[1]) & (fs[1] <= ymax)].T.copy()
    n_ps = fs_in_box.shape[0]

    Δx = (xmax - xmin) / (shape[0] - 1)
    Δy = (ymax - ymin) / (shape[1] - 1)

    mask = np.ones((shape[0], shape[1]))
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(n_ps):
                dx = fs_in_box[k, 0] - (xmin + i * Δx)
                dy = fs_in_box[k, 1] - (ymin + j * Δy)
                d = np.sqrt(dx * dx + dy * dy)
                mask[i, j] *= inverted_gaussian(d / fwhm)
    return mask


def mask_from_map(
    source,
    shape,
    xmin, xmax, ymin, ymax,
    catfn
):
    df = pd.read_hdf(catfn)
    ra0, dec0 = df.loc[source, ['ra', 'dec']]

    catbyname = df.loc[df['is_fixed'], ['ra', 'dec']].T

    fs = _projector(_return_fixed_source_pos(ra0, dec0, catbyname))

    point_source_mask = _punch_mask(xmin, xmax, ymin, ymax, shape, fs)

    return point_source_mask

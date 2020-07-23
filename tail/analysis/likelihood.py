from .container import Spectra, get_idx, get_idxs, FSky
from .foreground import DUST_AMP, DUST_AMP_STD

import sys

import numpy as np
import pandas as pd
from numba import jit, float64, bool_, int32, complex128, prange
from scipy.optimize import minimize
from scipy.stats import chi2

import emcee

import seaborn as sns

SPECTRA_MATRIX_ORDER = 'TEB'

DUST_POWER = -0.58
DUST_KNEE = 80.

# conversion ###################################################################


def to_spectra_matrix(array, spectra, cases=SPECTRA_MATRIX_ORDER):
    '''convert spectra into a matrix form

    :param array: starts with a dimension that holds different spectra
    :param spectra: list of spectrum names

    return
    ------
    the last 2 dimensions become the indices for spectra in matrix form
    '''
    shape_in = array.shape
    n = shape_in[0]
    m = len(cases)
    # the resultant matrix is a symmetric matrix
    assert m * (m + 1) == 2 * n

    shape_out = list(shape_in[1:]) + [m, m]
    res = np.empty(shape_out, dtype=array.dtype)
    for i in range(m):
        for j in range(i, m):
            spectrum = cases[i] + cases[j]
            idx = get_idx(spectra, spectrum)
            temp = array[idx]
            res[..., i, j] = temp
            if i != j:
                res[..., j, i] = temp
    return res


def from_spectra_matrix(array, spectra, cases=SPECTRA_MATRIX_ORDER):
    '''inverse function of `to_spectra_matrix`
    '''
    shape_in = array.shape
    n = len(spectra)
    m = shape_in[-1]
    assert shape_in[-2] == m
    assert m * (m + 1) == 2 * n

    shape_out = [n] + list(shape_in[:-2])
    res = np.empty(shape_out, dtype=array.dtype)
    for i in range(m):
        for j in range(i, m):
            spectrum = cases[i] + cases[j]
            idx = get_idx(spectra, spectrum)
            temp = array[..., i, j]
            res[idx] = temp
    return res

# Transformation ###############################################################


def get_rel_dust_spectra(spectra: Spectra):
    w_bl = spectra.w_bl[get_idx(spectra.spectra, 'BB'), 0, 0]
    l_min = spectra.l_min
    l_max = spectra.l_max

    return w_bl @ (np.power(np.arange(l_min, l_max) / DUST_KNEE, DUST_POWER))


@jit(float64[:](float64, int32, int32, int32, float64), nopython=True, nogil=True, cache=True)
def _beam_err(sigma_sq, l_min, bin_width, n_b, bin_width_inverse):
    delta_B_sq = np.zeros(n_b)
    for b in range(n_b):
        for dl in range(bin_width):
            l = l_min + b * bin_width + dl
            delta_B_sq[b] += np.exp(-l * (l + 1) * sigma_sq)
        delta_B_sq[b] *= bin_width_inverse
    return delta_B_sq


@jit(float64[:](float64, int32, int32, int32), nopython=True, nogil=True, cache=True)
def beam_err(sigma_sq, l_min, l_max, bin_width):
    '''calculate the binned beam error'''
    n = l_max - l_min
    n_b = n // bin_width
    assert n % bin_width == 0
    bin_width_inverse = 1. / bin_width
    return _beam_err(sigma_sq, l_min, bin_width, n_b, bin_width_inverse)


@jit(float64[:, :, ::1](float64, float64[:, :, ::1], float64[::1]), nopython=True, nogil=True, cache=True)
def scale_by_amplitude_parameter(theta, theory_spectra, leakage_spectra_BB):
    '''scale the theory BB spectra

    `theta`: A_BB
    `theory_spectra`: spectra in matrix form
    `leakage_spectra_BB`: BB leakage which won't scale with A_BB
    '''
    res = theory_spectra.copy()
    res[:, 2, 2] -= leakage_spectra_BB
    res[:, 2, 2] *= theta
    res[:, 2, 2] += leakage_spectra_BB
    return res


@jit(float64[:, :, ::1](float64[::1], float64[:, :, ::1], int32, int32, int32), nopython=True, nogil=True, cache=True)
def transform(thetas, array, l_min, l_max, bin_width):
    '''rotate and scale spectra in matrix form

    `thetas`: first 2 are scaling factors, 3rd one is angle to rotate, 4th is sigma_sq
    `array`: spectra in matrix form

    calling 1/a, theta, -sigma_sq on inverse_transform should be inverse of this

    This function is not used anywhere as of writing, but for testing the inverse relationship
    '''
    bin_width_inverse = 1. / bin_width
    n_b = array.shape[0]

    a = np.empty(3, dtype=thetas.dtype)
    a[0] = thetas[0]
    a[1] = thetas[1]
    a[2] = thetas[1]
    theta = thetas[2]
    sigma_sq = thetas[3]

    b_sq = _beam_err(sigma_sq, l_min, bin_width, n_b, bin_width_inverse)

    R = np.zeros((3, 3))
    R[0, 0] = 1.
    cos_2theta = np.cos(2. * theta)
    sin_2theta = np.sin(2. * theta)
    R[1, 1] = cos_2theta
    R[1, 2] = sin_2theta
    R[2, 1] = -sin_2theta
    R[2, 2] = cos_2theta

    res = np.zeros_like(array)
    for b in range(n_b):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        res[b, i, j] += b_sq[b] * R[i, k] * R[j, l] * a[k] * a[l] * array[b, k, l]
    return res


@jit(
    [
        complex128[:, :, ::1](float64[::1], complex128[:, :, ::1], int32, int32, int32),
        float64[:, :, ::1](float64[::1], float64[:, :, ::1], int32, int32, int32),
    ],
    nopython=True, nogil=True, cache=True,
)
def inverse_transform(thetas, array, l_min, l_max, bin_width):
    '''rotate and scale measured spectra in matrix form

    `thetas`: first 2 are scaling factors, 3rd one is angle to rotate, 4th is sigma_sq
    `array`: spectra in matrix form (global var)

    calling 1/a, theta, -sigma_sq on transform should be inverse of this
    '''
    bin_width_inverse = 1. / bin_width
    n_b = array.shape[0]

    a = np.empty(3, dtype=thetas.dtype)
    a[0] = thetas[0]
    a[1] = thetas[1]
    a[2] = thetas[1]
    theta = thetas[2]
    sigma_sq = thetas[3]

    b_sq = _beam_err(sigma_sq, l_min, bin_width, n_b, bin_width_inverse)

    R = np.zeros((3, 3))
    R[0, 0] = 1.
    cos_2theta = np.cos(2. * theta)
    sin_2theta = np.sin(2. * theta)
    R[1, 1] = cos_2theta
    R[1, 2] = -sin_2theta
    R[2, 1] = sin_2theta
    R[2, 2] = cos_2theta

    res = np.zeros_like(array)
    for b in range(n_b):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        res[b, i, j] += b_sq[b] * a[i] * a[j] * R[j, l] * R[i, k] * array[b, k, l]
    return res


@jit(
    [
        complex128[:, :, :, ::1](float64[::1], complex128[:, :, :, ::1], int32, int32, int32),
        float64[:, :, :, ::1](float64[::1], float64[:, :, :, ::1], int32, int32, int32),
    ],
    nopython=True, nogil=True, cache=True, parallel=True,
)
def inverse_transform_batch(thetas, array, l_min, l_max, bin_width):
    '''batch running inverse_transform per first dimension of array'''
    n = array.shape[0]
    res = np.empty_like(array)
    for i in prange(n):
        res[i] = inverse_transform(thetas, array[i], l_min, l_max, bin_width)
    return res


def run_test(array=None, l_min=None, l_max=None, bin_width=None):
    '''test get_transform and get_inverse_transform
    `array`: some sort of theory spectra s.t. TB, EB are 0.
    '''
    if array is None:
        # fake theory
        array = np.random.rand(24, 3, 3)
        # symmetric
        array += array.transpose(0, 2, 1)
        # TB, EB = 0
        array[:, 0, 2] = 0.
        array[:, 2, 0] = 0.
        array[:, 1, 2] = 0.
        array[:, 2, 1] = 0.
    if l_min is None:
        l_min = 600
    if l_max is None:
        l_max = 3000
    if bin_width is None:
        bin_width = 50

    # sanity check for transform
    np.testing.assert_array_equal(array, transform(np.array([1., 1., 0., 0.]), array, l_min, l_max, bin_width))

    res = transform(np.array([1.1, 1.2, 0.1, 0.]), array, l_min, l_max, bin_width)
    np.testing.assert_array_equal(res[:, 1, 2], res[:, 2, 1])
    np.testing.assert_array_equal(res[:, 0, 2], res[:, 2, 0])
    np.testing.assert_array_equal(res[:, 0, 1], res[:, 1, 0])

    # compare to "Self-calibrating" paper
    for theta in (np.random.rand(10) * (2. * np.pi)):
        theta_2 = theta * 2.

        mine = transform(np.array([1., 1., theta, 0.]), array, l_min, l_max, bin_width)

        np.testing.assert_array_equal(mine[:, 0, 1], np.cos(theta_2) * array[:, 0, 1])
        np.testing.assert_array_equal(mine[:, 1, 1], np.sin(theta_2)**2 * array[:, 2, 2] + np.cos(theta_2)**2 * array[:, 1, 1])
        np.testing.assert_allclose(mine[:, 1, 2], 0.5 * np.sin(4. * theta) * (array[:, 2, 2] - array[:, 1, 1]))
        np.testing.assert_array_equal(mine[:, 0, 2], -np.sin(theta_2) * array[:, 0, 1])
        np.testing.assert_array_equal(mine[:, 2, 2], np.cos(theta_2)**2 * array[:, 2, 2] + np.sin(theta_2)**2 * array[:, 1, 1])

    # sanity check for inverse_transform
    np.testing.assert_array_equal(array, inverse_transform(np.array([1., 1., 0., 0.]), array, l_min, l_max, bin_width))

    # check transform and inverse becomes identity
    for thetas in (np.random.rand(2, 4) * (2. * np.pi)):
        thetas[3] *= 1.e-10
        transformed = transform(thetas, array, l_min, l_max, bin_width)
        thetas_inv = np.empty_like(thetas)
        thetas_inv[0] = 1. / thetas[0]
        thetas_inv[1] = 1. / thetas[1]
        thetas_inv[2] = thetas[2]
        thetas_inv[3] = -thetas[3]
        np.testing.assert_allclose(array, inverse_transform(thetas_inv, transformed, l_min, l_max, bin_width), atol=1e-13)


# likelihood ###################################################################

# θ / thetas
# thetas[0] = A_BB
# thetas[1] = a_T
# thetas[2] = a_E
# thetas[3] = theta
# thetas[4] = sigma_sq
# thetas[5] = A_dust


@jit(bool_(float64[::1]), nopython=True, nogil=True, cache=True)
def prior(θ):
    '''global prior constraint

    need to double check the MCMC result isn't anywhere near these boundaries
    '''
    # degree
    scale = 30. / 180. * np.pi
    return (
        (0.5 < θ[1]) &
        (θ[1] < 2.) &
        (0.5 < θ[2]) &
        (θ[2] < 2.) &
        (-scale < θ[3]) &
        (θ[3] < scale)
    )


@jit(float64(float64[::1], complex128[:, :, ::1], float64[:, :, ::1], float64[::1], float64[::1], float64[::1], int32, int32, int32, int32), nopython=True, nogil=True, cache=True)
def log_likelihood(thetas, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx):
    '''
    thetas[0] = A_BB
    thetas[1] = a_T
    thetas[2] = a_E
    thetas[3] = theta
    thetas[4] = sigma_sq
    thetas[5] = A_dust

    :param int l_T_cutoff_idx: drop TT, TE, TB beyond to this threshold, in terms of the b-th bin index. Set this to -1 to disable this behavior
    '''
    if not prior(thetas):
        return -np.inf

    # complex, real is signal, imag is noise
    C_total_est_complex = inverse_transform(thetas[1:5], obs_spectra, l_min, l_max, bin_width)
    C_total = scale_by_amplitude_parameter(thetas[0], theory_spectra, leakage_spectra_BB)

    Nb = C_total_est_complex.imag
    C_total_est = C_total_est_complex.real + Nb

    n = C_total.shape[0]
    # auto noise-spectra only
    for b in range(n):
        for i in range(3):
            C_total[b, i, i] += Nb[b, i, i]

    # dust template
    C_total[:, 2, 2] += thetas[5] * rel_dust_spectra

    res = 0.
    # invalid l_T_cutoff_idx is considered to be disabling this cutoff
    if not (0 <= l_T_cutoff_idx <= n):
        l_T_cutoff_idx = n
    # TEB
    for b in range(l_T_cutoff_idx):
        # C-est times C-inverse
        temp = np.linalg.solve(C_total[b], C_total_est[b])
        res += dof[b] * (np.trace(temp) - np.log(np.linalg.det(temp)) - 3.)
    # EB only
    for b in range(l_T_cutoff_idx, n):
        # C-est times C-inverse
        temp = np.linalg.solve(C_total[b, 1:][:, 1:], C_total_est[b, 1:][:, 1:])
        res += dof[b] * (np.trace(temp) - np.log(np.linalg.det(temp)) - 2.)
    # dust: Guassian likelihood
    res += ((thetas[5] - DUST_AMP) / DUST_AMP_STD)**2

    return -0.5 * res


@jit(float64(float64[::1], complex128[:, :, ::1], float64[:, :, ::1], float64[::1], float64[::1], float64[::1], int32, int32, int32, int32), nopython=True, nogil=True, cache=True)
def neg_log_likelihood(thetas, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx):
    '''negative log-likelihood

    mainly for scipy.optimize.minimize
    '''
    return -log_likelihood(thetas, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx)


@jit(float64(float64[::1], float64, complex128[:, :, ::1], float64[:, :, ::1], float64[::1], float64[::1], float64[::1], int32, int32, int32, int32), nopython=True, nogil=True, cache=True)
def neg_log_likelihood_fixed(thetas, A_BB, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx):
    '''fixing A_BB in the likelihood

    mainly for scipy.optimize.minimize
    '''
    n = thetas.size
    thetas_null = np.empty(n + 1)
    thetas_null[0] = A_BB
    thetas_null[1:] = thetas
    return -log_likelihood(thetas_null, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx)


@jit(float64(float64[::1], float64[::1], complex128[:, :, ::1], float64[:, :, ::1], float64[::1], float64[::1], float64[::1], int32, int32, int32, int32), nopython=True, nogil=True, cache=True)
def likelihood_ratio(theta_hyp, theta, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx):
    return -2. * (
        log_likelihood(theta_hyp, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx) -
        log_likelihood(theta, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff_idx)
    )


@jit(float64[:, :, ::1](float64[::1], float64[:, :, ::1], float64[::1], int32, int32, int32), nopython=True, nogil=True, cache=True)
def inverse_transform_subtract_foreground(thetas, obs_spectra, rel_dust_spectra, l_min, l_max, bin_width):
    '''real spectra transformed and foreground subtracted using the likelihood thetas
    
    c.f. `log_likelihood`.
    '''
    C_total_est = inverse_transform(thetas[1:5], obs_spectra, l_min, l_max, bin_width)
    C_total_est[:, 2, 2] -= thetas[5] * rel_dust_spectra
    return C_total_est


@jit(float64[:, :, ::1](float64[:, ::1], float64[:, :, ::1], float64[:, :, ::1], float64[::1], int32, int32, int32), nopython=True, nogil=True, cache=True, parallel=True)
def err_likelihood(thetas_all, Cl, Cl_map, rel_dust_spectra, l_min, l_max, bin_width):
    '''calculate the error of the real spectra given the MCMC chain from the likelihood

    :param Cl_map: MAP real spectra from the MAP thetas
    '''
    sq_sum = np.zeros_like(Cl)
    n = thetas_all.shape[0]
    for i in prange(n):
        sq_sum += np.square(inverse_transform_subtract_foreground(thetas_all[i], Cl, rel_dust_spectra, l_min, l_max, bin_width) - Cl_map)
    return np.sqrt(sq_sum / n)


class ComputeLikelihood(object):

    def __init__(self, spectra: Spectra, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, l_min, l_max, bin_width, l_T_cutoff=None):
        self.spectra = spectra
        self.obs_spectra = obs_spectra
        self.theory_spectra = theory_spectra
        self.leakage_spectra_BB = leakage_spectra_BB
        self.rel_dust_spectra = rel_dust_spectra
        self.dof = dof
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = bin_width
        if l_T_cutoff is None:
            self.l_T_cutoff_idx = -1
        else:
            # included only if the whole bin is <= l_T_cutoff
            self.l_T_cutoff_idx = (l_T_cutoff + 1 - l_min) // bin_width
            print(f'l_T_cutoff is set to {l_T_cutoff} and l_T_cutoff_idx is determined to be self.l_T_cutoff_idx. Only these bins from TT, TE, TB will be included in likelihood estimation: {self.spectra.b_arange[:self.l_T_cutoff_idx]}', file=sys.stderr)

    @classmethod
    def load(cls, spectra: Spectra, f_sky: FSky, enforce_zero_theory=True, l_T_cutoff=None):
        '''prepare data from `spectra` for computing Likelihood

        :param bool enforce_zero_theory: if True, then enforce TB, EB spectra to be 0 in theory.

        leakage should not have been subtracted in `spectra` since it contributes to the Cl_total
        '''
        assert spectra.is_full

        temp = spectra.signal[:, 0, 0, :]
        if enforce_zero_theory:
            for idx in get_idxs(spectra.spectra, ('TB', 'TE')):
                temp[idx] = 0.
        theory_spectra = to_spectra_matrix(temp, spectra.spectra)
        leakage_spectra_BB = spectra.leakage[get_idx(spectra.spectra, 'BB'), 0, 0]

        # get realmaps
        map_case = spectra.map_case_real
        Cls = getattr(spectra, map_case)
        # pack signal in real and noise in imag
        Cls_complex = Cls[:, 0, 0, :, 0, 0] + Cls[:, 0, 0, :, 1, 0] * 1.j
        obs_spectra = to_spectra_matrix(Cls_complex, spectra.spectra)

        # Primary Sicence product is BB so we choose f_sky accordingly (i.e. it is less than optimal in TB, TE, TT)
        dof = spectra.rel_dof[get_idx(spectra.spectra, 'BB'), 0, 0] * f_sky.get_f_sky('p', 'noise')

        # dust template
        rel_dust_spectra = get_rel_dust_spectra(spectra)

        return cls(spectra, obs_spectra, theory_spectra, leakage_spectra_BB, rel_dust_spectra, dof, spectra.l_min, spectra.l_max, spectra.bin_width, l_T_cutoff=l_T_cutoff)

    @property
    def map(self):
        res = minimize(
            neg_log_likelihood,
            np.array([1., 1., 1., 0., 1.e-8, DUST_AMP]),
            args=(
                self.obs_spectra,
                self.theory_spectra,
                self.leakage_spectra_BB,
                self.rel_dust_spectra,
                self.dof,
                self.l_min,
                self.l_max,
                self.bin_width,
                self.l_T_cutoff_idx,
            ),
            method='Nelder-Mead',
            options={'maxiter': 10000},
        )
        print(res, file=sys.stderr)
        assert res.success
        return res.x

    @property
    def map_null(self):
        res = minimize(
            neg_log_likelihood_fixed,
            np.array([1., 1., 0., 1.e-8, DUST_AMP]),
            args=(
                0.,
                self.obs_spectra,
                self.theory_spectra,
                self.leakage_spectra_BB,
                self.rel_dust_spectra,
                self.dof,
                self.l_min,
                self.l_max,
                self.bin_width,
                self.l_T_cutoff_idx,
            ),
            method='Nelder-Mead',
            options={'maxiter': 10000},
        )
        print(res, file=sys.stderr)
        assert res.success
        theta_null = np.concatenate(([0.], res.x))
        return theta_null

    @property
    def map_fiducial(self):
        res = minimize(
            neg_log_likelihood_fixed,
            np.array([1., 1., 0., 1.e-8, DUST_AMP]),
            args=(
                1.,
                self.obs_spectra,
                self.theory_spectra,
                self.leakage_spectra_BB,
                self.rel_dust_spectra,
                self.dof,
                self.l_min,
                self.l_max,
                self.bin_width,
                self.l_T_cutoff_idx,
            ),
            method='Nelder-Mead',
            options={'maxiter': 10000},
        )
        print(res, file=sys.stderr)
        assert res.success
        theta_fiducial = np.concatenate(([1.], res.x))
        return theta_fiducial

    @property
    def LR_null(self):
        '''use Wilks' theorem as approximation'''
        dist = chi2(1)
        ratio_null = likelihood_ratio(
            self.map_null,
            self.map,
            self.obs_spectra,
            self.theory_spectra,
            self.leakage_spectra_BB,
            self.rel_dust_spectra,
            self.dof,
            self.l_min,
            self.l_max,
            self.bin_width,
            self.l_T_cutoff_idx,
        )
        return dist.cdf(ratio_null)

    @property
    def LR_fiducial(self):
        '''use Wilks' theorem as approximation'''
        dist = chi2(1)
        ratio_fiducial = likelihood_ratio(
            self.map_fiducial,
            self.map,
            self.obs_spectra,
            self.theory_spectra,
            self.leakage_spectra_BB,
            self.rel_dust_spectra,
            self.dof,
            self.l_min,
            self.l_max,
            self.bin_width,
            self.l_T_cutoff_idx,
        )
        return dist.cdf(ratio_fiducial)

    def emcee_reg(
        self,
        nwalkers=1000,
        n_run_initial=100,
        n_run=1000,
        scale=3.6 / 180. * np.pi,
        scale_sigma_sq=1.e-8,
        ndim=6,
    ):
        p0 = np.zeros((nwalkers, ndim))
        p0[:, :3] = np.random.uniform(0.5, 2, size=(nwalkers, 3))
        p0[:, 3] = np.random.normal(scale=scale, size=nwalkers)
        p0[:, 4] = np.random.normal(scale=scale_sigma_sq, size=nwalkers)
        p0[:, 5] = np.random.normal(scale=DUST_AMP_STD, size=nwalkers) + DUST_AMP
        assert np.all([prior(p) for p in p0])

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_likelihood,
            args=(
                self.obs_spectra,
                self.theory_spectra,
                self.leakage_spectra_BB,
                self.rel_dust_spectra,
                self.dof,
                self.l_min,
                self.l_max,
                self.bin_width,
                self.l_T_cutoff_idx,
            ),
        )

        pos, prob, state = sampler.run_mcmc(p0, n_run_initial)
        sampler.reset()

        _ = sampler.run_mcmc(pos, n_run)
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), file=sys.stderr)

        thetas = sampler.flatchain
        return thetas

    @property
    def thetas(self):
        if getattr(self, '_thetas', None) is None:
            self._thetas = self.emcee_reg()
        return self._thetas

    @staticmethod
    def theta_to_frame(thetas):
        df_theta = pd.DataFrame(thetas, columns=('A_BB', 'b_T', 'b_E', 'theta', 'sigma_sq', 'A_dust'))
        df_theta['a_T'] = df_theta.b_T.apply(np.reciprocal)
        df_theta['a_E'] = df_theta.b_E.apply(np.reciprocal)
        df_theta['theta-degree'] = df_theta.theta * (180. / np.pi)
        df_theta['sigma_sq-arcmin'] = df_theta.sigma_sq * (10800. / np.pi)**2
        df_theta['eta'] = df_theta.a_E / df_theta.a_T
        return df_theta

    @property
    def df_thetas(self):
        return self.theta_to_frame(self.thetas)

    def plot_mcmc(self, n_sample=1000, kind='pair'):
        df_plot = self.df_thetas[['a_T', 'eta', 'A_BB', 'theta-degree', 'sigma_sq-arcmin', 'A_dust']]
        if n_sample is not None:
            df_plot = df_plot.sample(n_sample)
        df_plot.columns = [r'$g$', r'$\eta$', r'$A_{BB}$', r'$\theta^\circ$', r'$\sigma^2$', r'$A_{dust}$']

        if kind == 'pair':
            return sns.pairplot(df_plot, diag_kind="kde")
        elif kind == 'grid':
            g = sns.PairGrid(df_plot)
            g.map_diag(sns.distplot)
            g.map_offdiag(sns.kdeplot, n_levels=4)
            return g
        else:
            raise ValueError(f'Invalid kind {kind}: can only be pair/grid.')

    def mcmc_ci(self):
        # TODO
        pte_null = (self.df_thetas.A_BB > 0.).mean()
        pte_fiducial = (self.df_thetas.A_BB > 1.).mean()

        # 90% Confidence interval and the MAP
        ci = self.df_thetas.A_BB.quantile(0.05), self.map[0], self.df_thetas.A_BB.quantile(0.95)

        # 90% Confidence Interval that $A_{BB}$ is smaller than this
        ci_up = self.df_thetas.A_BB.quantile(0.9)
        std = self.df_thetas.A_BB.std()
        return pte_null, pte_fiducial, ci, ci_up, std

    # use the MCMC chain to reconstruct the spectra and estimate the associated error-bars

    def err_mcmc_to_spectra(self):
        '''calculate error from MCMC chain, transform real spectra in `spectra`

        calculate the error from MCMC chain and bind it to spectra

        also transform the spectra according to the MAP thetas

        Warning: this modify self.spectra in-place, and should only be done once
        '''
        err_mcmc = getattr(self.spectra, 'err_mcmc', None)
        if err_mcmc is not None:
            raise RuntimeError('err_mcmc already calculated. This should not be run more than once!')

        thetas_map = self.map
        Cl = np.ascontiguousarray(self.obs_spectra.real)
        Cl_map = inverse_transform_subtract_foreground(thetas_map, Cl, self.rel_dust_spectra, self.l_min, self.l_max, self.bin_width)
        err_mcmc = err_likelihood(self.thetas, Cl, Cl_map, self.rel_dust_spectra, self.l_min, self.l_max, self.bin_width)

        self.spectra.transform_real(thetas_map, rel_dust_spectra=self.rel_dust_spectra)
        self.spectra.err_mcmc = from_spectra_matrix(err_mcmc, self.spectra.spectra)[:, None, None, :]
        return self.spectra

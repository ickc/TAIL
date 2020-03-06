from .container import NullSpectra, BinWidth

import sys
from functools import partial
import psutil

import numpy as np
import pandas as pd

from numba import jit, prange, int64

from numba import float64 as float_
from numpy import float64 as float__
# from numba import int32 as int_
from numpy import int32 as int__

from scipy.stats import ks_2samp, kstest
from scipy.special import erfc, gammainc, gamma, gammaincc
from scipy.integrate import quad

import matplotlib.pyplot as plt
import seaborn as sns

from dautil.util import map_parallel

NCPU = psutil.cpu_count(logical=False)


@jit(float_[:, :, :, ::1](float_[:, :, :, ::1]), nopython=True, nogil=True, cache=True)
def random_choice_sim(null):
    '''randomly '''
    result = np.empty_like(null)

    # first is real
    result[:, 0] = null[:, 0]

    # rest is sim
    n = null.shape[1] - 1
    idxs = np.random.choice(n, n) + 1
    result[:, 1:] = null[:, idxs]

    return result


def vectorize_0_1(func):
    '''vectorize `func` along 0, 1 axes'''
    @jit(float_[:, ::1](float_[:, :, :, ::1]), nopython=True, nogil=True)
    def aggregate_0_1(array):
        m, n, _, _ = array.shape
        result = np.empty((m, n), dtype=float__)

        for i in range(m):
            for j in range(n):
                result[i, j] = func(array[i, j, :, :])
        return result

    return aggregate_0_1


def vectorize_0_2_3(func):
    '''vectorize `func` along 0, 2, 3 axes'''

    @jit(float_[:, :, ::1](float_[:, :, :, ::1]), nopython=True, nogil=True)
    def aggregate(array):
        m, n, o, p = array.shape

        # reorder for contiguous
        temp = np.empty((m, o * p, n), dtype=float__)
        for i in range(m):
            temp[i] = array[i].reshape((n, o * p)).T
        temp = temp.reshape((m, o, p, n))

        result = np.empty((m, o, p), dtype=float__)
        for i in range(m):
            for j in range(o):
                for k in range(p):
                    result[i, j, k] = func(temp[i, j, k, :])
        return result

    return aggregate


@vectorize_0_2_3
@jit(nopython=True, nogil=True, cache=True)
def std_vectorize_0_2_3(x):
    '''take std around theoretical expectation'''
    return np.sqrt(np.square(x).mean())


@jit(float_[:, :, :, ::1](float_[:, :, :, ::1]), nopython=True, nogil=True, cache=True)
def null_to_chi(null):
    m, _, k, l = null.shape
    return null / std_vectorize_0_2_3(np.ascontiguousarray(null[:, 1:, :, :])).reshape((m, 1, k, l))

# 5 statistics #################################################################


@vectorize_0_1
@jit(float_(float_[:, :]), nopython=True, nogil=True, cache=True)
def Y_1(chi):
    return np.abs(chi.sum())


Y_2 = vectorize_0_1(np.max)


@vectorize_0_1
@jit(float_(float_[:, :]), nopython=True, nogil=True, cache=True)
def Y_3(chi_sq):
    return chi_sq.sum(axis=1).max()


@vectorize_0_1
@jit(float_(float_[:, :]), nopython=True, nogil=True, cache=True)
def Y_4(chi_sq):
    return np.asfortranarray(chi_sq).sum(axis=0).max()


Y_5 = vectorize_0_1(np.sum)


@jit(float_[::1](float_[:, ::1]), nopython=True, nogil=True, cache=True)
def G(array):
    m, n = array.shape

    temp = np.zeros(m, dtype=int__)

    for i in range(m):
        real = array[i, 0]
        for j in range(1, n):
            if array[i, j] > real:
                temp[i] += 1

    result = temp.astype(float__)
    result /= n - 1  # drop real
    return result


@jit(float_[:, ::1](float_[:, :, :, ::1]), nopython=True, nogil=True, cache=True)
def get_all_stats(null):
    m = null.shape[0]
    result = np.empty((5, m), dtype=float__)

    chi = null_to_chi(null)
    chi_sq = np.square(chi)

    result[0] = G(Y_1(chi))
    result[1] = G(Y_2(chi_sq))
    result[2] = G(Y_3(chi_sq))
    result[3] = G(Y_4(chi_sq))
    result[4] = G(Y_5(chi_sq))
    return result

# Holm Bonferroni Method #######################################################


@jit(float_(float_[:]), nopython=True, nogil=True, cache=True)
def holm_bonferroni_critical_alpha(ps):
    idxs = np.argsort(ps)

    # rank is the inverse of argsort
    # also equiv to argsort of argsort
    # see https://groups.google.com/forum/#!topic/numpy/ht5HSfnYeO0
    n = ps.size
    ranks = np.empty(n)
    ranks[idxs] = np.arange(n)

    return np.min(ps * (ps.size - ranks))


@jit(float_[:](float_[:, :, :, ::1]), nopython=True, nogil=True, cache=True)
def get_critical_alpha(null):
    ps = get_all_stats(null)

    result = np.empty(2, dtype=float__)
    # BB, EB, EE
    result[0] = holm_bonferroni_critical_alpha(ps[:, :3].flatten())
    result[1] = holm_bonferroni_critical_alpha(ps.flatten())
    return result


@jit(float_[:, ::1](float_[:, :, :, ::1], int64), nopython=True, nogil=True, parallel=True, cache=True)
def get_critical_alpha_bootstrap(null, N):
    result = np.empty((N, 2), dtype=float__)
    for i in prange(N):
        result[i] = get_critical_alpha(random_choice_sim(null))
    return result

# KS-test ######################################################################


def get_ks_p(null):
    chi = null_to_chi(null)
    real = chi[:, 0, :, :].flatten()
    sim = chi[:, 1:, :, :].flatten()
    return ks_2samp(real, sim).pvalue


def get_ks_p_bootstrap(null, *args):
    return get_ks_p(random_choice_sim(null))


def get_ks_p_bootstrap_N(null, N):
    return np.array(map_parallel(partial(get_ks_p_bootstrap, null), range(N), processes=NCPU), dtype=float__)

# Exact statistics #############################################################


def p_1(x, N):
    return erfc(np.abs(x) * np.sqrt(N * 0.5))


def p_chi_sq_max(x, n_b, n_n):

    def f(u):
        return gammainc(0.5 * n_b, 0.5 * u)**(n_n - 1) * np.exp(-0.5 * u) * u**(0.5 * n_b - 1.)

    return n_n / (2.**(0.5 * n_b) * gamma(0.5 * n_b)) * quad(f, x, np.inf)[0]


def p_2(x, N):
    return p_chi_sq_max(x, 1, N)


p_3 = p_chi_sq_max


def p_4(x, n_b, n_n):
    return p_chi_sq_max(x, n_n, n_b)


def p_5(x, N):
    return gammaincc(0.5 * N, 0.5 * x)


def get_all_stats_iid(chi):
    '''
    chi.shape == n_spectra, n_null_split, n_bin
    '''
    _, n_null_split, n_b = chi.shape
    N = n_null_split * n_b

    chi_sq = np.square(chi)

    m = chi.shape[0]
    result = np.empty((m, 5), dtype=float__)
    # go over chi for p_1 first
    for i in range(m):
        row = chi[i]
        result[i, 0] = p_1(row.mean(), N)
    # go over chi_sq
    for i in range(m):
        row = chi_sq[i]
        result[i, 1] = p_2(row.max(), N)
        result[i, 2] = p_3(row.sum(axis=1).max(), n_b, n_null_split)
        result[i, 3] = p_4(row.sum(axis=0).max(), n_b, n_null_split)
        result[i, 4] = p_5(row.sum(), N)

    return result

# KS-test on p-value comparing to uniform dist #################################


@jit(float_[::1](float_[:, :, :, ::1]), nopython=True, nogil=True, cache=True)
def get_ps_uniform(null):
    '''compute p-values for KS test, 1D output
    '''
    m, n, o, p = null.shape

    chi_abs = np.abs(null_to_chi(null))

    chi_abs_reshaped = np.ascontiguousarray(chi_abs.reshape(m * n, o * p).T).reshape(o * p * m, n)
    return G(chi_abs_reshaped).flatten()


def get_ks_p_uniform(null):
    '''perform KS test of p-values (comparing to uniform)
    per spectra, combined by Holm Bonferroni Method
    '''
    return kstest(get_ps_uniform(null), 'uniform').pvalue


def get_ks_p_uniform_bootstrap(null, *args):
    return get_ks_p_uniform(random_choice_sim(null))


def get_ks_p_uniform_bootstrap_N(null, N):
    return np.array(map_parallel(partial(get_ks_p_uniform_bootstrap, null), range(N), processes=20), dtype=float__)


class ComputeNullStats(BinWidth):
    '''prepare Spectra to compute null-statistics
    '''

    def __init__(
        self,
        spectra, null_split,
        l_min, l_max, bin_width,
        map_cases,
        Cl_null,
        n_bootstrap=10000,
        alpha=0.05,
    ):
        self.spectra = spectra
        self.null_split = null_split
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = bin_width
        self.map_cases = map_cases
        self.Cl_null = Cl_null

        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    @classmethod
    def load(cls, null_spectra: NullSpectra, map_cases=('realmaps', 'simmaps_TTEEBB_noise')):
        # reorder them as ('spectra', 'n', 'null_split', 'l')
        # and concat real and sim s.t. n=0 is real

        Cl_null = np.ascontiguousarray(np.concatenate(
            [
                getattr(null_spectra, map_case)[:, :, 0, :, 0, :].transpose((0, 3, 1, 2))
                for map_case in map_cases
            ],
            axis=1,
        ))
        return cls(
            null_spectra.spectra, null_spectra.null_split,
            null_spectra.l_min, null_spectra.l_max, null_spectra.bin_width,
            map_cases,
            Cl_null,
        )

    @property
    def chi(self):
        return null_to_chi(self.Cl_null)

    @property
    def plot_chi(self):
        fig, axes = plt.subplots(nrows=len(self.spectra))
        chi = self.chi
        for i, spectrum in enumerate(self.spectra):
            sns.distplot(chi[i, 0].flatten(), ax=axes[i], label=f'real, {spectrum}')
            sns.distplot(chi[i, 1:].flatten(), ax=axes[i], label=f'sim, {spectrum}')
        return fig

    @property
    def all_stats_iid(self):
        chi_real = np.ascontiguousarray(self.chi[:, 0, :, :])
        get_all_stats_iid(chi_real)
        return get_all_stats_iid(chi_real)

    @property
    def all_stats_iid_to_frame(self):
        '''all 5 statistics defined for all spectra in DataFrame
        '''
        df = pd.DataFrame(self.all_stats_iid, index=self.spectra, columns=[f'Y{i}' for i in range(1, 6)])
        df.index.name = 'spectra'
        df.columns.name = 'stats'
        return df

    @property
    def plot_all_stats_iid(self):
        return sns.heatmap(self.all_stats_iid_to_frame)

    @property
    def all_stats(self):
        '''all 5 statistics defined for all spectra
        '''
        return get_all_stats(self.Cl_null)

    @property
    def all_stats_to_frame(self):
        '''all 5 statistics defined for all spectra in DataFrame
        '''
        df = pd.DataFrame(self.all_stats.T, index=self.spectra, columns=[f'Y{i}' for i in range(1, 6)])
        df.index.name = 'spectra'
        df.columns.name = 'stats'
        return df

    @property
    def plot_all_stats(self):
        return sns.heatmap(self.all_stats_to_frame)

    @property
    def plot_all_stats_iid_rel_err(self):
        df = self.all_stats_to_frame
        df_iid = self.all_stats_iid_to_frame
        df_rel = np.abs((df - df_iid) / df)
        return sns.heatmap(df_rel)

    @property
    def critical_alpha(self):
        '''critical alpha level (FWER) using Holm–Bonferroni method

        return an array of alphas, first value for 15 numbers from 5 statistics
        from EE, EB, BB only, second for all 30 from all spectra
        '''
        return get_critical_alpha(self.Cl_null)

    @property
    def __critical_alpha_bootstrap__(self):
        '''estimate distribution of critical_alpha by bootstrap method

        Set no. of bootstrap by setting `self.n_bootstrap`.

        Run this again to resample.
        '''
        return get_critical_alpha_bootstrap(self.Cl_null, self.n_bootstrap)

    @property
    def critical_alpha_bootstrap(self):
        cache = getattr(self, '_critical_alpha_bootstrap', None)
        if cache is None:
            print(f"Running {self.n_bootstrap} bootstrap...", file=sys.stderr)
            cache = self.__critical_alpha_bootstrap__
            self._critical_alpha_bootstrap = cache
        return cache

    @property
    def plot_critical_alpha_bootstrap(self):
        ps = self.critical_alpha_bootstrap
        sns.distplot(ps[:, 0])
        plt.title(r'$\alpha_c$, BB, EB, EE')
        plt.show()
        sns.distplot(ps[:, 1])
        plt.title(r'$\alpha_c$, all')
        plt.show()

    @property
    def critical_alpha_median(self):
        return np.median(self.critical_alpha_bootstrap, axis=0)

    @property
    def critical_alpha_p_value(self):
        return (self.critical_alpha_bootstrap > self.alpha).mean(axis=0)

    @property
    def ks(self):
        '''perform ks-test on chi
        return an array of alphas, first value from EE, EB, BB only, second from all spectra
        '''
        return np.array([get_ks_p(self.Cl_null[:3]), get_ks_p(self.Cl_null)])

    @property
    def __ks_bootstrap__(self):
        return [
            get_ks_p_bootstrap_N(self.Cl_null[:3], self.n_bootstrap),
            get_ks_p_bootstrap_N(self.Cl_null, self.n_bootstrap),
        ]

    @property
    def ks_bootstrap(self):
        cache = getattr(self, '_ks_bootstrap', None)
        if cache is None:
            print(f"Running {self.n_bootstrap} bootstrap...", file=sys.stderr)
            cache = self.__ks_bootstrap__
            self._ks_bootstrap = cache
        return cache

    @property
    def plot_ks_bootstrap(self):
        temp = self.ks_bootstrap
        ps_ks_eb = temp[0]
        sns.distplot(ps_ks_eb)
        plt.title(r'KS $p$, BB, EB, EE')
        plt.show()
        ps_ks_eb = temp[1]
        sns.distplot(ps_ks_eb)
        plt.title(r'KS $p$, BB, EB, EE, TB, TE, TT')
        plt.show()

    @property
    def ks_p_value(self):
        ks_bootstraps = self.ks_bootstrap
        return [(ks_bootstrap > self.alpha).mean() for ks_bootstrap in ks_bootstraps]

    @property
    def ks_each(self):
        return np.array([get_ks_p(self.Cl_null[i:i + 1]) for i in range(len(self.spectra))])

    @property
    def ks_holm_bonferroni(self):
        return holm_bonferroni_critical_alpha(self.ks_each)

    @property
    def ks_uniform(self):
        '''perform ks-test on p-values
        return an array of alphas, first value from EE, EB, BB only, second from all spectra
        '''
        return np.array([get_ks_p_uniform(self.Cl_null[:3]), get_ks_p_uniform(self.Cl_null)])

    @property
    def plot_ks_uniform(self):
        sns.distplot(get_ps_uniform(self.Cl_null[:3]))
        plt.title(r'KS uniform, BB, EB, EE')
        plt.show()
        sns.distplot(get_ps_uniform(self.Cl_null))
        plt.title(r'KS uniform, BB, EB, EE, TB, TE, TT')
        plt.show()

    @property
    def __ks_uniform_bootstrap__(self):
        return [
            get_ks_p_uniform_bootstrap_N(self.Cl_null[:3], self.n_bootstrap),
            get_ks_p_uniform_bootstrap_N(self.Cl_null, self.n_bootstrap),
        ]

    @property
    def ks_uniform_bootstrap(self):
        cache = getattr(self, '_ks_uniform_bootstrap', None)
        if cache is None:
            print(f"Running {self.n_bootstrap} bootstrap...", file=sys.stderr)
            cache = self.__ks_uniform_bootstrap__
            self._ks_uniform_bootstrap = cache
        return cache

    @property
    def plot_ks_uniform_bootstrap(self):
        temp = self.ks_uniform_bootstrap
        ps_ks_eb = temp[0]
        sns.distplot(ps_ks_eb)
        plt.title(r'KS $p$, BB, EB, EE')
        plt.show()
        ps_ks_eb = temp[1]
        sns.distplot(ps_ks_eb)
        plt.title(r'KS $p$, BB, EB, EE, TB, TE, TT')
        plt.show()

    @property
    def ks_uniform_p_value(self):
        ks_bootstraps = self.ks_uniform_bootstrap
        return [(ks_bootstrap > self.alpha).mean() for ks_bootstrap in ks_bootstraps]

    @property
    def is_passing(self):
        '''determine if it passes null-tests

        Passing criteria: p-value from EE, EB, BB exceeds alpha=0.05
        '''
        return self.critical_alpha[0] >= self.alpha

    @property
    def summary(self):
        print(f'Critical alpha is {self.critical_alpha}. Probability of critical alpha not smaller than {self.alpha} is {self.critical_alpha_p_value}.\n\t*where first no. includes only EE, EB, BB and second includes all.')

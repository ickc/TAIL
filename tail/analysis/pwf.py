'''Pixel window functions related

These module provides

`p_l_integrate`: most accurate

`p_l_approx`: approxmate but fast

`get_p_l_fitted`: higher order function that return a function that is
fast but more accurate than `p_l_approx`.
'''

import sys

from numba import jit, vectorize, float64, int32, int64
from scipy.integrate import quad
from scipy.optimize import minimize
import numpy as np


@vectorize([float64(int32, float64), float64(int64, float64)], nopython=True)
def p_l_approx(l, Δ):
    '''calculate pixel window function approximately

    param int l: l, can be array
    param float Δ: angle in radian, can be array

    to calculate values for array of l and Δ, broadcast them into different dimension before passing in. e.g.

    >>> p_l(l[None, :], Δ[:, None])

    c.f. eq. 92 in
    Wu, J. H. P., Balbi, A., Borrill, J., Ferreira, P. G., Hanany, S., Jaffe, A. H., et al. (2001). Asymmetric Beams in Cosmic Microwave Background Anisotropy Experiments. The Astrophysical Journal Supplement Series, 132(1), 1–17. http://doi.org/10.1086/318947
    '''
    lΔ = l * Δ
    return np.exp(-lΔ**2.04 / 18.1) * (1 - 2.72e-2 * (lΔ * lΔ))


@jit(float64(float64, float64), nopython=True)
def _integrand(φ, lΔ):
    '''integrand of PWF

    c.f. eq. 91 in
    Wu, J. H. P., Balbi, A., Borrill, J., Ferreira, P. G., Hanany, S., Jaffe, A. H., et al. (2001). Asymmetric Beams in Cosmic Microwave Background Anisotropy Experiments. The Astrophysical Journal Supplement Series, 132(1), 1–17. http://doi.org/10.1086/318947
    '''
    cos_φ = np.cos(φ)
    sin_φ = np.sin(φ)

    lΔ_2 = 0.5 * lΔ
    temp = np.sin(lΔ_2 * cos_φ) * np.sin(lΔ_2 * sin_φ) / (lΔ * lΔ * cos_φ * sin_φ)
    return temp * temp


def p_l_integrate(ls, Δ):
    '''calculate pixel window function by integrate directly

    param int l: l, must be array
    param float Δ: angle in radian, cannot be array

    return
    ------

    PWF in 2d-array, where the columns are the errors
    '''
    n = ls.shape[0]
    res = np.empty((n, 2), dtype=np.float64)
    lsΔ = ls * Δ
    for i in range(ls.size):
        res[i] = 1. if lsΔ[i] == 0 else quad(_integrand, 0., 2 * np.pi, args=(lsΔ[i],))
    return res * (8. / np.pi)


@jit(nopython=True, nogil=True)
def p_l_fitting_func(lΔ, α, var, k, β):
    return np.exp(-lΔ**α / var) * (1. - k * lΔ**β)


def p_l_fitting(l, Δ):
    '''compute the best fit parameters for `p_l_fitting_func`

    lΔ forms dimensionless parameter for input
    the only way it affects the output is the
    range of values it includes in minimizing error
    '''

    p_l_exact = p_l_integrate(l, Δ)
    p_l = p_l_exact[:, 0]
    lΔ = l * Δ

    @jit(float64(float64[:]), nopython=True, nogil=True)
    def l2_loss(args):
        '''minimize chi-square
        lΔ, p_l are global
        '''
        α, var, k, β = args
        fitted = p_l_fitting_func(lΔ, α, var, k, β)
        return np.sum(np.square((fitted - p_l) / p_l))

    # the input guess is `p_l_approx` from above
    res = minimize(l2_loss, np.array([2.04, 18.1, 2.72e-2, 2.])).x

    rel_err_new = p_l_fitting_func(lΔ, *res) / p_l - 1.
    rel_err_old = p_l_approx(l, Δ) / p_l - 1.

    print(f'The fitted parameters over the given range is {res}, with maximum relative error over the given range as {np.abs(rel_err_new).max()}, comparing to {np.abs(rel_err_old).max()} from `p_l_approx`.', file=sys.stderr)

    return res


def get_p_l_fitted(l, Δ):
    '''return a PWF function with fitted parameters over the range
    specified by `l`, `Δ`
    '''
    α, var, k, β = p_l_fitting(l, Δ)

    @vectorize([float64(int32, float64), float64(int64, float64)], nopython=True)
    def p_l_fitted(l, Δ):
        '''calculate pixel window function approximately

        param int l: l, can be array
        param float Δ: angle in radian, can be array

        to calculate values for array of l and Δ, broadcast them into different dimension before passing in. e.g.

        >>> p_l(l[None, :], Δ[:, None])
        '''
        lΔ = l * Δ
        return p_l_fitting_func(lΔ, α, var, k, β)

    return p_l_fitted

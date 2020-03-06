from uncertainties import ufloat
import numpy as np
from numba import jit

from astropy.constants import h, c, k_B


_c = c.value
_h = h.value
_k = k_B.value
T_CMB_0 = 2.73

# These are from BK15
# dust charateristics
beta = ufloat(1.59, 0.11)
T = 19.6
# dust template
nu_0 = 353e9
# it was an asymetric error-bar in BK15
# choose worst case for Normal here
A_0 = ufloat(4.6, 1.1)

# our freq.
nu = 150e9


@jit(nopython=True, nogil=True, cache=True)
def planck_dist(nu, T):
    '''Planck's Law/distribution'''
    return 2. * _h * nu**3 / (_c * _c * np.expm1(_h * nu / (_k * T)))


@jit(nopython=True, nogil=True, cache=True)
def dB_dT(nu, T):
    '''derivative of Plank's distribution over Temperature
    
    Often, `dB_dT(nu, T_CMB_0)` is used as a conversion factor
    from T_CMB to Brightness, vice versa.
    '''
    x = _h * nu / (_k * T)
    return 2. * _k * (nu * x / _c)**2 * np.exp(x) / np.expm1(x)**2


@jit(nopython=True, nogil=True, cache=True)
def rel_dB_dT(nu, T):
    x = _h * nu / (_k * T)
    return nu**4 * np.exp(x) / np.expm1(x)**2


def planck_dist_modified(nu, T, beta):
    '''Modified Planck's Law/distribution'''
    return 2. * _h * np.power(nu, beta + 3.) / (_c * _c * np.expm1(_h * nu / (_k * T)))


def rel_planck_dist_modified(nu, T, beta):
    '''Modified Planck's Law/distribution'''
    return np.power(nu, beta + 3.) / np.expm1(_h * nu / (_k * T))


def scale_dust(nu, nu_0, A_0, T, beta):
    '''
    convert T_CMB to B
    scale using the dust template
    convert B back to T_CMB
    '''
    # scaling factor from T_CMB to Brightness conversion
    scale_CMB = rel_dB_dT(nu_0, T_CMB_0) / rel_dB_dT(nu, T_CMB_0)
    return A_0 * (
        scale_CMB *
        (rel_planck_dist_modified(nu, T, beta) / rel_planck_dist_modified(nu_0, T, beta))
    )**2


A = scale_dust(nu, nu_0, A_0, T, beta)

DUST_AMP = A.nominal_value
DUST_AMP_STD = A.std_dev

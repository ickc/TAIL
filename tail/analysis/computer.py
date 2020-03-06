
from .container import Spectra, FilterTransfer
from .loader import AllInput
from .helper import matrix_reduce, matrix_reduce_row, matrix_reduce_col, uncertainties_to_real, uncertainties_to_rel

import sys
from functools import partial

import numpy as np
from numba import jit, float64, complex128, int32
from numpy import linalg

# helper functions for ComputeFilterTransfer ###################################


def solve_left(pseudos, Ms, B_2, ths):
    '''
    pseudo-spectra
    ('spectra', 'null_split', 'sub_split', 'l')
    mode-coupling
    ('spectra', 'l', 'l')
    beam
    ('l')
    theory
    ('spectra', 'l')
    '''
    n_spectra, _, _, n_l = pseudos.shape
    # ('spectra', 'l')
    temp = np.einsum('ijk,k,ik->ij', Ms, B_2, ths)
    return pseudos / temp.reshape(n_spectra, 1, 1, n_l)


def solve_right(pseudos, Ms, B_2, ths):
    n_spectra, _, _, n_l = pseudos.shape
    # ('spectra', 'null_split', 'sub_split', 'l')
    # K's condition is worse than M so better solve with M first
    temp = linalg.solve(Ms.reshape(n_spectra, 1, 1, n_l, n_l), pseudos)
    return temp / (B_2.reshape(1, 1, 1, n_l) * ths.reshape(n_spectra, 1, 1, n_l))


@jit([complex128[:, ::1](complex128[:, ::1], int32), float64[:, ::1](float64[:, ::1], int32)], nopython=True, nogil=True)
def interp_F(Fs_binned, bin_width):
    m, n_b = Fs_binned.shape
    n = n_b * bin_width

    x_common = np.arange(n)
    x_common_binned = np.arange(0.5 * (bin_width - 1), 0.5 * ((2 * (n_b + 1) - 1) * bin_width - 1), bin_width)

    res = np.empty((m, n), dtype=Fs_binned.dtype)
    for i in range(m):
        res[i] = np.interp(x_common, x_common_binned, Fs_binned[i])
    return res


def solve_right_binned(bin_width, pseudos, Ms, B_2, ths):
    '''
    actually it is unsure how the binning should work
    e.g. similar to K_ll to K_bb binning, binning should occur
    with the combination of (Ms * B_2 * ths)
    but then it introduce yet another band power window function
    comparing to the one used in spectra

    Ignoring this for now since we are not multiplying F on the right
    in our analysis
    '''
    pseudos_binned = matrix_reduce_col(pseudos, bin_width)
    Ms_binned = matrix_reduce(Ms, bin_width)
    B_2_binned = matrix_reduce_col(B_2, bin_width)
    ths_binned = matrix_reduce_col(ths, bin_width)

    Fs_binned = solve_right(pseudos_binned, Ms_binned, B_2_binned, ths_binned)
    n_spectra, n_null_split, n_sub_split, n_b = Fs_binned.shape
    Fs = interp_F(Fs_binned.reshape(n_spectra * n_null_split * n_sub_split, n_b), bin_width)
    return Fs.reshape(pseudos.shape)


class ComputeFilterTransfer(object):

    def __init__(
        self,
        pseudos, Ms, B_2, ths,
        cases,
        null_split,
        full, norm, l_min, l_max
    ):
        self.pseudos = pseudos
        self.Ms = Ms
        self.B_2 = B_2
        self.ths = ths
        self.cases = cases
        self.null_split = null_split
        self.full = full
        self.norm = norm
        self.l_min = l_min
        self.l_max = l_max

    @classmethod
    def load(
        cls, filter_transfer_input: AllInput,
        norm='median', l_min=50, l_max=4300
    ):
        '''
        '''
        # pseudo-spectra
        # ('spectra', 'null_split', 'sub_split', 'l')
        pseudos = filter_transfer_input.pseudo_spectra.transform_for_filter_transfer(l_min, l_max, filter_transfer_input.cases)

        # mode-coupling
        # ('spectra', 'l', 'l')
        Ms = filter_transfer_input.mode_coupling.transform_for_filter_transfer(l_min, l_max, filter_transfer_input.cases, norm=norm)

        # In the calculation of filter transfer functions, the "exact" beam
        # is used in the simulation. # hence the error of the beam does not
        # propagate to the error of filter transfer functions
        # i.e. beam here is treated as exact
        B_2 = filter_transfer_input.beam_spectra.squared_interp(l_min, l_max)

        # ('spectra', 'l')
        ths = filter_transfer_input.theory_spectra.packing(l_min, l_max, filter_transfer_input.cases)

        return cls(
            pseudos, Ms, B_2, ths,
            tuple(filter_transfer_input.cases.keys()),
            filter_transfer_input.pseudo_spectra.null_split,
            filter_transfer_input.full, norm, l_min, l_max
        )

    def solve(self, right=False, bin_width=None):
        if right:
            if bin_width is None:
                solver = solve_right
            else:
                solver = partial(solve_right_binned, bin_width)
        else:
            solver = solve_left

        Fs = solver(self.pseudos, self.Ms, self.B_2, self.ths)
        return FilterTransfer(
            Fs,
            self.full,
            self.null_split,
            self.norm,
            right,
            self.cases,
            self.l_min, self.l_max, bin_width
        )


class ComputeSpectra(object):
    '''prepare arrays for computing spectra from its inputs

    pseudo_spectra: 6d-arrays: ('spectra', 'null_split', 'sub_split', 'l', 'sub_spectra', 'n')
    filter_transfer: 4d-arrays: ('spectra', 'null_split', 'sub_split', 'l') # same as rel_err
    K_ll: ('spectra', 'l', 'l') # same as Ms
    or if right: ('spectra', 'null_split', 'sub_split', 'l', 'l')

    Compute K_ll for next step
    '''

    def __init__(
        self,
        K_ll,
        # meta
        sub_spectra,
        spectra,
        l_min,
        l_max,
        null_split=['full'],
        full=True,
        right=False,
        rel_err=None,
        **kwargs
    ):
        self.K_ll = K_ll
        self.sub_spectra = sub_spectra
        self.spectra = spectra
        self.l_min = l_min
        self.l_max = l_max
        self.null_split = null_split
        self.full = full
        self.right = right
        self.rel_err = rel_err

        self.map_cases = tuple(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load(cls, spectra_input: AllInput, uncertainties=False):
        '''
        similar to `.filter_transfer.ComputeFilterTransfer.load`
        '''
        # meta
        filter_transfer = spectra_input.filter_transfer
        l_min = filter_transfer.l_min
        l_max = filter_transfer.l_max
        norm = filter_transfer.norm

        pseudo_spectra = spectra_input.pseudo_spectra
        spectra = pseudo_spectra.spectra
        pseudo_spectra_dict_l_sliced = pseudo_spectra.pseudo_spectra_dict_l_sliced(l_min, l_max)

        Fs = filter_transfer.solve(spectra, uncertainties=uncertainties)
        B_2 = spectra_input.beam_spectra.squared_interp(l_min, l_max, uncertainties=uncertainties)

        # since it is computationally infeasible to propagate the error through the MASTER equations,
        # only keep track of relative errors approximately
        if uncertainties:
            rel_err = uncertainties_to_rel(Fs * B_2)
            Fs = uncertainties_to_real(Fs)
            B_2 = uncertainties_to_real(B_2)
        else:
            rel_err = None

        # mode-coupling
        Ms = spectra_input.mode_coupling.transform_for_filter_transfer(l_min, l_max, spectra, norm=norm)

        right = filter_transfer.right
        if right:
            K_ll = Ms[:, None, None, :, :] * (Fs[:, :, :, None, :] * B_2[None, None, None, None, :])
        else:
            Fs_reshaped = Fs[:, :, :, :, None, None]
            # filter transfer corrected pseudo-spectra
            pseudo_spectra_dict_l_sliced = {
                spectrum: value / Fs_reshaped
                for spectrum, value in pseudo_spectra_dict_l_sliced.items()
            }
            K_ll = Ms * B_2

        return cls(
            K_ll,
            # meta
            pseudo_spectra.sub_spectra,
            spectra,
            l_min,
            l_max,
            null_split=pseudo_spectra.null_split,
            full=spectra_input.full,
            right=right,
            rel_err=rel_err,
            **pseudo_spectra_dict_l_sliced
        )

    def solve(
        self,
        bin_width: int,
        l_min: int = None,
        l_max: int = None,
        l_boundary=600,
        return_w=None
    ) -> Spectra:
        if return_w is None:
            # normally only interested in w in full-spectra
            return_w = self.full
            print(f'auto setting return_w to {return_w}', file=sys.stderr)
        if l_min is None:
            l_min = l_boundary - (l_boundary - self.l_min) // bin_width * bin_width
            print(f'auto l-min at {l_min}', file=sys.stderr)
        if l_max is None:
            l_max = l_boundary + (self.l_max - l_boundary) // bin_width * bin_width
            print(f'auto l-max at {l_max}', file=sys.stderr)

        l_rel_min = l_min - self.l_min
        l_rel_max = l_max - self.l_min
        K_ll = (
            self.K_ll[:, :, :, l_rel_min:l_rel_max][:, :, :, :, l_rel_min:l_rel_max]
        ) if self.right else (
            self.K_ll[:, l_rel_min:l_rel_max][:, :, l_rel_min:l_rel_max]
        )

        K_bb = matrix_reduce(K_ll, bin_width)
        if not self.right:
            K_bb_reshaped = K_bb[:, None, None, :, :]

        res = dict()
        # can speed up if packing map_cases in one array and do solve in 1 pass
        # but given there's only a few map_cases don't bother here
        for map_case in self.map_cases:
            Cl = getattr(self, map_case)[:, :, :, l_rel_min:l_rel_max]

            # reshape to what linalg.solve needed
            shape = Cl.shape
            n_spectra, n_null_split, n_sub_split, n_l, n_sub_spectra, n = shape
            Cl = Cl.reshape(n_spectra, n_null_split, n_sub_split, n_l, n_sub_spectra * n)

            Cb = matrix_reduce_row(Cl, bin_width)
            Cb_est = linalg.solve(K_bb, Cb) if self.right else linalg.solve(K_bb_reshaped, Cb)

            # reshape back
            Cb_est = Cb_est.reshape(list(Cb_est.shape[:-1]) + [n_sub_spectra, n])
            res[map_case] = Cb_est

        # if not self.right then w_bl is the same between full and null
        if return_w:
            K_bl = matrix_reduce_row(K_ll, bin_width)
            w_bl = linalg.solve(K_bb, K_bl)
        else:
            w_bl = None

        # ('spectra', 'null_split', 'sub_split', 'l')
        rel_err = None if self.rel_err is None else matrix_reduce_col(self.rel_err[:, :, :, l_rel_min:l_rel_max], bin_width)

        return Spectra(
            self.spectra, self.null_split, self.sub_spectra,
            l_min, l_max, bin_width,
            w_bl=w_bl,
            rel_err=rel_err,
            right=self.right,
            **res
        )

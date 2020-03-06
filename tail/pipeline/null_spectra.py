#!/usr/bin/env python

import argparse
from glob import iglob as glob
from functools import partial
from itertools import starmap

import h5py
import pandas as pd
import numpy as np
from numba import jit
from scipy import linalg

from dautil.util import map_parallel

IDX = pd.IndexSlice

_REAL_TO_SIM = 1e-12

__version__ = '0.2'


def get_inputs(df, df_theory, df_filter, f_modecoupling, l_common, spectrum, sub_split, null_split, compute_nl=False):
    '''``spectrum``: one of TT, EE, BB, TE, TE, EB
    '''

    df_Cl_sim = df.loc[IDX['Cl', spectrum, sub_split, null_split, :], l_common]
    df_Cl_sim.reset_index(level=(0, 1, 2, 3), inplace=True, drop=True)

    if compute_nl:
        df_Nl_sim = df.loc[IDX['Nl', spectrum, sub_split, null_split, :], l_common]
        df_Nl_sim.reset_index(level=(0, 1, 2, 3), inplace=True, drop=True)

        df_Cl_sim += 1.j * df_Nl_sim
        del df_Nl_sim

    M = f_modecoupling['{0}{0}'.format(spectrum)][:][l_common][:, l_common]

    # auto-spectra
    if spectrum[0] == spectrum[1]:
        F = df_filter.loc[IDX['{0}{0}'.format(spectrum), sub_split, null_split], l_common].values.real
    # cross-spectra
    else:
        F = df_filter.loc[IDX['{0}{0}{0}{0}'.format(spectrum[0]), sub_split, null_split], l_common].values.real
        F *= df_filter.loc[IDX['{0}{0}{0}{0}'.format(spectrum[1]), sub_split, null_split], l_common].values.real
        F = np.sqrt(F)

    Cl_th = df_theory.loc[l_common, spectrum].values if spectrum in ('TT', 'EE', 'BB', 'TE') else None
    return M, np.ascontiguousarray(F), df_Cl_sim, Cl_th


def matrix_reduce(M, b):
    '''reduce the resolution of the matrix M by bin-width ``b``
    and devided by ``b``
    '''
    if b == 1:
        return M
    else:
        m, n = M.shape
        return np.einsum('ijkl->ik', M.reshape(m // b, b, n // b, b)) / b


def matrix_reduce_row(M, b):
    '''reduce the resolution of the matrix M by bin-width ``b``
    and devided by ``b``
    '''
    if b == 1:
        return M
    else:
        m, n = M.shape
        return np.einsum('ijk->ik', M.reshape(m // b, b, n)) / b


def spectra_reduce(Cls, b):
    '''reduce the resolution across the second axis by bin-width ``b``
    and devided by ``b``
    '''
    if b == 1:
        return Cls
    else:
        m, n = Cls.shape
        return np.einsum('ijk->ij', Cls.reshape(m, n // b, b)) / b


@jit(nopython=True, nogil=True)
def solve_K_l(M, F, B_2):
    return F.reshape(-1, 1) * M * B_2.reshape(1, -1)


def solve(K_l, Cl_sim, bin_width, return_w=False):
    K_b = matrix_reduce(K_l, bin_width)
    Cl_sim_binned = spectra_reduce(Cl_sim, bin_width)
    Cl = linalg.solve(K_b, Cl_sim_binned.T).T
    if return_w:
        P_bl_K_ll = matrix_reduce_row(K_l, bin_width)
        w_bl = linalg.solve(K_b, P_bl_K_ll)
        return Cl, w_bl
    else:
        return Cl


def solve_spectra(df_pseudo, df_theory, df_filter, f_modecoupling, B_2, bin_width, l_common, l_common_binned, spectrum, sub_split, null_split, compute_nl=False, return_w=False):
    M, F, df_Cl_sim, Cl_th = get_inputs(df_pseudo, df_theory, df_filter, f_modecoupling, l_common, spectrum, sub_split, null_split, compute_nl=compute_nl)

    K_l = solve_K_l(M, F, B_2)
    del M, F

    if Cl_th is not None:
        df_Cl_sim.loc[IDX['theory', 0], :] = K_l @ Cl_th
        del Cl_th

    res = solve(K_l, df_Cl_sim.values, bin_width, return_w=return_w)
    del K_l

    if return_w:
        Cl, w_bl = res

    else:
        Cl = res
    
    df = pd.DataFrame(
        Cl,
        index=df_Cl_sim.index,
        columns=l_common_binned
    )

    if return_w:
        return df, w_bl
    else:
        return df


def main(pseudospectra, theory, filter_transfer, modecoupling, beam, bin_width, l_min, l_max, processes, compute_nl=False, return_w=False, l_boundary=600, l_lower=50, l_upper=4300):
    '''
    `l_boundary`: the pivot point of l bins. e.g. given a bin_width, 600 is always the boundary between 2 bins.
    `l_lower`: lowest l we trust. e.g. 50
    `l_upper`: highest l we trust. e.g. 4300 due to F_TT can becomes negative above that.
    '''
    if l_min < 0:
        l_min = l_boundary - (l_boundary - l_lower) // bin_width * bin_width
        print(f'auto l-min at {l_min}')
    if l_max < 0:
        l_max = l_boundary + (l_upper - l_boundary) // bin_width * bin_width
        print(f'auto l-max at {l_max}')

    l_common = np.arange(l_min, l_max)
    l_common_binned = l_common.reshape(-1, bin_width).mean(axis=1)

    # Reading
    df_beam = pd.read_hdf(beam)['all']
    B_2 = np.square(np.interp(l_common, df_beam.index, df_beam.values.real))
    del df_beam

    df = pd.concat(
        map_parallel(pd.read_hdf, glob(pseudospectra), mode='multithreading', processes=processes)
    )

    df_theory = pd.read_hdf(theory) * _REAL_TO_SIM
    df_filter = pd.read_hdf(filter_transfer)

    # mapping all cases
    index = pd.MultiIndex.from_product(
        (
            df.index.levels[1],
            df.index.levels[2],
            df.index.levels[3]
        ),
        names=df.index.names[1:4]
    )
    if return_w:
        index_w = pd.MultiIndex.from_product(
            (
                df.index.levels[1],
                df.index.levels[2],
                df.index.levels[3],
                l_common_binned
            ),
            names=df.index.names[1:4] + ['b']
        )

    with h5py.File(modecoupling, 'r') as f:
        res = list(starmap(
            partial(solve_spectra, df, df_theory, df_filter, f, B_2, bin_width, l_common, l_common_binned, compute_nl=compute_nl, return_w=return_w),
            index
        ))

    if return_w:
        Cls, w_bls = list(map(list, zip(*res)))
    else:
        Cls = res

    df_spectra = pd.concat(
        Cls,
        keys=index
    )
    del l_common_binned, B_2, df, df_theory, df_filter

    df_spectra.index.names = index.names + df_spectra.index.names[-2:]
    del index

    df_spectra.sort_index(inplace=True)

    if return_w:
        df_window = pd.DataFrame(
            np.concatenate(w_bls, axis=0),
            index=index_w,
            columns=l_common
        )
        df_window.sort_index(inplace=True)
        return df_spectra, df_window
    else:
        return df_spectra


def cli():
    parser = argparse.ArgumentParser(description="Obtain final spectra from pseudo-spectra, mode-coupling, filter transfer function, etc.")

    parser.add_argument('-o', '--output', required=True,
                        help='Output HDF5 filename.')

    parser.add_argument('--modecoupling', required=True,
                        help='Input modecoupling HDF5 file.')
    parser.add_argument('--pseudospectra', required=True,
                        help='Input pseudospectra DataFrame in HDF5. Can be a glob pattern.')
    parser.add_argument('--theory', required=True,
                        help='Input theory DataFrame in HDF5.')
    parser.add_argument('--beam', required=True,
                        help='Input beam DataFrame in HDF5.')
    parser.add_argument('--filter-transfer', required=True,
                        help='Input filter transfer function in HDF5.')
    parser.add_argument('--compute-nl', action='store_true',
                        help='If specified, compute Nl and store as the imaginary part of the spectra DataFrame.')
    parser.add_argument('--return-w', action='store_true',
                        help='If specified, return the band-power window function w.')

    parser.add_argument('--bin-width', default=300, type=int,
                        help='bin width. Default: 300')
    parser.add_argument('--l-min', type=int, default=-1,
                        help='Minimum l. Default: auto-calculated. Lowest value: 2.')
    parser.add_argument('--l-max', type=int, default=-1,
                        help='maximum l (exclusive). Default: auto-calculated. Highest value: 4901')

    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='No. of parallel processes.')

    args = parser.parse_args()

    res = main(
        args.pseudospectra,
        args.theory,
        args.filter_transfer,
        args.modecoupling,
        args.beam,
        args.bin_width,
        args.l_min,
        args.l_max,
        args.processes,
        compute_nl=args.compute_nl,
        return_w=args.return_w
    )

    if args.return_w:
        df_spectra, df_window = res
    else:
        df_spectra = res

    df_spectra.to_hdf(
        args.output,
        'spectra',
        mode='w',
        format='table',
        complevel=args.compress_level,
        fletcher32=True,
    )
    if args.return_w:
        df_window.to_hdf(
            args.output,
            'window',
            mode='a',
            format='table',
            complevel=args.compress_level,
            fletcher32=True,
        )


if __name__ == "__main__":
    cli()

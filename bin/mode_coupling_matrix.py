#!/usr/bin/env python

import argparse
import timeit
from functools import partial

import h5py
import numpy as np

from tail.modecoupling import calc_M

__version__ = 0.1

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}


def calc_M_timed(mask1, mask2, Mtype, pure, pixel_size=0.0005817764173314432, l_max=3000, dl=10, normalize=False):
    '''a thin wrapper of calc_M just to show wall clock time used in this function
    '''
    time = timeit.default_timer()
    M = calc_M(mask1, mask2, Mtype, pure, pixel_size=pixel_size, l_max=l_max, dl=dl, normalize=normalize)
    time -= timeit.default_timer()
    print('{},{},{}'.format(Mtype, pure, -time))
    return M


def main(
        input,
        output,
        dl,
        lmax,
        pixel_size,
        processes,
        compress_level,
        no_mpi
    ):
    # pixel size in radian
    pixel_size = pixel_size * np.pi / 10800.

    with h5py.File(input, 'r') as f:
        tmask = f['t'][:]
        pmask = f['p'][:]

    _calc_M = partial(calc_M_timed, pixel_size=pixel_size, l_max=lmax, dl=dl)
    _iter = (
        (pmask, None, 'EBEB', 'hybrid'),  # 1406s
        (pmask, None, 'EBEB', 'pseudo'),  # 1404s
        (pmask, None, 'BBBB', 'hybrid'),  # 1010s
        (pmask, tmask, 'TBTB', 'hybrid'), # 1010s
        (pmask, tmask, 'TETE', 'hybrid'), #  414s
        (pmask, None, 'EEEE', 'hybrid'),  #  412s
        (tmask, None, 'TTTT', 'hybrid'),  #   57s
    )

    if not no_mpi:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            Ms = executor.starmap(
                _calc_M,
                _iter
            )
    else:
        from dautil.util import starmap_parallel
        Ms = starmap_parallel(
            _calc_M,
            _iter,
            processes=processes
        )

    with h5py.File(output, libver='latest') as f:
        for M, name in zip(
            Ms,
            (
                'EBEB',
                'EBEB_pseudo',
                'BBBB',
                'TBTB',
                'TETE',
                'EEEE',
                'TTTT',
            )
        ):
            f.create_dataset(
                name,
                data=M,
                compression_opts=compress_level,
                **H5_CREATE_KW
            )


def cli():
    parser = argparse.ArgumentParser('Accurate mode-coupling matrices calculation.')

    parser.add_argument('input', help='Input HDF5 filename, holding temperature and polarization masks.')
    parser.add_argument('-o', '--output', required=True, help='output HDF5 filename.')

    parser.add_argument('--dl', action='store', dest='dl',
                        type=int, help='binning for ell', default=10)
    parser.add_argument('--lmax', action='store', type=int,
                        help='maximum multipole moment', default=3000)
    parser.add_argument('--pixel-size', type=float, default=2.,
                        help='pixel size in arcminute. Default: 2.')

    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='No. of parallel processes.')
    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('--no-mpi', action='store_true',
                        help='If specified, do not use MPI. This option is primarily for debug only.')

    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.dl,
        args.lmax,
        args.pixel_size,
        args.processes,
        args.compress_level,
        args.no_mpi
    )


if __name__ == '__main__':
    cli()

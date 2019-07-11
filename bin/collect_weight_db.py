#!/usr/bin/env python

import argparse
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numba import jit

from dautil.date_time import strptime_np
from tail.window import zero_pad_ratio
from tail.util import effective_sky

__version__ = '0.3'


def get_greg(path):
    return strptime_np(path.parent.parent.name)


def get_idx(path):
    greg = get_greg(path)

    temp = path.stem.split('_')
    null_split = '_'.join(temp[1:-1])
    sub_split = temp[-1]
    return null_split, sub_split, greg


def get_weight_fsky(tmask_apo, pmask_apo, pixel_size, path):
    with h5py.File(path, 'r') as f:
        tweights = f['w0'][:]
        pweights = f['w4'][:]
    tweight = tweights.sum()
    pweight = pweights.sum()
    tmask = tmask_apo * zero_pad_ratio(tweights > 0.)
    pmask = pmask_apo * zero_pad_ratio(pweights > 0.)
    t_fsky = effective_sky(tmask, pixel_size)
    p_fsky = effective_sky(pmask, pixel_size)
    return tweight, pweight, t_fsky, p_fsky


def main(
    basedir,
    glob,
    mask_path,
    pixel_size,
    no_mpi,
    processes,
    full
):
    all_paths = list(basedir.glob(glob))
    pixel_size *= np.pi / 10800.
    with h5py.File(mask_path, 'r') as f2:
        tmask = f2['t'][:]
        pmask = f2['p'][:]

    _get_weight_fsky = partial(get_weight_fsky, tmask, pmask, pixel_size)
    if not no_mpi:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            data = executor.map(
                _get_weight_fsky,
                all_paths
            )
    else:
        from dautil.util import map_parallel
        data = map_parallel(
            _get_weight_fsky,
            all_paths,
            processes=processes
        )

    index = pd.Index(
        list(map(get_greg, all_paths)),
        name='date'
    ) if full else pd.MultiIndex.from_tuples(
        list(map(get_idx, all_paths)),
        names=('nullsplit', 'subsplit', 'date')
    )

    df = pd.DataFrame(
        data=data,
        index=index,
        columns=('tweight', 'pweight', 't_fsky', 'p_fsky')
    )

    df.sort_index(inplace=True)
    return df


def cli():
    parser = argparse.ArgumentParser(description="Collect temperature and polarization stat. such as total weight and f-sky.")

    parser.add_argument('basedir',
                        help="Input base directory of weights.")
    parser.add_argument('-m', '--mask', required=True,
                        help='HDF5 file holding the apodization masks.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output filename for DataFrame in HDF5.')
    parser.add_argument('--glob', default='????????_??????/coadd/realmap_*_?.hdf5',
                        help='glob pattern. Default: ????????_??????/coadd/realmap_*_?.hdf5')
    parser.add_argument('--full', action='store_true',
                        help='If specified, assume full map intead.')

    parser.add_argument('--no-mpi', action='store_true',
                        help='If specified, do not use MPI.')
    parser.add_argument('-p', '--processes', default=1, type=int,
                        help='No. of multiprocessing processes when --no-mpi is used.')
    parser.add_argument('--pixel-size', default=2., type=float)

    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9. Note: 9 comparing to 4 adds about 29 percent of compute time.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    df = main(
        Path(args.basedir),
        args.glob,
        args.mask,
        args.pixel_size,
        args.no_mpi,
        args.processes,
        args.full
    )
    df.to_hdf(
        args.output,
        'df',
        format='table',
        complevel=args.compress_level,
        fletcher32=True
    )


if __name__ == "__main__":
    cli()

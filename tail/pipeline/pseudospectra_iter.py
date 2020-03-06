#!/usr/bin/env python

import argparse
import os
from functools import partial
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from dautil.IO import makedirs
from dautil.IO.stdio import redirect_stdout_stderr
from dautil.IO.timeit_IO import timeit_IO
from tail.pseudospectra_IO import get_pseudospectra_IO
from tail.util import norm_fft

__version__ = '0.1'


# decorate it to redirect stdout/err and timeit
get_pseudospectra_IO = timeit_IO(redirect_stdout_stderr(get_pseudospectra_IO))


def get_pseudospectra_wrap(
    df_weights,
    df_nauto,
    tmask,
    pmask,
    tnorm,
    pnorm,
    pixel_size,
    lmax,
    n_x,
    h5_weight_basedir,
    h5_signal_basedir,
    weight_name,
    processes,
    compress_level,
    outdir,
    mapcase,
    name,
    isim,
    nullsplit,
    redirect=False,
    mode='multithreading',
):
    makedirs(outdir)
    filename = outdir / '{}_{}'.format(name, nullsplit)
    outpath = filename.with_suffix('.hdf5')
    if outpath.is_file():
        print('{} existed. Skipping...'.format(outpath))
        return
    df = get_pseudospectra_IO(
        df_weights,
        df_nauto,
        tmask,
        pmask,
        tnorm,
        pnorm,
        pixel_size,
        lmax,
        n_x,
        h5_weight_basedir,
        h5_signal_basedir,
        weight_name,
        processes,
        mapcase,
        name,
        isim,
        nullsplit,
        stdout=filename.with_suffix('.out') if redirect else None,
        stderr=filename.with_suffix('.err') if redirect else None,
        timeit_filename=filename.with_suffix('.time'),
        mode=mode,
    )
    df.to_hdf(
        outpath,
        'pseudospectra',
        format='table',
        complevel=compress_level,
        fletcher32=True
    )
    return


def main(
    outbasedir,
    h5_signal_basedir,

    name,
    mapcase,
    nreal,
    isim,
    mapsplits,

    processes=1,
    no_mpi=False,
    redirect=True,
    compress_level=9,

    weight_name='realmap',
    lmax=3000,
    pixel_size=2.,

    # these must be filled in
    selectionpath=None,
    maskpath=None,
    weightdir=None,
    mode='multithreading',
):
    df_weights = pd.read_hdf(selectionpath, 'weights')
    df_nauto = pd.read_hdf(selectionpath, 'n_auto')
    with h5py.File(str(maskpath), 'r') as f:
        tmask = f['t'][:]
        pmask = f['p'][:]
    tnorm = norm_fft(tmask)
    pnorm = norm_fft(pmask)
    n_x = tmask.shape[0]
    pixel_size *= np.pi / 10800.

    _get_pseudospectra_IO = partial(
        get_pseudospectra_wrap,
        df_weights,
        df_nauto,
        tmask,
        pmask,
        tnorm,
        pnorm,
        pixel_size,
        lmax,
        n_x,
        weightdir,
        h5_signal_basedir,
        weight_name,
        processes,
        compress_level,
        outbasedir / mapcase,
        mapcase,
        name if nreal is None else name + '{0:03}'.format(isim),
        isim,
        redirect=redirect,
        mode=mode,
    )

    if no_mpi:
        from dautil.util import map_parallel
        map_parallel(_get_pseudospectra_IO, mapsplits, processes=processes)
    else:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            executor.map(_get_pseudospectra_IO, mapsplits)


def cli():
    parser = argparse.ArgumentParser(description="New pseudospectra pipeline.")

    parser.add_argument('yaml', type=argparse.FileType('r'), default=sys.stdin,
                        help='Import YAML config file.')
    parser.add_argument('-i', '--basedir', required=True,
                        help="Output directory.")
    parser.add_argument('-o', '--outbasedir', required=True,
                        help="Output directory.")
    parser.add_argument('--map-case', required=True,
                        help="e.g. null_simmaps_TTBB.")
    parser.add_argument('--nreal',
                        help="The total no. of simulations in the end.")
    parser.add_argument('--isim', type=int, default=0,
                        help="The i-th simulation.")
    parser.add_argument('--name', default='realmap',
                        help="Name, either realmap or sim.")

    parser.add_argument('--no-mpi', action='store_true',
                        help='If specified, do not use MPI.')
    parser.add_argument('-p', '--processes', default=1, type=int,
                        help='No. of Multiprocessing processes per MPI process')
    parser.add_argument('--mode', default='multithreading',
                        help='mode to use for parallel IO per process. Default: multithreading. Other choice: multiprocessing.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('--redirect', action='store_true',
                        help='If specified, redirect stdout and stderr to sidecars of output files. Default: True if --no-mpi is not specified.')
    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')

    args = parser.parse_args()

    if not args.no_mpi:
        args.redirect = True

    dict_ = yaml.load(args.yaml)

    pseudo = dict_['pseudo']
    # only mapsplits here
    try:
        mapsplits = dict_['pseudo_iter']['mapsplits']
    except KeyError:
        mapsplits = None
    del dict_

    main(
        Path(args.outbasedir),
        Path(args.basedir),

        args.name,
        args.map_case,
        args.nreal,
        args.isim,
        mapsplits,

        processes=args.processes,
        no_mpi=args.no_mpi,
        redirect=args.redirect,
        compress_level=args.compress_level,

        weight_name=pseudo.get('weight_name', 'realmap'),
        lmax=pseudo.get('lmax', 3000),
        pixel_size=pseudo.get('pixel_size', 2),
        selectionpath=Path(os.path.expandvars(pseudo['selectionpath'])),
        maskpath=Path(os.path.expandvars(pseudo['maskpath'])),
        weightdir=Path(os.path.expandvars(pseudo['weightdir'])),
        mode=args.mode,
    )


if __name__ == "__main__":
    cli()

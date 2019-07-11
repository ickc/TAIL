#!/usr/bin/env python

import argparse
import os
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from dautil.IO import makedirs
from dautil.IO.stdio import redirect_stdout_stderr
from dautil.IO.timeit_IO import timeit_IO
from tail.pseudospectra_IO import get_pseudospectra_IO
from tail.util import norm_fft

IDX = pd.IndexSlice

__version__ = '0.2'


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
    filename = os.path.join(outdir, name if nullsplit is None else '_'.join((name, nullsplit)))
    outpath = filename + '.hdf5'
    if os.path.isfile(outpath):
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
        stdout=filename + '.out' if redirect else None,
        stderr=filename + '.err' if redirect else None,
        timeit_filename=filename + '.time',
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


def get_iter(
    outbasedir,
    outsubdirs,
    mapcases,
    nreals,
    names,
    mapsplits
):
    return [
        (
            os.path.join(outbasedir, outsubdir),
            mapcase,
            name if nreal is None else name + '{0:03}'.format(isim),
            isim,
            mapsplit
        )
        for mapcase, nreal, name, outsubdir in zip(mapcases, nreals, names, outsubdirs)
        for isim in ((0,) if nreal is None else range(nreal))
        for mapsplit in ([None] if mapsplits is None else mapsplits)
    ]


def main(
    outbasedir,
    h5_selection_path,
    h5_mask_path,
    h5_weight_basedir,
    h5_signal_basedir,
    processes=1,
    no_mpi=False,
    redirect=True,
    compress_level=9,
    mode='multithreading',
    # from pseudo_iter
    weight_name='realmap',
    lmax=3000,
    pixel_size=2.,
    # mapsplits is None means full map
    mapsplits=None,
    # these must be filled in
    # o is outdir
    o=None,
    name=None,
    mapcase=None,
    nreal=None,
):
    df_weights = pd.read_hdf(h5_selection_path, 'weights')
    # full map has no split so no n_auto
    df_nauto = None if mapsplits is None else pd.read_hdf(h5_selection_path, 'n_auto')

    with h5py.File(str(h5_mask_path), 'r') as f:
        tmask = f['t'][:]
        pmask = f['p'][:]
    tnorm = norm_fft(tmask)
    pnorm = norm_fft(pmask)
    n_x = tmask.shape[0]
    pixel_size *= np.pi / 10800.

    iter_ = get_iter(
        outbasedir,
        o,
        mapcase,
        nreal,
        name,
        mapsplits
    )
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
        h5_weight_basedir,
        h5_signal_basedir,
        weight_name,
        processes,
        compress_level,
        redirect=redirect,
        mode=mode,
    )

    if not no_mpi:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            executor.starmap(_get_pseudospectra_IO, iter_)
    else:
        from dautil.util import starmap_parallel
        starmap_parallel(_get_pseudospectra_IO, iter_, processes=processes)


def cli():  # TODO
    parser = argparse.ArgumentParser(description="New pseudospectra pipeline.")

    parser.add_argument('yaml',
                        help='Import YAML config file.')
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

    with open(args.yaml, 'r') as f:
        dict_ = yaml.load(f)
    pseudo = {key: Path(os.path.expandvars(value)) for key, value in dict_['pseudo'].items()}
    pseudo_iter = dict_['pseudo_iter']

    main(
        pseudo['outbasedir'],
        pseudo['selectionpath'],
        pseudo['maskpath'],
        pseudo['weightdir'],
        pseudo['basedir'],
        processes=args.processes,
        no_mpi=args.no_mpi,
        redirect=args.redirect,
        compress_level=args.compress_level,
        mode=args.mode,
        **pseudo_iter
    )


if __name__ == "__main__":
    cli()

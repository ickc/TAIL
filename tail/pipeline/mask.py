#!/usr/bin/env python

import argparse
from functools import partial, reduce
from pathlib import Path

import h5py
import numpy as np

from dautil.util import map_parallel
from tail.window import standard_mask, zero_pad_ratio
from tail.pointsource import mask_from_map

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


def get_dataset(dataset, path):
    with h5py.File(str(path), 'r') as f:
        return f[dataset][:]


def get_mask(dataset, path):
    return get_dataset(dataset, path) > 0.


def get_apodized_mask(basedir, dataset):
    final_weight = get_dataset(dataset, basedir / 'coadd/realmap.hdf5')

    final_weight *= reduce(
        np.logical_and,
        map(
            partial(get_mask, dataset),
            basedir.glob('????????_??????/coadd/realmap.hdf5')
        )
    )

    return standard_mask(final_weight)


def main(
    basedir,
    output,
    no_ps,
    catalog,
    pixel_size,
    width,
    source,
    compress_level):
    basedir = Path(basedir)

    temp_mask, pol_mask = map_parallel(partial(get_apodized_mask, basedir), ('w0', 'w4'), p=2)
    if not no_ps:
        n = int(round(width / pixel_size * 60.)) + 1
        shape = np.array((n, n))

        # pixel size in radian
        pixel_size = pixel_size * np.pi / 10800.
        # defined in libmap.py
        xmax = width * np.pi / 360.
        xmin = -xmax
        ymax = xmax
        ymin = xmin
        ps_mask = zero_pad_ratio(mask_from_map(
            source,
            shape,
            xmin, xmax, ymin, ymax,
            catalog
        ))
        temp_mask *= ps_mask
        pol_mask *= ps_mask

    with h5py.File(output, libver='latest') as f:
        # tmask
        f.create_dataset('t',
            data=temp_mask,
            compression_opts=compress_level,
            **H5_CREATE_KW
        )
        # pmask
        f.create_dataset('p',
            data=pol_mask,
            compression_opts=compress_level,
            **H5_CREATE_KW
        )


def cli():
    parser = argparse.ArgumentParser(description="Generate masks from final weights.")

    parser.add_argument('basedir',
                        help='Base input directory of maps.')
    parser.add_argument('-o', '--output', required=True,
                        help='Base output directory of maps.')

    parser.add_argument('--no-ps', action='store_true', help='If specified, do not apply point-source mask.')

    parser.add_argument('--catalog', help='Pandas DataFrame in HDF5 holding point-source catalog.')
    parser.add_argument('--pixel-size', default=2.)
    parser.add_argument('--width', default=45.)
    parser.add_argument('--source', default='BICEP')

    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    main(
        args.basedir,
        args.output,
        args.no_ps,
        args.catalog,
        args.pixel_size,
        args.width,
        args.source,
        args.compress_level
    )


if __name__ == "__main__":
    cli()

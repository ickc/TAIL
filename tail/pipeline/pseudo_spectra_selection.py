#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd

__version__ = '0.2'

IDX = pd.IndexSlice


def get_broad_map(fsky, fraction=0.05):
    '''``fsky``: a Series object holding ``fsky``
    This is an updated selection criteria after optimization.
    '''
    max_effective_sky = fsky.max()
    return ((1. - fraction) * max_effective_sky < fsky)


def get_good_map(weight, factor=10.):
    med = weight.median()
    return (med / factor < weight) & (weight < med * factor)


def sort_by_auto_first(df, inplace=False):
    '''``df``: has MultiIndex of level (nullsplit, subsplit, date)
    sort per nullsplit such that date with both subsplit 0/1 comes first
    '''
    if not inplace:
        df = df.copy()

    # drop subsplit to column
    # index levels: nullsplit, date
    df.reset_index(level=1, inplace=True)

    # create a column that per nullsplit per date shows the negative count of number of subsplits
    df['neg_count'] = df.pweight.groupby(level=(0, 1)).count().map(lambda x: -x)

    # drop date to column
    # index levels: nullsplit
    df.reset_index(level=1, inplace=True)
    # add neg_count to index
    # index levels: nullsplit, neg_count
    df.set_index('neg_count', inplace=True, append=True)
    # add subsplit to index
    # index levels: nullsplit, neg_count, subsplit
    df.set_index('subsplit', inplace=True, append=True)
    # add date to index
    # index levels: nullsplit, neg_count, subsplit, date
    df.set_index('date', inplace=True, append=True)

    # sort by index: nullsplit, neg_count, subsplit, date
    # such that per nullsplit, the one with more count comes first
    df.sort_index(inplace=True)

    # drop neg_count to column
    # index levels: nullsplit, subsplit, date, same as original
    df.reset_index(level=1, inplace=True)

    if not inplace:
        return df


def get_n_auto(df):
    '''per nullsplit, count no. of dates that has both subsplits.
    '''
    result = df.loc[IDX[:, '0', :], 'pweight'][df['neg_count'] == -2].groupby(level=0).count()
    result.name = 'n_auto'
    return result


def test_sort_by_auto_first(df_sorted, df_auto, inplace=False):
    import warnings

    if not inplace:
        df_sorted = df_sorted.copy()
    # for convenience in getting the date below
    df_sorted.reset_index(2, inplace=True)

    nullsplits = df_sorted.index.levels[0].values

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    for nullsplit in nullsplits:
        n_auto = df_auto[nullsplit]
        greg_0 = df_sorted.loc[IDX[nullsplit, '0'], 'date'].values
        greg_1 = df_sorted.loc[IDX[nullsplit, '1'], 'date'].values
        np.testing.assert_array_equal(greg_0[:n_auto], greg_1[:n_auto])


def main_full(df):
    # selection criteria
    df['is_broad_map'] = get_broad_map(df.p_fsky)
    df['is_good_map'] = get_good_map(df.pweight)
    # filter by selection criteria
    df = df[df.is_broad_map & df.is_good_map]
    return df


def main(df):
    # selection criteria
    df['is_broad_map'] = pd.concat([get_broad_map(group.p_fsky) for __, group in df.groupby(level=(0, 1))])
    df['is_good_map'] = pd.concat([get_good_map(group.pweight) for __, group in df.groupby(level=(0, 1))])
    # filter by selection criteria
    df = df[df.is_broad_map & df.is_good_map]

    # sort per nullsplit such that date with both subsplit 0/1 comes first
    sort_by_auto_first(df, inplace=True)
    df_auto = get_n_auto(df)
    # testing the function are doing what it claims
    test_sort_by_auto_first(df, df_auto)
    return df, df_auto


def cli():
    parser = argparse.ArgumentParser(description="Pseudo-spectra selection, sorting, and get n_auto.")

    parser.add_argument('db_path',
                        help="Input DataFrame holding weights of all null-maps.")
    parser.add_argument('-o', '--output', required=True,
                        help='Output filename for DataFrame in HDF5.')
    parser.add_argument('--full', action='store_true',
                        help='If specified, assume full map intead.')

    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    df = pd.read_hdf(args.db_path)

    if args.full:
        df = main_full(df)
    else:
        df, df_auto = main(df)
        df_auto.to_hdf(
            args.output,
            'n_auto',
            format='table',
            complevel=args.compress_level,
            fletcher32=True
        )

    df[['tweight', 'pweight']].to_hdf(
        args.output,
        'weights',
        format='table',
        complevel=args.compress_level,
        fletcher32=True
    )


if __name__ == "__main__":
    cli()

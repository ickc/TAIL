import timeit
from functools import partial

import h5py
import numpy as np
import pandas as pd

from dautil.date_time import strttime_np
from dautil.util import map_parallel, mask, zero_padding
from tail.pseudospectra import pseudospectra_auto_all, pseudospectra_cross_all
from tail.qu_to_eb import iqu_to_teb

IDX = pd.IndexSlice


def get_IQU_IO(n_x, h5_weight_basedir, h5_signal_basedir, weight_name, nullsplit, subsplit, mapcase, name, date):

    def _get(path, dataset):
        with h5py.File(str(path), 'r') as f:
            return f[dataset][:]

    def solve(hit, num, denom):
        array = mask(np.divide, idxs=(0, 1))(num, denom, mask=hit)
        return zero_padding(array, (n_x, n_x))

    greg = strttime_np(date)
    h5_signal_path = h5_signal_basedir / mapcase / greg / 'coadd' / (name + '.hdf5' if nullsplit is None else '{}_{}_{}.hdf5'.format(name, nullsplit, subsplit))
    h5_weight_path = h5_weight_basedir / greg / 'coadd' / (weight_name + '.hdf5' if nullsplit is None else '{}_{}_{}.hdf5'.format(weight_name, nullsplit, subsplit))
    del greg

    # get IQU
    w0 = _get(h5_weight_path, 'w0')

    hit = w0 > 0.

    I = solve(
        hit,
        _get(h5_signal_path, 'd0'),
        w0
    )
    del w0

    w4 = _get(h5_weight_path, 'w4')
    del h5_weight_path

    Q = solve(
        hit,
        _get(h5_signal_path, 'd4r'),
        w4
    )

    U = solve(
        hit,
        _get(h5_signal_path, 'd4i'),
        w4
    )
    return I, Q, U


def get_pseudospectra_IO(
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
    mode='multithreading',
):
    result = []
    teb_arrays = []
    tpweightses = []

    SUBSPLITS = ('0',) if nullsplit is None else ('0', '1')

    for subsplit in SUBSPLITS:
        temp = df_weights.reset_index() \
            if nullsplit is None else \
            df_weights.loc[IDX[nullsplit, subsplit, :]].reset_index(level=2)

        dates = temp.date.values
        tpweights = temp[['tweight', 'pweight']].values.T
        del temp
        tpweightses.append(tpweights)

        # reading IQU
        _get_IQU_IO = partial(get_IQU_IO, n_x, h5_weight_basedir, h5_signal_basedir, weight_name, nullsplit, subsplit, mapcase, name)

        time = timeit.default_timer()
        IQUs = map_parallel(_get_IQU_IO, dates, mode=mode, processes=processes)
        del dates
        time -= timeit.default_timer()
        print('IQU IO,{},{}'.format(subsplit, -time))

        # 4d-array
        # 1st dim: I/Q/U at first then T/E/B
        # 2nd dim: no. of observations
        # 3/4 dim: the actual array of I/Q/U or T/E/B
        iqu_array = np.stack(list(map(list, zip(*IQUs))))
        del IQUs

        time = timeit.default_timer()
        teb_array = iqu_to_teb(iqu_array, tmask, pmask, tnorm, pnorm, pixel_size, n_x)
        time -= timeit.default_timer()
        print('IQU to TEB,{},{}'.format(subsplit, -time))

        del iqu_array
        teb_arrays.append(teb_array)

        time = timeit.default_timer()
        t_s, t_n, e_s, e_n, b_s, b_n, te_s, te_n, tb_s, tb_n, eb_s, eb_n = pseudospectra_auto_all(
            pixel_size, lmax,
            teb_array, tpweights
        )
        del tpweights, teb_array
        time -= timeit.default_timer()
        print('pseudospectra,{},{}'.format(subsplit, -time))

        # in full map case, avoid having nan in index values
        nullsplit_value = 'full' if nullsplit is None else nullsplit
        result += [
            (('Cl', 'TT', subsplit, nullsplit_value, mapcase, isim), t_s),
            (('Nl', 'TT', subsplit, nullsplit_value, mapcase, isim), t_n),
            (('Cl', 'EE', subsplit, nullsplit_value, mapcase, isim), e_s),
            (('Nl', 'EE', subsplit, nullsplit_value, mapcase, isim), e_n),
            (('Cl', 'BB', subsplit, nullsplit_value, mapcase, isim), b_s),
            (('Nl', 'BB', subsplit, nullsplit_value, mapcase, isim), b_n),
            (('Cl', 'TE', subsplit, nullsplit_value, mapcase, isim), te_s),
            (('Nl', 'TE', subsplit, nullsplit_value, mapcase, isim), te_n),
            (('Cl', 'TB', subsplit, nullsplit_value, mapcase, isim), tb_s),
            (('Nl', 'TB', subsplit, nullsplit_value, mapcase, isim), tb_n),
            (('Cl', 'EB', subsplit, nullsplit_value, mapcase, isim), eb_s),
            (('Nl', 'EB', subsplit, nullsplit_value, mapcase, isim), eb_n)
        ]

    # cross spectra
    if nullsplit is not None:
        subsplit = '2'

        time = timeit.default_timer()

        t_s, t_n, e_s, e_n, b_s, b_n, te_s, te_n, tb_s, tb_n, eb_s, eb_n = pseudospectra_cross_all(
            pixel_size, lmax,
            teb_arrays[0], tpweightses[0], teb_arrays[1], tpweightses[1],
            df_nauto[nullsplit]
        )
        del teb_arrays, tpweightses
        time -= timeit.default_timer()
        print('pseudospectra,{},{}'.format(subsplit, -time))

        result += [
            (('Cl', 'TT', subsplit, nullsplit, mapcase, isim), t_s),
            (('Nl', 'TT', subsplit, nullsplit, mapcase, isim), t_n),
            (('Cl', 'EE', subsplit, nullsplit, mapcase, isim), e_s),
            (('Nl', 'EE', subsplit, nullsplit, mapcase, isim), e_n),
            (('Cl', 'BB', subsplit, nullsplit, mapcase, isim), b_s),
            (('Nl', 'BB', subsplit, nullsplit, mapcase, isim), b_n),
            (('Cl', 'TE', subsplit, nullsplit, mapcase, isim), te_s),
            (('Nl', 'TE', subsplit, nullsplit, mapcase, isim), te_n),
            (('Cl', 'TB', subsplit, nullsplit, mapcase, isim), tb_s),
            (('Nl', 'TB', subsplit, nullsplit, mapcase, isim), tb_n),
            (('Cl', 'EB', subsplit, nullsplit, mapcase, isim), eb_s),
            (('Nl', 'EB', subsplit, nullsplit, mapcase, isim), eb_n)
        ]

    df = pd.DataFrame.from_dict(dict(result)).T
    del result
    df.index.names = ('sub_spectra', 'spectra', 'sub_split', 'null_split', 'map_case', 'n')
    return df

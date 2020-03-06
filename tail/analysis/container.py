from pathlib import Path
import sys
import io

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from numba import jit, float64, int32, prange

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}


def get_idx(array, value):
    return np.nonzero(array == value)[0][0]


def get_idxs(array, values):
    return np.nonzero(np.logical_or.reduce([array == value for value in values]))[0]


def h5_check_before_create(f, name, data, compress_level=9):
    '''create datasets in hdf5 file object with checking for identity if exists

    if `name` exists in `f`, read the dataset and check it is identical to `data`

    handle string datasets automatically

    Note that f can be an HDF5 group
    '''
    data_array = np.asarray(data)

    # HDF5 expects `np.string_`
    if data_array.dtype.type is np.unicode_:
        data_array = data_array.astype(np.string_)

    if name in f:
        print(f'{name} exists in {f}, checking on-file data identical to the one to be written...', file=sys.stderr)
        data_f = f[name][:]
        np.testing.assert_array_equal(data_array, data_f)
    else:
        f.create_dataset(
            name,
            data=data_array,
            compression_opts=compress_level,
            **H5_CREATE_KW
        )


class FSky(object):

    methods = np.array(['naive', 'signal', 'noise'])

    def __init__(self, f_skys, cases=('p', 't'), pixel_size=np.pi / 5400.):
        self.f_skys = f_skys
        self.cases = np.asarray(cases)
        self.pixel_size = pixel_size

    def get_f_sky(self, case, method):
        return self.f_skys[
            get_idx(self.cases, case),
            get_idx(self.methods, method),
        ]

    @classmethod
    def load(cls, path: Path, pixel_size=np.pi / 5400.):
        '''
        :param Path path: path to mask in HDF5 container.
        :param float pixel_size: pixel width in radian. Default: 2 arcmin.
        '''
        from ..util import effective_sky, effective_sky_noise_dominated

        masks = Generic.load_h5(path)

        f_skys = np.empty((len(masks.cases), 3))
        for i, case in enumerate(masks.cases):
            mask = getattr(masks, case)
            # naive f_sky
            f_skys[i, 0] = effective_sky(mask.astype(np.bool), pixel_size)
            # signal-dominated
            f_skys[i, 1] = effective_sky(mask, pixel_size)
            # noise-dominated
            f_skys[i, 2] = effective_sky_noise_dominated(mask, pixel_size)
        return cls(
            f_skys,
            cases=masks.cases,
            pixel_size=pixel_size,
        )

    def to_sq_deg(self):
        return FSky(self.f_skys * (129600. / np.pi), cases=self.cases, pixel_size=self.pixel_size * (180. / np.pi))


class Generic(object):
    prefix = ''

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        temp = tuple(kwargs.keys())
        assert 'cases' not in temp
        self.cases = temp

    def __eq__(self, other):
        if not isinstance(other, Generic):
            return NotImplementedError

        for key, value in vars(self).items():
            my_data = np.asarray(value)
            try:
                other_data = np.asarray(getattr(other, key))
            except AttributeError:
                return False
            if not np.array_equal(my_data, other_data):
                return False
        return True

    @classmethod
    def load_h5(cls, path, keys=None, **kwargs):
        '''load from HDF5 file

        This assume the output written by `save` below.
        '''
        with h5py.File(path, 'r') as file:
            f = file[cls.prefix] if cls.prefix else file
            if keys is None:
                keys = f.keys()
            kwargs = {case: f[case][:] for case in keys}
        return cls(**kwargs)

    def save(self, path, compress_level=9):
        '''save object using HDF5 container

        this implement the HDF5 structure that `load_h5` expects
        '''
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, libver='latest') as f:
            for case in self.cases:
                h5_check_before_create(f, f'{self.prefix}/{case}', getattr(self, case), compress_level=compress_level)


class ModeCoupling(Generic):
    '''mode coupling matrices that read to and write from memory

    example keys are ['BBBB', 'EBEB', 'EBEB_pseudo', 'EEEE', 'TBTB', 'TETE', 'TTTT']
    '''
    prefix = 'mode_coupling'

    @staticmethod
    def l_slice(l_min, l_max):
        '''return a slice on l given min, max

        ModeCoupling is assumed to have spectra starting from l=0
        '''
        return slice(l_min, l_max)

    @classmethod
    def load_h5_legacy(cls, path, **kwargs):
        '''load from HDF5 file

        This assume the legacy HDF5 file output from `pipeline.mode_coupling_matrx`.
        '''
        with h5py.File(path, 'r') as f:
            kwargs = {case: f[case][:] for case in f}
        return cls(**kwargs)

    @classmethod
    def load(cls, path, spec='h5', **kwargs):
        '''load from file(s) according to `spec`
        '''
        method = {
            'h5_legacy': cls.load_h5_legacy,
            'h5': cls.load_h5,
        }
        return method[spec](path, **kwargs)

    def normalize_median(self, case='TTTT'):
        '''normalize all mode coupling matrices by a common ratio,
        calculated using median of sum of row of TTTT matrix

        Comparing to `normalize_row`, this preserves the "shape" by
        normalized by a common factor. The normalization constant should
        be like an integral constant that's independent.

        Default using TTTT since it is the simplest mode-coupling matrix
        '''
        assert case in self.cases
        array = getattr(self, case)
        norm_row = array.sum(axis=1)
        ratio = np.median(norm_row)

        for case in self.cases:
            temp = getattr(self, case)
            temp /= ratio
        return ratio

    def normalize_row(self):
        '''normalize all mode coupling matrices using `tail.util.normalize_row`
        '''
        from tail.util import normalize_row as normalize_func

        for case in self.cases:
            array = getattr(self, case)
            normalize_func(array)

    def normalize(self, norm='median'):
        method = {
            'median': self.normalize_median,
            'row': self.normalize_row,
        }
        return method[norm]()

    def transform_for_filter_transfer(self, l_min, l_max, cases, norm='median'):
        '''transform the array in a format needed by `.filter_transfer.ComputeFilterTransfer`
        '''
        self.normalize(norm=norm)
        l_idxs = self.l_slice(l_min, l_max)
        Ms = np.stack(
            [
                getattr(self, case * 2)[l_idxs][:, l_idxs]
                for case in cases
            ]
        )
        return Ms


class BinWidth(object):
    '''Mixin for any class that has attributes l_min, l_max, bin_width

    The distinction between l and b variants of methods is basically l
    is consecutive integers, b should be middle of l-bin.

    One example is `w_bl`, where the row b-values should be `b_range`,
    while the column l-values should be `l_range`.
    '''

    def __init__(self, l_min, l_max, bin_width):
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = bin_width
        raise NotImplementedError

    @property
    def l_range(self):
        '''get l-range
        '''
        return range(self.l_min, self.l_max)

    @property
    def l_arange(self):
        '''get l-range
        '''
        return np.arange(self.l_min, self.l_max, dtype=np.int32)

    @property
    def b_range(self):
        '''get values of middle of bins

        in case bin_width == 1, b_range == ls
        '''
        bin_width = self.bin_width
        bin_width_half = bin_width // 2
        b_min = self.l_min + bin_width_half
        b_max = self.l_max + bin_width_half
        return range(b_min, b_max, bin_width)

    @property
    def b_arange(self):
        '''get values of middle of bins

        in case bin_width == 1, b_range == ls
        '''
        bin_width = self.bin_width
        bin_width_half = bin_width // 2
        b_min = self.l_min + bin_width_half
        b_max = self.l_max + bin_width_half
        return np.arange(b_min, b_max, bin_width, dtype=np.int32)

    def l_slice(self, l_min, l_max):
        '''return a slice on l given min, max
        '''
        return slice(l_min - self.l_min, l_max - self.l_min)

    def b_slice(self, l_min, l_max):
        '''return a slice on b given min, max
        '''
        return slice((l_min - self.l_min) // self.bin_width, (l_max - self.l_min) // self.bin_width)

# helper for analytical error-bars #############################################


@jit(nopython=True, nogil=True, cache=True)
def rel_err_analytic_auto(Cl, Nl):
    '''relative analytical error bar without the dof multiplicative factor for auto-spectra
    '''
    return Cl + Nl


@jit(nopython=True, nogil=True, cache=True)
def rel_err_analytic_cross(Cl12, Cl11, Cl22, Nl11, Nl22):
    '''relative analytical error bar without the dof multiplicative factor for cross-spectra
    '''
    return np.sqrt(np.square(Cl12) + (Cl11 + Nl11) * (Cl22 + Nl22))


@jit(nopython=True, nogil=True, cache=True, parallel=True)
def rel_err_analytic_all(Cls, Nls, cross_to_auto_idxs, rel_dof=None):
    '''relative analytical error bar without the dof multiplicative factor for all spectra

    Both `Cls` and `Nls` should be 4d-array in this order:
    ('spectra', 'null_split', 'sub_split', 'l')

    `rel_dof`: if not None, then in this order: ('spectra', 'l'). Cases where `rel_dof`
    can depends on null-split isn't supported here (but easy to modify to support that case)

    return: ('spectra', 'null_split', 'sub_split', 'l')
    '''
    n_spectra, n_null_split, n_sub_split, n_l = Cls.shape
    assert n_sub_split in (1, 3)

    if rel_dof is not None:
        assert rel_dof.ndim == 2
        rel_dof_T = np.ascontiguousarray(rel_dof.T)
        # after: ('l', 'spectra')

    # trapose for locality
    Cls_T = np.ascontiguousarray(Cls.transpose((1, 3, 2, 0)))
    Nls_T = np.ascontiguousarray(Nls.transpose((1, 3, 2, 0)))
    # after: ('null_split', 'l', 'sub_split', 'spectra')

    res = np.empty_like(Cls_T)
    for i in prange(n_null_split):
        for j in range(n_l):
            if rel_dof is not None:
                rel_dof_j = rel_dof_T[j]
            Cl_ij = Cls_T[i, j]
            Nl_ij = Nls_T[i, j]
            # sub_split
            for k in range(n_sub_split):
                # auto spectra within null-splits
                if k < 2:
                    Cl_ijk = Cl_ij[k]
                    Nl_ijk = Nl_ij[k]
                    for l in range(n_spectra):
                        spectrum_idx1, spectrum_idx2 = cross_to_auto_idxs[l]
                        # auto e.g. BB_0
                        # in this case l equals them too
                        if spectrum_idx1 == spectrum_idx2:
                            Cl = Cl_ijk[l]
                            Nl = Nl_ijk[l]
                            temp = rel_err_analytic_auto(Cl, Nl)
                            if rel_dof is not None:
                                temp *= np.sqrt(2. / rel_dof_j[l])
                        # cross e.g. EB_0
                        else:
                            Cl12 = Cl_ijk[l]
                            Cl11 = Cl_ijk[spectrum_idx1]
                            Cl22 = Cl_ijk[spectrum_idx2]
                            Nl11 = Nl_ijk[spectrum_idx1]
                            Nl22 = Nl_ijk[spectrum_idx2]
                            temp = rel_err_analytic_cross(Cl12, Cl11, Cl22, Nl11, Nl22)
                            if rel_dof is not None:
                                temp *= np.power(rel_dof_j[spectrum_idx1] * rel_dof_j[spectrum_idx2], -0.25)
                        res[i, j, k, l] = temp
                # k == 2: cross-spectra between null-splits
                else:
                    for l in range(n_spectra):
                        spectrum_idx1, spectrum_idx2 = cross_to_auto_idxs[l]
                        # cross of same spectrum from 2 splits e.g. B_0 x B_1
                        # in this case l equals them too
                        if spectrum_idx1 == spectrum_idx2:
                            Cl11 = Cl_ij[0, l]
                            Cl22 = Cl_ij[1, l]
                            Cl12 = Cl_ij[2, l]
                            Nl11 = Nl_ij[0, l]
                            Nl22 = Nl_ij[1, l]
                            temp = rel_err_analytic_cross(Cl12, Cl11, Cl22, Nl11, Nl22)
                            if rel_dof is not None:
                                temp /= np.sqrt(rel_dof_j[l])
                        # cross of cross e.g. E_0 x B_1
                        else:
                            # 2 cases e.g. E_0 x B_1 and B_0 x E_1
                            # case 1:
                            Cl11 = Cl_ij[0, spectrum_idx1]
                            Cl22 = Cl_ij[1, spectrum_idx2]
                            Cl12 = Cl_ij[2, l]
                            Nl11 = Nl_ij[0, spectrum_idx1]
                            Nl22 = Nl_ij[1, spectrum_idx2]
                            temp = rel_err_analytic_cross(Cl12, Cl11, Cl22, Nl11, Nl22)
                            # case 2:
                            Cl11 = Cl_ij[0, spectrum_idx2]
                            Cl22 = Cl_ij[1, spectrum_idx1]
                            Nl11 = Nl_ij[0, spectrum_idx2]
                            Nl22 = Nl_ij[1, spectrum_idx1]
                            temp2 = rel_err_analytic_cross(Cl12, Cl11, Cl22, Nl11, Nl22)
                            # recall that in pseudo-spectra calculation (c.f. `_form_1d_cross_spectra_cross`)
                            # the 2 different crosses are averaged
                            # here we assume they are iid Normal to propagate the error
                            temp = np.sqrt(temp * temp + temp2 * temp2) * 0.5
                            if rel_dof is not None:
                                temp *= np.power(rel_dof_j[spectrum_idx1] * rel_dof_j[spectrum_idx2], -0.25)
                        res[i, j, k, l] = temp
    # back to ('spectra', 'null_split', 'sub_split', 'l')
    return np.ascontiguousarray(res.transpose((3, 0, 2, 1)))

################################################################################


class GenericSpectra(Generic, BinWidth):
    '''Essentially GenericSpectra is like anything in PseudoSpectra that's
    applicable to Spectra as well.
    '''

    names = ('spectra', 'null_split', 'sub_split', 'l', 'sub_spectra', 'n')
    meta_prefix = 'meta'
    rel_prefix = ''

    def __init__(self, spectra, null_split, sub_spectra, **kwargs):
        '''
        '''
        self.sub_spectra = sub_spectra
        self.spectra = spectra

        self.map_cases = tuple(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

        # this is hard-coded, basically the defaults of PseudoSpectra
        self.l_min = 0
        # this is n_l
        self.l_max = kwargs[self.map_cases[0]].shape[3]
        self.bin_width = 1

        self.is_full = self.get_is_full(null_split)
        self.null_split = ['full'] if self.is_full else null_split

        self.cases = ('spectra', 'sub_spectra') if self.is_full else ('spectra', 'sub_spectra', 'null_split')
        self.prefix = self.get_prefix(self.is_full)

    @staticmethod
    def get_is_full(null_split):
        ''' return True if `null_split` equals [None] or ['full']
        '''
        if len(null_split) != 1:
            return False
        temp = null_split[0]
        if temp is None or temp == 'full':
            return True
        return False

    @classmethod
    def get_prefix(cls, full):
        return '/'.join((
            'full' if full else 'null',
            cls.rel_prefix
        ))

    @classmethod
    def load_h5(cls, *args, **kwargs):
        raise NotImplementedError

    def to_frame(self, sort_index=False, ordering=[4, 0, 2, 1, 5, 3], map_case_level=4, data_frame_col_level='l'):
        '''convert to DataFrame container

        ordering: default to map from current ordering
        ('spectra', 'null_split', 'sub_split', 'l', 'sub_spectra', 'n')
        to legacy ordering:
        ('sub_spectra', 'spectra', 'sub_split', 'null_split', 'n', 'l')

        `map_case_level` ignored if ordering is None

        suggested args:

        `to_frame(sort_index=False, ordering=None)` for new ordering

        `to_frame(sort_index=True)` for legacy ordering if slowness from `sort_index`
        is not of concern
        '''
        from dautil.pandas_util import ndarray_to_series

        if ordering:
            dataframe_names = [self.names[i] for i in ordering]
        else:
            map_case_level = 0
            # self.names is tuple so casting to list make a copy
            dataframe_names = list(self.names)
        dataframe_names.insert(map_case_level, 'map_case')

        dfs = []
        for map_case in self.map_cases:
            values = getattr(self, map_case)
            shape = values.shape
            levels = [
                self.spectra,
                self.null_split,
                # historically sub_split is 0, 1, 2 in str
                list(map(str, range(shape[2]))),
                self.b_range,
                self.sub_spectra,
                # n
                range(shape[5])
            ]
            if ordering:
                values = values.transpose(ordering)
                levels = [levels[i] for i in ordering]
            values = np.expand_dims(values, map_case_level)
            levels.insert(map_case_level, [map_case])

            dfs.append(ndarray_to_series(values, levels, dataframe_names))
        df = pd.concat(dfs)
        if data_frame_col_level:
            df = df.unstack(level=data_frame_col_level)
        if sort_index:
            df.sort_index(inplace=True)
        return df

    def to_frame_4d(self, array_4d):
        '''convert a 4d array to DataFrame representation where the
        4d array is the first 4 dimensions of GenericSpectra (i.e. `self.names[:4]`)
        '''
        _, _, n_sub_split, n_l = array_4d.shape
        index = pd.MultiIndex.from_product(
            (self.spectra, self.null_split, map(str, range(n_sub_split))),
            names=self.names[:3]
        )
        df = pd.DataFrame(
            array_4d.reshape(-1, n_l),
            index=index,
            columns=self.b_range,
        )
        df.columns.name = self.names[3]
        return df

    def save(self, path, compress_level=9):
        '''save object using HDF5 container

        this implement the HDF5 structure that `load` expects
        '''
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, libver='latest') as f:
            raise NotImplementedError

    @property
    def signal(self):
        '''obtain total signal from no noise simulation

        If leakage is not subtracted, then for those calculated explicitly in `FILTER_SPECTRA_TO_MAP_CASES`,

        For PseudoSpectra: signal - leakage = pseudo spectra of theory

        For Spectra: signal - leakage = BPWF theory

        apart from floating point error.

        Return
        ------

        4d array of the total signal with axes ('spectra', 'null_split',
        'sub_split', 'l')
        '''
        map_cases = [map_case for map_case in self.map_cases if 'sim' in map_case and 'noise' not in map_case]
        assert len(map_cases) == 2
        print(f'These 2 no noise simulations are found to construct the total signal: {map_cases}. One should have TTEE only, another should have TTBB only.', file=sys.stderr)
        Cl_signal = (
            getattr(self, map_cases[0])[:, :, :, :, 0, :].mean(axis=-1) +
            getattr(self, map_cases[1])[:, :, :, :, 0, :].mean(axis=-1)
        )
        # TT signal has been double counted:
        idx = get_idx(self.spectra, 'TT')
        Cl_signal[idx] *= 0.5
        return Cl_signal

    @property
    def leakage(self):
        '''compute leakage from no noise simulation

        `cases`: Automatically determined, look up dict for simulation name
        to the spectra it affected. Included spectra should
        be those expected to be null when there's no leakage.

        Calculated this in PseudoSpectra or Spectra, full or null,
        should be valid, since MASTER is linear.

        Return
        ------

        4d array of the leakage with axes ('spectra', 'null_split',
        'sub_split', 'l')
        '''
        lookup = {
            'BB': ('BB', 'EB', 'TB'),
            'EE': ('EB', 'EE', 'TB', 'TE')
        }
        # it is convoluted to prevent multiple TTEE (or TTBB) no noise maps at the same time
        temp = dict()
        for map_case in self.map_cases:
            if 'simmaps' in map_case and 'noise' not in map_case:
                print(f'No noise simulation {map_case} identified', file=sys.stderr)
                for spectrum in lookup:
                    if spectrum not in map_case:
                        print(f'{map_case} has no {spectrum}...', file=sys.stderr)
                        temp[spectrum] = map_case
        cases = {map_case: lookup[spectrum] for spectrum, map_case in temp.items()}
        print(f'Determined to calculate leakage according to {cases}', file=sys.stderr)

        sample = getattr(self, self.map_cases[0])
        leakage = np.zeros(sample.shape[:4], dtype=sample.dtype)
        for map_case, spectra in cases.items():
            Cl = getattr(self, map_case)
            for idx in get_idxs(self.spectra, spectra):
                leakage[idx] += Cl[idx, :, :, :, 0, :].mean(axis=-1)

        return leakage

    @property
    def signal_to_frame(self):
        signal = self.signal
        return self.to_frame_4d(signal)

    @property
    def leakage_to_frame(self):
        leakage = self.leakage
        return self.to_frame_4d(leakage)

    def subtracting_leakage(self):
        '''Subtracting leakage in all spectra inplace
        '''
        lookup = {
            'BB': ('BB', 'EB', 'TB'),
            'EE': ('EB', 'EE', 'TB', 'TE')
        }
        # loop through no noise simulations
        for map_case in self.map_cases:
            if 'simmaps' in map_case and 'noise' not in map_case:
                print(f'{map_case} identified as no noise simulation,', file=sys.stderr, end=' ')
                # there should only be one match here per map_case
                for spectrum_not_contained, spectra in lookup.items():
                    if spectrum_not_contained not in map_case:
                        print(f'which has no {spectrum_not_contained}, constructing {spectra} leakage from it...', file=sys.stderr)
                        Cl = getattr(self, map_case)
                        # loop through spectra
                        for idx in get_idxs(self.spectra, spectra):
                            leakage = Cl[idx, :, :, :, 0, :].mean(axis=-1)
                            # loop through map_cases that should contain leakage from map_case
                            for map_case_other in self.map_cases:
                                if map_case_other == map_case or 'real' in map_case_other or 'noise' in map_case_other:
                                    Cl_other = getattr(self, map_case_other)
                                    print(f'Subtracting {self.spectra[idx]} leakage from {map_case_other}...', file=sys.stderr)
                                    Cl_other[idx, :, :, :, 0, :] -= leakage[:, :, :, None]

    @property
    def cross_to_auto_idxs(self):
        spectra = self.spectra
        res = np.empty((len(spectra), 2), dtype=np.int64)
        for i, spectrum in enumerate(spectra):
            temp1 = get_idx(spectra, spectrum[0] * 2)
            temp2 = get_idx(spectra, spectrum[1] * 2)
            # sort them
            if temp1 > temp2:
                temp1, temp2 = temp2, temp1
            res[i, 0] = temp1
            res[i, 1] = temp2
        return res

    def inverse_transform_spectra(self, Cl, thetas, rel_dust_spectra=None):
        '''perform inverse transform on spectra
        
        :param Cl: 6d-array in the order defined by GenericSpectra:
            ('spectra', 'null_split', 'sub_split', 'l', 'sub_spectra', 'n')
        :param spectra: array of spectra names
        :param thetas: parameters from likelihood

        c.f. `inverse_transform_subtract_foreground`
        '''
        from .likelihood import inverse_transform_batch, to_spectra_matrix, from_spectra_matrix

        # 'null_split', 'sub_split', 'sub_spectra', 'n', 'l', 'spectra', 'spectra'
        Cl_matrix = to_spectra_matrix(Cl.transpose(0, 1, 2, 4, 5, 3), self.spectra)
        shape = Cl_matrix.shape
        n_b = shape[4]
        m = shape[6]
        Cl_matrix_transformed = inverse_transform_batch(thetas[1:5], Cl_matrix.reshape((-1, n_b, m, m)), self.l_min, self.l_max, self.bin_width).reshape(shape)
        Cl_transformed = from_spectra_matrix(Cl_matrix_transformed.transpose(0, 1, 4, 2, 3, 5, 6), self.spectra)

        if rel_dust_spectra is not None:
            Cl_transformed[get_idx(self.spectra, 'BB'), :, :, :, get_idx(self.sub_spectra, 'Cl'), 0] -= thetas[5] * rel_dust_spectra[None, None, :]
        return Cl_transformed

    def transform_real(
        self,
        thetas,
        rel_dust_spectra=None,
    ):
        '''transform real spectra according to likelihood result

        if `rel_dust_spectra` is not None, MAP dust template will be subtracted too
        '''
        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Transforming {map_case}...', file=sys.stderr)
        Cl = getattr(self, map_case)

        Cl_transformed = self.inverse_transform_spectra(Cl, thetas, rel_dust_spectra=rel_dust_spectra)

        setattr(self, map_case, Cl_transformed)

    def scaling(self, compute_pwf=True, sim_noise_scaling=True, sim_noise_matching=True, angle=(np.pi / 5400.)):
        '''scaling with PWF and/or matching real and sim noise spectra inplace.

        param bool compute_pwf: scaling the real data by pwf to correct for it. Here we assume
        the PWF is slowly varying therefore we can correct this in pseudo-spectra for each l.

        param bool sim_noise_scaling: scaling the noise spectra and the delta
        (relative to theory spectra) of the signal spectra of noise-simulation

        param bool sim_noise_matching: match that of the real noise spectra
        after pwf correction. This is useful if there's noise spectra mismatch
        between real and sim. This should not happen, but can be a quick fix if
        real map is made slightly differently than simulated maps without redoing
        all simulations. (This indeed happened betwee v1 and v1.0.2 run.)

        param float angle: angle used in calculating PWF if pwf is True. Default: 2 arcmin in radian.

        If `compute_pwf`: real spectra are scaled in place.

        If `sim_noise_scaling`: with-noise-simluated spectra are scaled in place.

        This should be done before leakage subtraction to capture the full variance including contribution from
        leakage.

        Unimplemented are the cases that the simulated maps also has a PWF. i.e. input maps and output maps
        has different pixel sizes in simulation.
        '''
        # get real spectra
        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Found 1 real spectra {map_case}', file=sys.stderr)
        Cl_real = getattr(self, map_case)

        if compute_pwf:
            from .pwf import p_l_integrate

            # check bin_width
            # bin_width might not be 1 if it is Spectra, or
            # in older pipelines where the pseudo-spectra
            # is also binned
            bin_width = getattr(self, 'bin_width', None)
            if bin_width is not None and bin_width != 1:
                print(f'bin-width {bin_width} is not 1 and is unspported to correct for PWF.', sys.stderr)
                raise ValueError

            # get l-range
            l_min = self.l_min
            n_l = Cl_real.shape[3]
            l_max = l_min + n_l
            ls = np.arange(l_min, l_max)

            # compute PWF
            # ignore the error term returned from p_l_integrate
            pwf_inv_sq = np.reciprocal(np.square(np.ascontiguousarray(
                p_l_integrate(ls, angle)[:, 0]
            )))[None, None, None, :, None, None]

            # scaling by PWF
            print(f'Scaling {map_case} by PWF...', file=sys.stderr)
            # both sub-spectra (Cl, Nl) scaled by pwf
            Cl_real *= pwf_inv_sq

            del bin_width, l_min, n_l, l_max, ls

        # here the noise part of the simulation is real noise (say from sign-flip noise)
        if sim_noise_scaling:
            assert tuple(self.sub_spectra) == ('Cl', 'Nl')

            Cl_th = self.signal

            # get sim-noise spectra
            map_cases = [map_case for map_case in self.map_cases if 'sim' in map_case and 'noise' in map_case]
            assert len(map_cases) == 1
            map_case = map_cases[0]
            print(f'Found 1 simulated spectra with noise: {map_case}', file=sys.stderr)
            Cl_sim = getattr(self, map_case)

            # determine scale
            if sim_noise_matching:
                print('Compute scale by ratio of analytical errorbar between the noise spectra from real-maps vs. simulations.', file=sys.stderr)

                # get apparent noise spectra (Nl)
                # previously verified all noise-spectra in simulation is appromixately the same as the real noise spectra
                # if both shares identical mapmaking configuration
                Nl = Cl_sim[:, :, :, :, 1, :].mean(-1)
                # this contains PWF from above (if compute_pwf)
                Nl_actual = Cl_real[:, :, :, :, 1, 0]

                cross_to_auto_idxs = self.cross_to_auto_idxs
                rel_err = rel_err_analytic_all(Cl_th, Nl, cross_to_auto_idxs, rel_dof=None)
                rel_err_actual = rel_err_analytic_all(Cl_th, Nl_actual, cross_to_auto_idxs, rel_dof=None)

                with np.errstate(divide='ignore', invalid='ignore'):
                    scale = (rel_err_actual / rel_err)[:, :, :, :, None]
                del Nl, Nl_actual, cross_to_auto_idxs, rel_err, rel_err_actual
                # l == 0 case gives nan
                scale[:, :, :, 0, :] = 1.
                isnan = np.isnan(scale)
                if np.any(isnan):
                    print(f'scale is nan at these location: {np.argwhere(isnan)}. Forcing it to be 1.', file=sys.stderr)
                    scale[isnan] = 1.
                del isnan
                print(f'Min/max scale is {scale.min()}, {scale.max()}', file=sys.stderr)
            elif compute_pwf:
                # scaling by PWF
                print(f'Scaling {map_case} by PWF...', file=sys.stderr)
                scale = pwf_inv_sq
            else:
                print(f'Doing nothing as not computing PWF and not matching noise in simulation.', file=sys.stderr)
                scale = None

            # scaling
            if scale is not None:
                print(f'Scaling noise spectra in {map_case}...', file=sys.stderr)
                Cl_sim[:, :, :, :, 1, :] *= scale
                print(f'Scaling variance of the signal spectra in {map_case}...', file=sys.stderr)
                Cl_th = Cl_th[:, :, :, :, None]
                Cl_sim[:, :, :, :, 0, :] -= Cl_th
                Cl_sim[:, :, :, :, 0, :] *= scale
                Cl_sim[:, :, :, :, 0, :] += Cl_th

    @property
    def err_mc(self):
        map_cases = [map_case for map_case in self.map_cases if 'sim' in map_case and 'noise' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Using {map_case} to estimate MC error-bar', file=sys.stderr)
        Cls = np.array([[[[0.]]]]) if isinstance(self, NullSpectra) else self.signal  # or self.theory[:, None, None, :] in conjunction with rel_err_analytic above
        Cl_sim = getattr(self, map_case)[:, :, :, :, 0]
        return np.sqrt(np.square(Cl_sim - Cls[:, :, :, :, None]).mean(axis=-1))

    @property
    def pte(self):
        map_cases = [map_case for map_case in self.map_cases if 'sim' in map_case and 'noise' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Found noise simulation {map_case}', file=sys.stderr)
        Cl_sims = getattr(self, map_case)[:, :, :, :, 0]

        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Found real spectra {map_case}', file=sys.stderr)
        Cl_real = getattr(self, map_case)[:, :, :, :, 0, 0]

        err = self.err_mc

        chi_real = Cl_real / err
        chi_sims = Cl_sims / err[:, :, :, :, None]

        chi_sq_real = np.square(chi_real).sum(axis=3)
        chi_sq_sims = np.square(chi_sims).sum(axis=3)

        return (chi_sq_real[:, :, :, None] < chi_sq_sims).mean(axis=3)

    def _df_for_plot(self, subtract_leakage=True):
        err_mc = self.err_mc

        err_mcmc = getattr(self, 'err_mcmc', None)
        if err_mcmc is not None:
            print('Found err_mcmc. MCMC error and MC error will be combined assuming idependent Gaussian.', file=sys.stderr)
            err_mc = np.sqrt(np.square(err_mc) + np.square(err_mcmc))

        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Found measured spectra from map-case {map_case}.', file=sys.stderr)
        Cl = getattr(self, map_case)[:, :, :, :, 0, 0]
        signal = self.signal

        if subtract_leakage:
            leakage = self.leakage
            Cl -= leakage
            signal -= leakage

        df = pd.DataFrame({
            'measured': self.to_frame_4d(Cl).stack(),
            'theory': self.to_frame_4d(signal).stack(),
            'err': self.to_frame_4d(err_mc).stack(),
        }).reset_index()
        df['err_l'] = self.bin_width * 0.5

        return df.melt(
            id_vars=('spectra', 'null_split', 'sub_split', 'l', 'err', 'err_l'),
            value_vars=('measured', 'theory'),
            var_name='case',
            value_name='Cl'
        )

    def plot_spectra(self, **kwargs):
        '''plot real spectra and expected (signal) spectra with MC error-bar

        for log scale, try

        >>> fig.update_layout(xaxis_type="log", yaxis_type="log")
        '''
        import plotly.express as px

        return px.scatter(
            self._df_for_plot(**kwargs),
            x='l', y='Cl',
            error_x='err_l', error_y='err',
            color='spectra', symbol='case',
            title='Measured and theory spectra with MC error-bar',
        )

    def _df_for_plot_err_mcmc(self):
        err_mcmc = getattr(self, 'err_mcmc', None)
        if err_mcmc is None:
            raise RuntimeError('err_mcmc not exist. Consider running ComputeLikelihood.err_mcmc_to_spectra')

        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Found measured spectra from map-case {map_case}.', file=sys.stderr)
        Cl = getattr(self, map_case)[:, :, :, :, 0, 0]

        df = pd.DataFrame({
            'measured': self.to_frame_4d(Cl).stack(),
            'err_mc': self.to_frame_4d(self.err_mc).stack(),
            'err_mcmc': self.to_frame_4d(err_mcmc).stack(),
        })
        df.columns.name = 'case'
        df_tidy = df.stack().to_frame('Cl').reset_index()
        df_tidy['err_l'] = self.bin_width * 0.5
        return df_tidy

    def plot_err_mcmc(self):
        import plotly.express as px

        return px.scatter(self._df_for_plot_err_mcmc(), x='l', y='Cl', error_x='err_l', color='spectra', symbol='case', title='MCMC error-bar vs. MC error-bar with measured spectra')


class PseudoSpectra(GenericSpectra):
    '''pseudo-spectra

    This is supposed to be in "D_l"-space and in microK

    In all the `map_cases` arrays, they should have 6 dimensions
    corresponding to `names` below

    Warning: `l_min != 0` case is not implemented here
    and probably failed silently?
    '''
    rel_prefix = 'pseudo_spectra'

    @classmethod
    def load_pandas(cls, df, remove_null_in_map_case=True, **kwargs):
        '''load from pandas DataFrame

        Form a pair with `to_frame`.

        `kwargs` ignored.
        '''
        from dautil.pandas_util import df_to_ndarray

        map_cases = tuple(df.index.get_level_values('map_case').unique())

        sub_spectra = None
        spectra = None
        null_split = None
        kwargs = dict()
        for map_case in map_cases:
            # the transpose is to put the column dimension in the last array dimension
            # as pandas' column is contiguous
            # names: ('sub_spectra', 'spectra', 'sub_split', 'null_split', 'n', 'l')
            values, levels, names = df_to_ndarray(df.xs(map_case, level='map_case').T, unique=True)

            assert tuple(names) == ('sub_spectra', 'spectra', 'sub_split', 'null_split', 'n', 'l')
            assert set(names) == set(cls.names)

            # check all map_case shares the same sub_spectra, spectra, null_split
            if sub_spectra is None:
                sub_spectra = tuple(levels[0])
            else:
                assert sub_spectra == tuple(levels[0])
            if spectra is None:
                spectra = tuple(levels[1])
            else:
                assert spectra == tuple(levels[1])
            if null_split is None:
                null_split = tuple(levels[3])
            else:
                assert null_split == tuple(levels[3])

            # check that sub_split, n, l are just trivial arange
            for i in (2, 4, 5):
                np.testing.assert_array_equal(levels[i].values.astype(np.int), np.arange(values.shape[i]))

            # in the past null_ is hard-coded in the beginning
            map_case_new = map_case[5:] if remove_null_in_map_case and map_case.startswith('null_') else map_case
            # new order: ('spectra', 'null_split', 'sub_split', 'l', 'sub_spectra', 'n')
            kwargs[map_case_new] = np.ascontiguousarray(values.transpose((1, 3, 2, 5, 0, 4)))

        return cls(spectra, null_split, sub_spectra, **kwargs)

    @staticmethod
    def load_h5_pandas_to_frame(paths):
        '''load from HDF5 files with pandas
        '''
        df = pd.concat((pd.read_hdf(path) for path in paths))

        # pandas doesn't save columns name to HDF5
        df.columns.name = 'l'

        # convert to microK
        df *= 1e12
        df.sort_index(inplace=True)
        return df

    @classmethod
    def load_h5_pandas(cls, paths, **kwargs):
        '''load from HDF5 files with pandas

        Form a pair with `to_frame`.
        '''
        df = cls.load_h5_pandas_to_frame(paths)
        return cls.load_pandas(df)

    @classmethod
    def load_h5(cls, path, map_cases=None, full=True, **kwargs):
        '''load from HDF5 file

        This assume the output written by `save` below.
        '''
        with h5py.File(path, 'r') as file:
            f = file[cls.meta_prefix]

            sub_spectra = f['sub_spectra'][:].astype(np.unicode_)
            spectra = f['spectra'][:].astype(np.unicode_)
            null_split = ['full'] if full else f['null_split'][:].astype(np.unicode_)

            f = file[cls.get_prefix(full)]

            map_cases_temp = f.keys()
            if map_cases is not None:
                map_cases = set(map_cases) & set(map_cases_temp)
            else:
                map_cases = map_cases_temp

            kwargs = {case: f[case][:] for case in map_cases}
        return cls(spectra, null_split, sub_spectra, **kwargs)

    @classmethod
    def load(cls, paths, spec='h5', **kwargs):
        '''load from file(s) according to `spec`
        '''
        try:
            method = {
                'pandas': cls.load_pandas,
                'h5_pandas': cls.load_h5_pandas,
                'h5': cls.load_h5,
            }
        except AttributeError:
            print(f'spec {spec} is not implemented in {cls}.', file=sys.stderr)
            raise NotImplementedError
        return method[spec](paths, **kwargs)

    def save(self, path, compress_level=9):
        '''save object using HDF5 container

        this implement the HDF5 structure that `load` expects
        '''
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, libver='latest') as f:
            for case in self.cases:
                h5_check_before_create(f, f'{self.meta_prefix}/{case}', getattr(self, case), compress_level=compress_level)
            for case in self.map_cases:
                h5_check_before_create(f, f'{self.prefix}/{case}', getattr(self, case), compress_level=compress_level)

    def transform_for_filter_transfer(self, l_min, l_max, cases):
        '''transform the array in a format needed by `.filter_transfer.ComputeFilterTransfer`
        '''
        sub_spectra_idx = get_idx(self.sub_spectra, 'Cl')
        l_idxs = self.l_slice(l_min, l_max)

        res = []
        for case, map_cases in cases.items():
            spectra_idx = get_idx(self.spectra, case)
            # axis -1 is the i-th simulation axis
            ps_array = np.concatenate(
                [
                    getattr(self, map_case)[spectra_idx, :, :, l_idxs, sub_spectra_idx, :]
                    for map_case in map_cases
                ],
                axis=-1
            )
            # ('null_split', 'sub_split', 'l')
            ps_mean_sem = ps_array.mean(axis=-1) + stats.sem(ps_array, axis=-1) * 1.j
            res.append(ps_mean_sem)
        # ('spectra', 'null_split', 'sub_split', 'l')
        pseudos = np.ascontiguousarray(np.stack(res))
        return pseudos

    def pseudo_spectra_dict_l_sliced(self, l_min, l_max):
        '''get all pseudo_spectra in a dict container with l sliced
        primarily for .spectra.ComputeSpectra.load
        '''
        l_idxs = self.l_slice(l_min, l_max)
        return {
            map_case:
            getattr(self, map_case)[:, :, :, l_idxs]
            for map_case in self.map_cases
        }


class GenericDataFrameContainer(Generic):
    prefix = ''

    @staticmethod
    def load_h5_pandas_to_frame(path, **kwargs):
        '''load from HDF5 files with pandas
        '''
        df = pd.read_hdf(path)
        return df

    @classmethod
    def load_to_frame(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_frame(cls, path, **kwargs):
        df = cls.load_to_frame(path, **kwargs)
        df.reset_index(inplace=True)
        return cls(**{name: group.values for name, group in df.items()})

    def to_frame(self, set_index='l'):
        df = pd.DataFrame({case: getattr(self, case) for case in self.cases})
        if set_index:
            df.set_index(set_index, inplace=True)
        return df


class TheorySpectra(GenericDataFrameContainer):
    '''Theory input spectra
    '''
    prefix = 'theory_spectra'

    @staticmethod
    def load_class_txt_to_frame(path, camb=True, **kwargs):
        '''read CLASS' .dat output and convert to DataFrame

        `camb`: if True, assumed ``format = camb`` is used when
        generating the .dat from CLASS
        '''
        # read once
        with open(path, 'r') as f:
            text = f.read()

        # get last comment line
        comment = None
        for line in text.split('\n'):
            if line.startswith('#'):
                comment = line
        # remove beginning '#'
        comment = comment[1:]
        # parse comment line: get name after ':'
        names = [name.split(':')[1] for name in comment.strip().split()]

        with io.StringIO(text) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col=0, comment='#', header=None, names=names)
            return df if camb else df * 1.e12

    @classmethod
    def load_to_frame(cls, path, **kwargs):
        '''load from file according to extension
        '''
        method = {
            '.dat': cls.load_class_txt_to_frame,
            '.hdf5': cls.load_h5_pandas_to_frame,
        }
        df = method[Path(path).suffix.lower()](path, **kwargs)
        return df

    @classmethod
    def load(cls, path, spec='h5', **kwargs):
        '''load from file(s) according to `spec`
        '''
        method = {
            None: cls.load_from_frame,
            'h5': cls.load_h5,
        }
        return method[spec](path, **kwargs)

    def l_slice(self, l_min, l_max):
        '''return a slice on l given min, max
        '''
        idxs = get_idxs(self.l, (l_min, l_max))
        return slice(*idxs)

    def packing(self, l_min, l_max, spectra):
        '''pack theory spectra in a 2d contiguous array
        '''
        l_idxs = self.l_slice(l_min, l_max)
        # for spectrum not existed, assume theory expectation is 0
        ths = np.zeros((len(spectra), l_max - l_min), dtype=getattr(self, next(iter(spectra))).dtype)
        for idx, spectrum in enumerate(spectra):
            Cl = getattr(self, spectrum, None)
            if Cl is not None:
                ths[idx] = Cl[l_idxs]
        return ths


class BeamSpectra(GenericDataFrameContainer):
    prefix = 'beam_spectra'

    @classmethod
    def load_to_frame(cls, path, **kwargs):
        '''load from file according to extension
        '''
        method = {
            '.hdf5': cls.load_h5_pandas_to_frame,
        }
        df = method[Path(path).suffix.lower()](path, **kwargs)
        # historically the beam file has no index name
        df.index.name = 'l'
        return df

    @classmethod
    def load(cls, path, spec='h5', **kwargs):
        '''load from file(s) according to `spec`
        '''
        method = {
            'h5_pandas': cls.load_from_frame,
            'h5': cls.load_h5,
        }
        return method[spec](path, **kwargs)

    def squared_interp(self, l_min, l_max, uncertainties=False):
        '''return the squared beam, interp over the interval [l_min, l_max)

        used in ComputeFilterTransfer for example
        '''
        from .helper import complex_to_uncertainties

        B = (
            complex_to_uncertainties(
                np.interp(range(l_min, l_max), self.l, self.all)
            )
        ) if uncertainties else (
            np.interp(range(l_min, l_max), self.l, self.all.real)
        )
        return np.square(B)


def get_theory(theory_spectra: TheorySpectra, w_bl, l_min, l_max, spectra):
    if w_bl.ndim != 3:
        print('Did not implement the case where w_bl can depends on null-splits yet.', file=sys.stderr)
        raise NotImplementedError
    ths = theory_spectra.packing(l_min, l_max, spectra)
    return np.einsum('ijk,ik->ij', w_bl, ths)


class Spectra(GenericSpectra):
    '''similar to PseudoSpectra but for master corrected spectra
    '''
    rel_prefix = 'spectra'

    def __init__(
        self,
        spectra, null_split, sub_spectra,
        l_min, l_max, bin_width,
        w_bl=None,
        rel_err=None,
        right=False,
        theory=None,
        **kwargs
    ):
        # in addtion to the vars above, this adds self.is_full, self.cases, self.prefix
        super().__init__(spectra, null_split, sub_spectra, **kwargs)
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = bin_width
        self.w_bl = w_bl
        self.rel_err = rel_err
        self.right = right
        self.theory = theory

    @property
    def theory_to_frame(self):
        if self.theory is None:
            return None
        else:
            df = pd.DataFrame(self.theory, index=self.spectra, columns=self.b_range)
            df.index.name = self.names[0]
            df.columns.name = self.names[3]
            return df

    @staticmethod
    @jit(float64[:, ::1](float64[:, :, ::1], int32, int32), nopython=True, nogil=True)
    def get_rel_dof(w_bl, l_min, bin_width):
        '''calculate the total dof in a bin given w_bl relative to full sky coverage

        dof given f_sky will be this times f_sky

        return a 2d-array with this order: ('spectra', 'b')
        '''
        n_spectra, n_b, n_l = w_bl.shape
        res = np.empty((n_spectra, n_b), dtype=w_bl.dtype)
        two_times_l_min_plus_1 = 2 * l_min + 1
        for i in range(n_spectra):
            for b in range(n_b):
                nu_pos = 0.
                nu_neg = 0.
                for l in range(n_l):
                    temp = (2 * l + two_times_l_min_plus_1) * w_bl[i, b, l]
                    if temp >= 0.:
                        nu_pos += temp
                    else:
                        nu_neg -= temp
                diff = nu_pos - nu_neg
                res[i, b] = (diff * diff * bin_width) / (nu_pos + nu_neg)
        return res

    @property
    def rel_dof(self):
        return None if self.w_bl is None else self.get_rel_dof(self.w_bl, self.l_min, self.bin_width)

    @property
    def rel_err_analytic(self):
        map_cases = [map_case for map_case in self.map_cases if 'real' in map_case]
        assert len(map_cases) == 1
        map_case = map_cases[0]
        print(f'Using noise spectra from {map_case} to estimate analytical error-bar', file=sys.stderr)
        # or self.theory[:, None, None, :] in conjunction with err_mc below (beware of Cls, Nls shape mis-match if not self.is_full)
        Cls = self.signal
        Nls = getattr(self, map_case)[:, :, :, :, 1, 0]
        return rel_err_analytic_all(Cls, Nls, self.cross_to_auto_idxs, rel_dof=self.rel_dof)

    def err_analytic(self, f_sky: FSky):
        f_sky_dict = {
            'BB': f_sky.get_f_sky('p', 'noise'),
            'EB': f_sky.get_f_sky('p', 'noise'),
            'EE': f_sky.get_f_sky('p', 'noise'),
            'TB': f_sky.get_f_sky('t', 'noise'),
            'TE': f_sky.get_f_sky('t', 'noise'),
            'TT': f_sky.get_f_sky('t', 'signal'),
        }
        f_sky_array = np.array([f_sky_dict[spectrum] for spectrum in self.spectra])
        return self.rel_err_analytic / np.sqrt(f_sky_array[:, None, None, None])

    def save(self, path, compress_level=9):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, libver='latest') as f:
            # ('spectra', 'sub_spectra', 'null_split')
            for case in self.cases:
                h5_check_before_create(f, f'{self.meta_prefix}/{case}', getattr(self, case), compress_level=compress_level)

            # l_min, l_max, bin_width
            subdir = '_'.join(map(str, (self.l_min, self.l_max, self.bin_width)))
            for case in self.map_cases:
                h5_check_before_create(f, f'{self.prefix}/{subdir}/map_cases/{case}', getattr(self, case), compress_level=compress_level)

            if self.rel_err is not None:
                h5_check_before_create(f, f'{self.prefix}/{subdir}/rel_err', self.rel_err, compress_level=compress_level)
            if self.w_bl is not None:
                # w_bl: same for both full and null if not right
                # if not right then override this and put in the full dir instead
                prefix = f'full/{self.rel_prefix}' if (not self.is_full and not self.right) else self.prefix
                h5_check_before_create(f, f'{prefix}/{subdir}/w_bl', self.w_bl, compress_level=compress_level)

            # right is already saved in filter_transfer in `f['meta/filter_transfer'][1].astype(np.unicode_) == 'right'

    @classmethod
    def load(
        cls,
        path,
        bin_width,
        l_min=None,
        l_max=None,
        map_cases=None, full=True,
        path_theory_spectra: Path = None,
        **kwargs
    ):
        '''load from HDF5 file

        This assume the output written by `save` above.
        '''
        with h5py.File(path, 'r') as file:
            f = file[cls.meta_prefix]

            sub_spectra = f['sub_spectra'][:].astype(np.unicode_)
            spectra = f['spectra'][:].astype(np.unicode_)
            null_split = ['full'] if full else f['null_split'][:].astype(np.unicode_)

            prefix = cls.get_prefix(full)
            f = file[f'{prefix}']

            # l_min, l_max, bin_width
            if l_min is None or l_max is None:
                print(f'l_min/l_max not specified,', file=sys.stderr, end=' ')
                subdirs = [subdir for subdir in f.keys() if subdir.endswith(f'_{bin_width}')]
                n_subdirs = len(subdirs)
                if n_subdirs == 1:
                    subdir = subdirs[0]
                    l_min, l_max, _ = list(map(int, subdir.split('_')))
                    print(f'auto-choose l_min at {l_min} and l_max at {l_max}.', file=sys.stderr)
                elif n_subdirs == 0:
                    raise ValueError(f'bin-width {bin_width} not in {f}, consider generating that and save to file.')
                elif n_subdirs > 1:
                    raise ValueError(f'Multiple groups with bin-width {bin_width} found in {f}: {subdirs}, consider specifying l_min and l_max.')

            else:
                subdir = '_'.join(map(str, (l_min, l_max, bin_width)))
                if subdir not in f:
                    raise ValueError(f'l_min, l_max and bin_width at {l_min}, {l_max}, {bin_width} not in {f}, consider generating that and save to file.')

            f = file[f'{prefix}/{subdir}/map_cases']

            map_cases_temp = f.keys()
            if map_cases is not None:
                map_cases = set(map_cases) & set(map_cases_temp)
            else:
                map_cases = map_cases_temp

            kwargs = {case: f[case][:] for case in map_cases}

            f = file[f'{prefix}/{subdir}']
            rel_err = f['rel_err'][:] if 'rel_err' in f else None

            if 'meta/filter_transfer' in file:
                right = file['meta/filter_transfer'][1].astype(np.unicode_) == 'right'
            else:
                print(f'Cannot find meta/filter_transfer from {file}, you may have saved the filter transfer functions and the spectra in different files. Assume right=False for now...', file=sys.stderr)
                right = False

            # w_bl: same for both full and null if not right
            # if not right then override this and put in the full dir instead
            if (not full and not right):
                f = file[f'full/{cls.rel_prefix}/{subdir}']
            w_bl = f['w_bl'][:] if 'w_bl' in f else None

        # theory
        if path_theory_spectra is None:
            theory = None
        elif w_bl is None:
            print('Did not find w_bl, ignore getting theory spectra.', file=sys.stderr)
            theory = None
        else:
            theory_spectra = TheorySpectra.load(path_theory_spectra, keys=('l', 'BB', 'EE', 'TE', 'TT'))
            theory = get_theory(theory_spectra, w_bl, l_min, l_max, spectra)

        return cls(
            spectra, null_split, sub_spectra,
            l_min, l_max, bin_width,
            w_bl=w_bl,
            rel_err=rel_err,
            right=right,
            theory=theory,
            **kwargs
        )

    def slicing_l(self, l_min, l_max, ascontiguousarray=False):
        '''return a Spectra that is sliced with the given l-range

        :param bool ascontiguousarray: if True, return contiguous arrays
        when sliced. Note that this means it is not a view on the original
        array anymore.

        Note that it does not check if the l_min, l_max appears right on the
        boundary of the l-bins and will propagate as is, e.g. in `w_bl`.
        '''
        b_slice = self.b_slice(l_min, l_max)

        w_bl = self.w_bl
        if w_bl is not None:
            l_slice = self.l_slice(l_min, l_max)
            w_bl = w_bl[:, b_slice][:, :, l_slice]
            if ascontiguousarray:
                w_bl = np.ascontiguousarray(w_bl)

        rel_err = self.rel_err
        if rel_err is not None:
            rel_err = rel_err[:, :, :, b_slice]
            if ascontiguousarray:
                rel_err = np.ascontiguousarray(rel_err)

        theory = self.theory
        if theory is not None:
            theory = theory[:, b_slice]
            if ascontiguousarray:
                theory = np.ascontiguousarray(theory)

        kwargs = {map_case: getattr(self, map_case)[:, :, :, b_slice] for map_case in self.map_cases}
        if ascontiguousarray:
            kwargs = {key: np.ascontiguousarray(value) for key, value in kwargs.items()}

        return Spectra(
            self.spectra, self.null_split, self.sub_spectra,
            l_min, l_max, self.bin_width,
            w_bl=w_bl,
            rel_err=rel_err,
            right=self.right,
            theory=theory,
            **kwargs
        )


class NullSpectra(Spectra):

    def __init__(
        self,
        spectra, null_split, sub_spectra,
        l_min, l_max, bin_width,
        w_bl=None,
        rel_err=None,
        right=False,
        theory=None,
        err_analytic=None,
        **kwargs
    ):
        super().__init__(
            spectra, null_split, sub_spectra,
            l_min, l_max, bin_width,
            w_bl=w_bl,
            rel_err=rel_err,
            right=right,
            theory=theory,
            **kwargs
        )
        assert not self.is_full
        n_sub_split = getattr(self, self.map_cases[0]).shape[2]
        assert n_sub_split == 1

        # this should override inherrited property
        self.err_analytic = err_analytic

    @classmethod
    def load(cls, spectra: Spectra, path_mask: Path = None, f_sky: FSky = None):
        '''
        if either `path_mask` or `f_sky` is given, calculate FSky and analytical error-bar
        '''
        assert tuple(spectra.sub_spectra) == ('Cl', 'Nl')

        kwargs = dict()
        for map_case in spectra.map_cases:
            Cl = getattr(spectra, map_case)
            n_spectra, n_null_split, n_sub_split, n_l, n_sub_spectra, n = Cl.shape
            assert n_sub_split == 3
            assert n_sub_spectra == 2
            Cl_null = np.empty((n_spectra, n_null_split, 1, n_l, 1, n), dtype=Cl.dtype)
            # signal spectra
            Cl_null[:, :, 0, :, 0, :] = (
                Cl[:, :, 0, :, 0, :] +
                Cl[:, :, 1, :, 0, :] -
                Cl[:, :, 2, :, 0, :] * 2.
            )
            kwargs[map_case] = Cl_null

        if path_mask is not None or f_sky is not None:
            if f_sky is None:
                f_sky = FSky.load(path_mask)

            err_analytic = spectra.err_analytic(f_sky)
            # n_sub_split
            assert err_analytic.shape[2] == 3
            err_combined = np.sqrt(
                np.square(err_analytic[:, :, 0]) +
                np.square(err_analytic[:, :, 1]) +
                np.square(err_analytic[:, :, 2]) * 4.
            )[:, :, None, :]  # add axis to sub_split level
        else:
            err_combined = None

        return cls(
            spectra.spectra, spectra.null_split, spectra.sub_spectra[:1],
            spectra.l_min, spectra.l_max, spectra.bin_width,
            w_bl=spectra.w_bl,
            rel_err=spectra.rel_err,
            right=spectra.right,
            theory=None,  # NullSpectra always has theory expectation of 0
            err_analytic=err_combined,
            **kwargs
        )


class FilterTransfer(Generic):

    meta_prefix = 'meta'
    rel_prefix = 'filter_transfer'
    names = ('spectra', 'null_split', 'sub_split', 'l')

    def __init__(
        self,
        Fs,
        full,
        null_split,
        norm,
        right,
        cases,
        l_min, l_max, bin_width
    ):
        self.Fs = Fs
        self.full = full
        self.null_split = null_split
        self.norm = norm
        self.right = right
        self.cases = cases
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = bin_width

        self.prefix = '/'.join((
            'full' if full else 'null',
            self.rel_prefix,
        ))

    def to_frame(self):
        index = pd.MultiIndex.from_product(
            (
                self.cases,
                self.null_split,
                # sub_split historically is str
                map(str, range(self.Fs.shape[2])),
            ),
            names=self.names[:-1]
        )
        df = pd.DataFrame(
            self.Fs.reshape(-1, self.Fs.shape[-1]),
            index=index,
            columns=range(self.l_min, self.l_max)
        )
        df.columns.name = self.names[-1]
        return df

    def save(self, path, compress_level=9):
        with h5py.File(path, libver='latest') as f:
            # Fs
            h5_check_before_create(f, f'{self.prefix}/F', self.Fs, compress_level=compress_level)

            # l_min, l_max, bin_width
            if self.bin_width is None:
                array = np.array([self.l_min, self.l_max], dtype=np.uint16)
            else:
                array = np.array([self.l_min, self.l_max, self.bin_width], dtype=np.uint16)
            h5_check_before_create(f, f'{self.prefix}/l', array, compress_level=compress_level)

            # null_split
            if not self.full:
                h5_check_before_create(f, f'{self.meta_prefix}/null_split', self.null_split, compress_level=compress_level)

            # norm, right, cases
            norm = 'None' if self.norm is None else self.norm
            right = 'right' if self.right else 'left'
            array = np.array([norm, right] + list(self.cases), dtype=np.string_)
            h5_check_before_create(f, f'{self.meta_prefix}/filter_transfer', array, compress_level=compress_level)

    @classmethod
    def load(cls, path, full=True):

        prefix = '/'.join((
            'full' if full else 'null',
            cls.rel_prefix,
        ))

        with h5py.File(path, 'r') as f:
            # Fs
            Fs = f[f'{prefix}/F'][:]

            # l_min, l_max, bin_width
            temp = f[f'{prefix}/l'][:]
            if temp.size == 3:
                l_min, l_max, bin_width = temp
            else:
                l_min, l_max = temp
                bin_width = None

            # null_split
            null_split = ['full'] if full else f[f'{cls.meta_prefix}/null_split'][:].astype(np.unicode_)

            # norm, right, cases
            temp = f[f'{cls.meta_prefix}/filter_transfer'][:].astype(np.unicode_)
            temp2 = temp[0]
            norm = None if temp2 == 'None' else temp2
            right = temp[1] == 'right'
            cases = temp[2:]

        return cls(
            Fs,
            full,
            null_split,
            norm,
            right,
            cases,
            l_min, l_max, bin_width
        )

    def solve(self, spectra, uncertainties=False):
        '''calculate filter transfer functions for cross spectra too

        param spectra: list of spectrum such as EE, TE, TB, etc. `self.cases`
        is auto-spectra only. And cross-spectra are calculated using sqrt of
        product of auto-spectra
        '''
        if uncertainties:
            from .helper import complex_to_uncertainties
            from uncertainties import unumpy as up

            Fs = complex_to_uncertainties(self.Fs)
            sqrt = up.sqrt
        else:
            Fs = self.Fs.real
            sqrt = np.sqrt

        shape = list(Fs.shape)
        shape[0] = len(spectra)
        res = np.empty(shape, dtype=Fs.dtype)
        # for easy look up
        F_dict = {spectrum: F for spectrum, F in zip(self.cases, Fs)}
        for i, spectrum in enumerate(spectra):
            spectrum_0 = spectrum[0]
            spectrum_1 = spectrum[1]
            if spectrum in F_dict:
                res[i] = F_dict[spectrum]
            # if not calculated explicitly, estimate from that of auto-spectra
            # e.g. EB, TB
            else:
                res[i] = sqrt(F_dict[spectrum_0 * 2] * F_dict[spectrum_1 * 2])
        return res

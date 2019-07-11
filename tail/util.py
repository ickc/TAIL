# normalize ############################################################

import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def normalize_max(mask):
    '''in-place normalize maximum value to be 1.
    '''
    mask /= np.max(mask)


@jit(nopython=True, nogil=True)
def normalize_row(mask):
    '''in-place normalize each row sum to 1.
    '''
    n = mask.shape[0]
    for i in range(n):
        mask[i] /= mask[i].sum()


@jit(nopython=True, nogil=True)
def norm_fft(mask):
    '''Calculcates the norm for Fourier space values.'''
    shape = mask.shape
    return np.reciprocal(np.sqrt(np.mean(np.square(mask))) * (shape[0] * shape[1]))

# others ###############################################################

@jit(nopython=True, nogil=True)
def fill_nan(array, fill_boundary=True):
    '''``array``: 1d-array
    ``fill_boundary``: if True, fill nan at the boundaries by the nearest
    value.
    in-place filling nan with interpolated values
    '''
    n = array.size

    def get_next_idx(current_idx):
        non_nan_not_found = True
        next_idx = current_idx + 1
        while non_nan_not_found and next_idx < n:
            if np.isnan(array[next_idx]):
                next_idx += 1
            else:
                non_nan_not_found = False
        return next_idx

    def fill(current_idx, next_idx):
        d_idx = next_idx - current_idx
        # case of no nan between 2 idxs
        if d_idx == 1:
            return
        # case of left boundary
        elif current_idx == -1:
            # case of all nan
            if next_idx == n:
                print('Whole array is nan. Nothing is filled.')
                return
            # case of left boundary only, fill_boundary
            elif fill_boundary:
                fill_value = array[next_idx]
                for i in range(next_idx):
                    array[i] = fill_value
                return
        # case of right boundary only, fill_boundary
        elif next_idx == n and fill_boundary:
            fill_value = array[current_idx]
            for i in range(current_idx + 1, n):
                array[i] = fill_value
            return
        # case of not boundary, not all nan
        else:
            start = array[current_idx]
            step = (array[next_idx] - start) / d_idx
            fill_value = start + step
            for i in range(current_idx + 1, next_idx):
                array[i] = fill_value
                fill_value += step
            return

    current_idx = -1
    done = False
    while not done:
        next_idx = get_next_idx(current_idx)
        fill(current_idx, next_idx)
        current_idx = next_idx
        if next_idx == n:
            done = True


@jit(nopython=True, nogil=True)
def effective_sky_noise_dominated(mask, pixel_size):
    '''calculate effective sky coverage assuming noise dominated
    '''
    return 0.25 * pixel_size * pixel_size * mask.sum()**2. / (np.square(mask).sum() * np.pi)


@jit(nopython=True, nogil=True)
def effective_sky(mask, pixel_size):
    '''calculate effective sky coverage assuming signal dominated
    '''
    mask_2 = np.square(mask)
    return effective_sky_noise_dominated(mask_2, pixel_size)

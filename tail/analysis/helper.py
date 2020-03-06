def complex_to_uncertainties(array):
    from uncertainties import unumpy as up

    return up.uarray(array.real, array.imag)


def uncertainties_to_real(uarray):
    from uncertainties import unumpy as up

    return up.nominal_values(uarray)


def uncertainties_to_complex(uarray):
    from uncertainties import unumpy as up

    return up.nominal_values(uarray) + up.std_devs(uarray) * 1.j


def uncertainties_to_rel(uarray):
    from uncertainties import unumpy as up

    return up.std_devs(uarray) / up.nominal_values(uarray)


def matrix_reduce(M, b):
    '''reduce the resolution of the matrix M by bin-width ``b``
    and devided by ``b`` from its last 2 dimensions
    '''
    if b == 1:
        return M
    else:
        shape = M.shape
        m, n = shape[-2:]
        shape_new = list(shape[:-2]) + [m // b, b, n // b, b]
        return M.reshape(shape_new).sum(axis=(-3, -1)) / b
        # return np.einsum('...ijkl->...ik', M.reshape(shape_new)) / b


def matrix_reduce_row(M, b):
    '''reduce the resolution of the matrix M by bin-width ``b``
    and devided by ``b`` from its 2nd last dimension
    '''
    if b == 1:
        return M
    else:
        shape = M.shape
        m, n = shape[-2:]
        shape_new = list(shape[:-2]) + [m // b, b, n]
        return M.reshape(shape_new).mean(axis=-2)
        # return np.einsum('...ijk->...ik', M.reshape(shape_new)) / b


def matrix_reduce_col(M, b):
    '''reduce the resolution of the matrix M by bin-width ``b``
    and devided by ``b`` from its last dimension
    '''
    if b == 1:
        return M
    else:
        shape = M.shape
        n = shape[-1]
        shape_new = list(shape[:-1]) + [n // b, b]
        return M.reshape(shape_new).mean(axis=-1)
        # return np.einsum('...ij->...i', M.reshape(shape_new)) / b

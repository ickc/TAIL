from ..analysis.loader import AllInput
from ..analysis.computer import ComputeFilterTransfer, ComputeSpectra

from pathlib import Path
from typing import List
import sys

import numpy as np
import h5py


def solve_spectra(
    path: Path,
    *,
    out_path: Path,
    full: bool = False,
    # pseudo-spectra manipulation
    transform_real: bool = False,
    thetas_map: List[float] = [],
    scaling: bool = False,
    sim_noise_matching: bool = False,
    subtracting_leakage: bool = False,
    # filter options
    right: bool = False,
    filter_bin_width: int = None,
    filter_divided_from_spectra: bool = False,
    # spectra options
    bin_widths: List[int] = [50],
    compute_pwf: bool = False,
    # misc
    mode_coupling_identity_BB: bool = False,
    return_w: bool = None,
):
    '''solve spectra using modified MASTER

    :param Path path: input path to pseudo-spectra, etc.
    :param Path out_path: output Path
    :param bool full: if True calculate full spectra else null spectra
    :param bool transform_real: transform the real spectra by thetas by reading from out_path
    :param list[float] thetas_map: Use together with transform_real. if specified, use this instead of reading from out_path
    :param bool scaling: run scaling in pseudo_spectra
    :param bool sim_noise_matching: set sim_noise_matching used in scaling
    :param bool subtracting_leakage: subtracting leakage
    :param bool right: filter transfer function option
    :param int filter_bin_width: filter transfer function option
    :param bool filter_divided_from_spectra: if True, divide the filter transfer
        function from pseudo-spectra. Else, multiply it into K_ll. Invalid if
        right is True.
    :param list[int] bin_widths: list of bin-width of the final spectra
    :param bool compute_pwf: compute PWF in solving spectra. This is mutually exclusive with scaling in pseudo-spectra.
    :param bool mode_coupling_identity_BB: set BB mode-coupling matrix to identity matrix
    :param bool return_w: return w_bl or not. If None, infer automatically.
    '''
    if scaling and compute_pwf:
        raise ValueError(f'scaling and compute_pwf should not be specified at the same time.')
    if transform_real:
        if thetas_map:
            thetas_map = np.array(thetas_map)
        else:
            with h5py.File(out_path, 'r') as f:
                thetas_map = f['/meta/thetas_map'][:]

    all_input = AllInput.load(path, full=full)
    if mode_coupling_identity_BB:
        all_input.mode_coupling.BBBB = np.identity(all_input.mode_coupling.BBBB.shape[0], dtype=all_input.mode_coupling.BBBB.dtype)
    if transform_real:
        print(f'transform real spectra using thetas {thetas_map}.', file=sys.stderr)
        all_input.pseudo_spectra.transform_real(thetas_map)
    if scaling:
        all_input.pseudo_spectra.scaling(sim_noise_matching=sim_noise_matching)
    if subtracting_leakage:
        all_input.pseudo_spectra.subtracting_leakage()
    filter_transfer_computer = ComputeFilterTransfer.load(all_input)
    all_input.filter_transfer = filter_transfer_computer.solve(right=right, bin_width=filter_bin_width)

    all_input.filter_transfer.save(out_path)

    spectra_computer = ComputeSpectra.load(all_input, filter_divided_from_spectra=filter_divided_from_spectra, compute_pwf=compute_pwf)
    for bin_width in bin_widths:
        spectra = spectra_computer.solve(bin_width, return_w=return_w)
        spectra.save(out_path)


def cli():
    import defopt

    defopt.run(solve_spectra)

from ..analysis.container import FSky
from ..analysis.loader import AllInput
from ..analysis.computer import ComputeFilterTransfer, ComputeSpectra
from ..analysis.likelihood import ComputeLikelihood

from pathlib import Path
from copy import deepcopy
import sys

import h5py


def iterative_map(
    path: Path,
    path_mask: Path,
    *,
    out_path: Path,
    # pseudo-spectra manipulation
    scaling: bool = True,
    sim_noise_matching: bool = True,
    # filter options
    right: bool = False,
    filter_bin_width: int = None,
    # spectra options
    bin_width: int = 50,
    # narrow l-range for science
    l_min: int = 600,
    l_max: int = 3000,
    # no. of iterations
    n: int = 100,
):
    '''calculate MAP iteratively

    :param Path path: input path to pseudo-spectra, etc.
    :param Path path_mask: input path of the mask for FSky
    :param Path out_path: output Path
    :param bool scaling: run scaling in pseudo_spectra
    :param bool sim_noise_matching: set sim_noise_matching used in scaling
    :param bool right: filter transfer function option
    :param int filter_bin_width: filter transfer function option
    :param int bin_width: bin_width of the final spectra
    :param int l_min: starting l-range for science
    :param int l_max: ending l-range for science
    :param int n: no. of iterations
    '''
    f_sky = FSky.load(path_mask)

    all_input = AllInput.load(path, full=True)
    filter_transfer_computer = ComputeFilterTransfer.load(all_input)
    all_input.filter_transfer = filter_transfer_computer.solve(right=right, bin_width=filter_bin_width)

    pseudo_spectra = all_input.pseudo_spectra
    thetas = None
    for i in range(n):
        # make a copy before modify in-place
        all_input.pseudo_spectra = deepcopy(pseudo_spectra)
        print(f'Made a deepcopy of pseudo_spectra: {all_input.pseudo_spectra}', file=sys.stderr)

        if thetas is not None:
            all_input.pseudo_spectra.transform_real(thetas)

        if scaling:
            all_input.pseudo_spectra.scaling(sim_noise_matching=sim_noise_matching)

        spectra_computer = ComputeSpectra.load(all_input, uncertainties=False)

        spectra = spectra_computer.solve(bin_width)

        # if not last loop...
        if i == n - 1:
            break

        spectra = spectra.slicing_l(l_min, l_max, ascontiguousarray=True)
        likelihood_computer = ComputeLikelihood.load(spectra, f_sky)
        thetas_diff = likelihood_computer.map

        if thetas is None:
            thetas = thetas_diff
            print(f"{i},thetas,{','.join(map(str, thetas))}")
        else:
            # update thetas
            thetas[0] = thetas_diff[0]
            thetas[1:3] *= thetas_diff[1:3]
            thetas[3:5] += thetas_diff[3:5]
            thetas[5] = thetas_diff[5]
            print(f"{i},thetas_diff,{','.join(map(str, thetas_diff))}")
            print(f"{i},thetas,{','.join(map(str, thetas))}")

    # save
    all_input.filter_transfer.save(out_path)
    spectra.save(out_path)
    with h5py.File(out_path) as f:
        f['meta/thetas_map'] = thetas


def cli():
    import defopt

    defopt.run(iterative_map)

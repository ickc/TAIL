from .container import PseudoSpectra, ModeCoupling, BeamSpectra, TheorySpectra, FilterTransfer

from pathlib import Path

FILTER_SPECTRA_TO_MAP_CASES = {
    'BB': ['simmaps_TTBB'],
    'EE': ['simmaps_TTEE'],
    'TT': ['simmaps_TTBB', 'simmaps_TTEE'],
}


class AllInput(object):

    def __init__(
        self,
        pseudo_spectra: PseudoSpectra,
        mode_coupling: ModeCoupling,
        beam_spectra: BeamSpectra,
        theory_spectra: TheorySpectra,
        *,
        filter_transfer: FilterTransfer = None,
        full: bool = True,
        cases: dict = FILTER_SPECTRA_TO_MAP_CASES
    ):
        '''load all input from post-making pipelines

        This essentially loads a superset of both `FilterTransferInput`
        and `SpectraInput` only without filter_transfer itself
        which is to be calculated and assigned to self.filter_transfer
        '''
        self.pseudo_spectra = pseudo_spectra
        self.mode_coupling = mode_coupling
        self.beam_spectra = beam_spectra
        self.theory_spectra = theory_spectra

        self.full = full
        self.cases = cases
        self.filter_transfer = filter_transfer

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        full: bool = True,
        cases: dict = FILTER_SPECTRA_TO_MAP_CASES
    ):
        '''construct from hdf5 file from `path`

        param dict cases: to be passed to constructor. See docstring from class itself
        '''

        pseudo_spectra = PseudoSpectra.load_h5(path, full=full)
        spectra = pseudo_spectra.spectra

        # mode_coupling: load the 6 "canonical" only
        mode_coupling = ModeCoupling.load(path, keys=(spectrum * 2 for spectrum in spectra))

        beam_spectra = BeamSpectra.load(path, keys=('l', 'all'))
        theory_spectra = TheorySpectra.load(path, keys=list(cases) + ['l'])

        return cls(
            pseudo_spectra, mode_coupling, beam_spectra, theory_spectra,
            full=full,
            cases=cases
        )


class FilterTransferInput(AllInput):

    def __init__(
        self,
        pseudo_spectra: PseudoSpectra,
        mode_coupling: ModeCoupling,
        beam_spectra: BeamSpectra,
        theory_spectra: TheorySpectra,
        *,
        full: bool = True,
        cases: dict = FILTER_SPECTRA_TO_MAP_CASES
    ):
        self.pseudo_spectra = pseudo_spectra
        self.mode_coupling = mode_coupling
        self.beam_spectra = beam_spectra
        self.theory_spectra = theory_spectra

        self.full = full
        self.cases = cases

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        full: bool = True,
        cases: dict = FILTER_SPECTRA_TO_MAP_CASES
    ):
        '''construct from hdf5 file from `path`

        param dict cases: to be passed to constructor. See docstring from class itself
        '''
        map_cases = set(sum(cases.values(), []))
        pseudo_spectra = PseudoSpectra.load_h5(path, map_cases=map_cases, full=full)

        mode_coupling_keys = [case * 2 for case in cases]
        mode_coupling = ModeCoupling.load(path, keys=mode_coupling_keys)

        beam_spectra = BeamSpectra.load(path, keys=('l', 'all'))
        theory_spectra = TheorySpectra.load(path, keys=list(cases) + ['l'])

        return cls(
            pseudo_spectra, mode_coupling, beam_spectra, theory_spectra,
            full=full,
            cases=cases
        )


class SpectraInput(AllInput):
    '''take all the inputs that is needed to compute spectra

    inputs:

    - pseudo spectra
        - meta:
            - sub_spectra
            - spectra
            - null_split
    - mode-coupling matrices
    - beam spectra
    - filter transfer functions
    '''

    def __init__(
        self,
        pseudo_spectra: PseudoSpectra,
        mode_coupling: ModeCoupling,
        beam_spectra: BeamSpectra,
        filter_transfer: FilterTransfer,
        *,
        full: bool = True,
    ):
        self.pseudo_spectra = pseudo_spectra
        self.mode_coupling = mode_coupling
        self.beam_spectra = beam_spectra
        self.filter_transfer = filter_transfer

        self.full = full

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        path_cache: Path = None,
        full: bool = True,
    ):
        '''
        param Path path_cache: path to HDF5 file containing filter transfer
        function. If None, assume it is the same as `path`.
        '''
        if path_cache is None:
            path_cache = path

        pseudo_spectra = PseudoSpectra.load_h5(path, full=full)
        spectra = pseudo_spectra.spectra

        # mode_coupling: load the 6 "canonical" only
        mode_coupling = ModeCoupling.load(path, keys=(spectrum * 2 for spectrum in spectra))

        beam_spectra = BeamSpectra.load(path, keys=('l', 'all'))

        filter_transfer = FilterTransfer.load(path_cache, full=full)

        return cls(
            pseudo_spectra, mode_coupling, beam_spectra, filter_transfer,
            full=full,
        )

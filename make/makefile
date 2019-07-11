P = 4
BINWIDTHS = 5 50
BINWIDTHS_FULL = 5 40 50 100 150 200 250 300 350 400 450 500 550
SPECTRAS = $(patsubst %,spectra-%.hdf5,$(BINWIDTHS))
FULLSPECTRAS = $(patsubst %,spectra-full-%.hdf5,$(BINWIDTHS_FULL))
FULLSPECTRAS_LEAK = $(patsubst %,spectra-full-leakage_subtracted-%.hdf5,$(BINWIDTHS_FULL))
FULLSPECTRAS_LEAK_THEORY = $(patsubst %,spectra-full-leakage_subtract_theory-%.hdf5,$(BINWIDTHS_FULL))

default: null full theory
all: null full leak theory spectra-full-leakage_subtract_theory-SPP.hdf5 spectra-full-leakage_subtracted-SPP.hdf5
null: $(SPECTRAS)
full: $(FULLSPECTRAS)
leak: $(FULLSPECTRAS_LEAK)
theory: $(FULLSPECTRAS_LEAK_THEORY)

args = --modecoupling modecoupling_norm.hdf5 --theory "$$SCRATCH/data/largepatch_high/class_output/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl_lensed.hdf5" --beam "$$SCRATCH/data/largepatch_high/jupiter_combined_bls_repack.hdf5"

clean:
	rm -f $(SPECTRAS) $(FULLSPECTRAS) $(FULLSPECTRAS_LEAK) $(FULLSPECTRAS_LEAK_THEORY) spectra-full-leakage_subtract_theory-SPP.hdf5 spectra-full-leakage_subtracted-SPP.hdf5

spectra-%.hdf5: filter_norm.hdf5 modecoupling_norm.hdf5 $(wildcard pseudo/*.hdf5)
	null_spectra.py $(args) -o $@ --filter-transfer $< --bin-width $* --pseudospectra 'pseudo/*.hdf5' -p $(P)

spectra-full-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full.hdf5

spectra-full-leakage_subtract_theory-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full_leakage_subtract_theory.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full_leakage_subtract_theory.hdf5

spectra-full-leakage_subtract_theory-SPP.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full_leakage_subtract_theory.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width 400 --pseudospectra pseudo_full_leakage_subtract_theory.hdf5 --l-min 100 --l-max 2500

spectra-full-leakage_subtracted-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full_leakage_subtracted.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full_leakage_subtracted.hdf5

spectra-full-leakage_subtracted-SPP.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full_leakage_subtracted.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width 400 --pseudospectra pseudo_full_leakage_subtracted.hdf5 --l-min 100 --l-max 2500

print-%:
	$(info $* = $($*))

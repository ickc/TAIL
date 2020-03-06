P = 4
BINWIDTHS = 5 50 100
BINWIDTHS_FULL = 5 40 50 100 150 200 250 300 350 400 450 500 550
SPECTRAS = $(patsubst %,spectra-%.hdf5,$(BINWIDTHS))
FULLSPECTRAS = $(patsubst %,spectra-full-%.hdf5,$(BINWIDTHS_FULL))
FULLSPECTRAS_LEAK = $(patsubst %,spectra-full-leakage_subtracted-%.hdf5,$(BINWIDTHS_FULL))
FULLSPECTRAS_LEAK_THEORY = $(patsubst %,spectra-full-leakage_subtract_theory-%.hdf5,$(BINWIDTHS_FULL))
SPECTRAS_PWF = $(patsubst %,spectra-pixel_window_function-%.hdf5,$(BINWIDTHS))
FULLSPECTRAS_PWF = $(patsubst %,spectra-full-pixel_window_function-%.hdf5,$(BINWIDTHS_FULL))
SPECTRAS_LEAK_THEORY_PWF = $(patsubst %,spectra-leakage_subtract_theory_TB_EB_BB_PWF-pixel_window_function-%.hdf5,$(BINWIDTHS))
FULLSPECTRAS_LEAK_THEORY_PWF = $(patsubst %,spectra-full-leakage_subtract_theory_TB_EB_BB_PWF-pixel_window_function-%.hdf5,$(BINWIDTHS_FULL))

default: null full pwf
all: null full leak theory spectra-full-leakage_subtract_theory-SPP.hdf5 spectra-full-leakage_subtracted-SPP.hdf5
null: $(SPECTRAS)
full: $(FULLSPECTRAS)
leak: $(FULLSPECTRAS_LEAK)
theory: $(FULLSPECTRAS_LEAK_THEORY)
pwf: $(SPECTRAS_PWF) $(FULLSPECTRAS_PWF)
pwf-theory: $(SPECTRAS_LEAK_THEORY_PWF) $(FULLSPECTRAS_LEAK_THEORY_PWF)

args_common = --modecoupling modecoupling_norm.hdf5 --theory "$$SCRATCH/data/largepatch_high/class_output/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl_lensed.hdf5"
args = $(args_common) --beam "$$SCRATCH/data/largepatch_high/jupiter_combined_bls_repack.hdf5"
args_pwf = $(args_common) --beam bl_pl.hdf5

clean:
	rm -f $(SPECTRAS) $(FULLSPECTRAS) $(FULLSPECTRAS_LEAK) $(FULLSPECTRAS_LEAK_THEORY) spectra-full-leakage_subtract_theory-SPP.hdf5 spectra-full-leakage_subtracted-SPP.hdf5

spectra-%.hdf5: filter_norm.hdf5 modecoupling_norm.hdf5 $(wildcard pseudo/*.hdf5)
	null_spectra.py $(args) -o $@ --filter-transfer $< --bin-width $* --pseudospectra 'pseudo/*.hdf5' -p $(P)

spectra-full-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full.hdf5
	null_spectra.py $(args) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full.hdf5

spectra-pixel_window_function-%.hdf5: filter_norm.hdf5 modecoupling_norm.hdf5 $(wildcard pseudo/*.hdf5)
	null_spectra.py $(args_pwf) -o $@ --filter-transfer $< --bin-width $* --pseudospectra 'pseudo/*.hdf5' -p $(P)

spectra-full-pixel_window_function-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full.hdf5
	null_spectra.py $(args_pwf) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full.hdf5

spectra-leakage_subtract_theory_TB_EB_BB_PWF-pixel_window_function-%.hdf5: filter_norm.hdf5 modecoupling_norm.hdf5 $(wildcard pseudo_leakage_subtract_theory_TB_EB_BB_PWF/*.hdf5)
	null_spectra.py $(args_pwf) -o $@ --filter-transfer $< --bin-width $* --pseudospectra 'pseudo_leakage_subtract_theory_TB_EB_BB_PWF/*.hdf5' -p $(P)

spectra-full-leakage_subtract_theory_TB_EB_BB_PWF-pixel_window_function-%.hdf5: filter_full_norm.hdf5 modecoupling_norm.hdf5 pseudo_full_leakage_subtract_theory_TB_EB_BB_PWF.hdf5
	null_spectra.py $(args_pwf) -o $@ --filter-transfer $< --compute-nl --return-w --bin-width $* --pseudospectra pseudo_full_leakage_subtract_theory_TB_EB_BB_PWF.hdf5

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

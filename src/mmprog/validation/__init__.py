"""Validators behind the ExternalValidation component.

Each module exposes one function that returns ``(summary dict, table or None)``. The
summary always carries ``status`` ("ran", "skipped" or "failed") and ``reason``, so the
run log and ``validation_summary.json`` say what was checked and what could not be.

    program_specificity   does the morphology prediction track the program, not any gene set of the
                          same expression level (expression-matched random sets; split-gene halves)
    spatial_fidelity      does the predicted map reproduce the measured spatial structure (Moran's I,
                          plasma-cell-dense niche contrast)
    robustness            do predictions survive intensity, blur, rotation and noise perturbations
    independent_cohort    morphology-only scoring of a cohort the model never saw (data-gated)
    protein_bridge        the CODEX / PhenoCycler bridge (data-gated; no public myeloma pairing exists)
"""

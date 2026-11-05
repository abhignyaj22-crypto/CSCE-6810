# Contributions established by this run

Run: `synthetic_v3`. Data: synthetic fixture, 6 patients, 18000 cells after QC and subsampling. Evaluation: patient-level 3-fold cross-validation.

## 1. A measured ceiling for every target

Split-half reliability (binomial thinning, Spearman-Brown corrected) gives the correlation an independent re-measurement of the same cells would reach. No image model can be expected to exceed its square root.

| target | median r_sb | attainable ceiling sqrt(r_sb) |
|---|---|---|
| anchor_adhesion_survival_cell_z | 0.232 | 0.482 |
| anchor_adhesion_survival_nb_z | 0.979 | 0.989 |
| anchor_cytotoxicity_cell_z | 0.485 | 0.696 |
| anchor_cytotoxicity_nb_z | 0.897 | 0.947 |
| anchor_pc_stress_survival_cell_z | 0.012 | 0.110 |
| anchor_pc_stress_survival_nb_z | 0.449 | 0.670 |
| anchor_t_exhaustion_cell_z | 0.052 | 0.228 |
| anchor_t_exhaustion_nb_z | 0.779 | 0.883 |
| anchor_wnt_camdr_cell_z | 0.634 | 0.796 |
| anchor_wnt_camdr_nb_z | 0.991 | 0.995 |
| anchor_wnt_core_cell_z | 0.288 | 0.537 |
| anchor_wnt_core_nb_z | 0.983 | 0.991 |
| prog_adhesion_survival_cell_z | 0.583 | 0.764 |
| prog_adhesion_survival_nb_z | 0.937 | 0.968 |
| prog_cytotoxicity_cell_z | 0.741 | 0.861 |
| prog_cytotoxicity_nb_z | 0.724 | 0.851 |
| prog_extra_0_cell_z | 0.797 | 0.893 |
| prog_extra_0_nb_z | 0.809 | 0.899 |
| prog_extra_1_cell_z | 0.763 | 0.873 |
| prog_extra_1_nb_z | 0.813 | 0.902 |
| prog_pc_stress_survival_cell_z | -0.013 | 0.000 |
| prog_pc_stress_survival_nb_z | 0.018 | 0.134 |
| prog_t_exhaustion_cell_z | 0.277 | 0.526 |
| prog_t_exhaustion_nb_z | 0.439 | 0.663 |
| prog_wnt_camdr_cell_z | 0.900 | 0.949 |
| prog_wnt_camdr_nb_z | 0.975 | 0.987 |
| prog_wnt_core_cell_z | 0.626 | 0.791 |
| prog_wnt_core_nb_z | 0.957 | 0.978 |

## 2. Held-out performance against baselines and ablations

| target | shuffled | composition_only | anchor_only_targets | align | no_smooth | no_comp | no_grid | no_neighbours | no_program_tokens | full | oracle | full 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_adhesion_survival_cell_z | nan | 0.124 | 0.384 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_adhesion_survival_nb_z | nan | 0.148 | 0.909 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_cytotoxicity_cell_z | nan | 0.606 | 0.223 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_cytotoxicity_nb_z | nan | -0.006 | 0.744 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_pc_stress_survival_cell_z | nan | 0.030 | 0.068 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_pc_stress_survival_nb_z | nan | 0.123 | 0.662 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_t_exhaustion_cell_z | nan | 0.114 | 0.068 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_t_exhaustion_nb_z | nan | -0.015 | 0.515 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_camdr_cell_z | nan | 0.150 | 0.511 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_camdr_nb_z | nan | 0.148 | 0.921 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_core_cell_z | nan | 0.112 | 0.430 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_core_nb_z | nan | 0.147 | 0.922 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| prog_adhesion_survival_cell_z | 0.003 | 0.445 | nan | 0.350 | 0.352 | 0.356 | 0.349 | 0.327 | 0.351 | 0.352 | 0.356 | (0.077, 0.651) |
| prog_adhesion_survival_nb_z | 0.008 | 0.172 | nan | 0.831 | 0.847 | 0.849 | 0.842 | 0.762 | 0.810 | 0.846 | 0.841 | (0.331, 0.94) |
| prog_cytotoxicity_cell_z | -0.007 | 0.701 | nan | 0.031 | 0.030 | 0.033 | 0.024 | 0.041 | 0.030 | 0.031 | 0.087 | (0.016, 0.101) |
| prog_cytotoxicity_nb_z | 0.008 | 0.037 | nan | 0.139 | 0.141 | 0.141 | 0.142 | 0.121 | 0.123 | 0.140 | 0.140 | (0.12, 0.463) |
| prog_extra_0_cell_z | -0.004 | 0.750 | nan | 0.083 | 0.083 | 0.079 | 0.086 | 0.063 | 0.089 | 0.083 | 0.081 | (0.022, 0.123) |
| prog_extra_0_nb_z | 0.000 | 0.004 | nan | 0.386 | 0.390 | 0.413 | 0.406 | 0.334 | 0.370 | 0.392 | 0.413 | (0.115, 0.47) |
| prog_extra_1_cell_z | 0.014 | 0.754 | nan | 0.081 | 0.081 | 0.080 | 0.081 | 0.082 | 0.079 | 0.081 | 0.083 | (0.013, 0.2) |
| prog_extra_1_nb_z | 0.009 | 0.108 | nan | 0.389 | 0.387 | 0.388 | 0.380 | 0.376 | 0.369 | 0.388 | 0.379 | (0.055, 0.682) |
| prog_pc_stress_survival_cell_z | 0.009 | 0.027 | nan | 0.026 | 0.017 | 0.016 | 0.014 | 0.027 | 0.020 | 0.019 | 0.028 | (0.01, 0.04) |
| prog_pc_stress_survival_nb_z | 0.013 | 0.046 | nan | 0.158 | 0.156 | 0.147 | 0.150 | 0.134 | 0.152 | 0.157 | 0.151 | (0.041, 0.22) |
| prog_t_exhaustion_cell_z | -0.009 | 0.123 | nan | 0.040 | 0.042 | 0.081 | 0.059 | 0.026 | 0.061 | 0.041 | 0.059 | (-0.022, 0.086) |
| prog_t_exhaustion_nb_z | -0.006 | -0.034 | nan | 0.224 | 0.221 | 0.332 | 0.161 | 0.193 | 0.319 | 0.218 | 0.202 | (-0.161, 0.501) |
| prog_wnt_camdr_cell_z | 0.008 | 0.497 | nan | 0.331 | 0.334 | 0.338 | 0.347 | 0.339 | 0.319 | 0.332 | 0.367 | (0.142, 0.382) |
| prog_wnt_camdr_nb_z | 0.002 | -0.035 | nan | 0.689 | 0.715 | 0.743 | 0.725 | 0.620 | 0.710 | 0.718 | 0.760 | (0.371, 0.76) |
| prog_wnt_core_cell_z | 0.005 | 0.433 | nan | 0.377 | 0.378 | 0.382 | 0.377 | 0.353 | 0.377 | 0.379 | 0.376 | (0.101, 0.682) |
| prog_wnt_core_nb_z | -0.001 | 0.178 | nan | 0.871 | 0.874 | 0.865 | 0.864 | 0.806 | 0.837 | 0.875 | 0.858 | (0.496, 0.945) |

Reliability-adjusted (r divided by sqrt of the median noise ceiling):

| target | shuffled | composition_only | anchor_only_targets | align | no_smooth | no_comp | no_grid | no_neighbours | no_program_tokens | full | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_adhesion_survival_cell_z | nan | 0.259 | 0.796 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_adhesion_survival_nb_z | nan | 0.150 | 0.917 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_cytotoxicity_cell_z | nan | 0.823 | 0.303 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_cytotoxicity_nb_z | nan | -0.006 | 0.785 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_pc_stress_survival_cell_z | nan | 0.133 | 0.304 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_pc_stress_survival_nb_z | nan | 0.184 | 0.954 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_t_exhaustion_cell_z | nan | 0.503 | 0.272 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_t_exhaustion_nb_z | nan | -0.019 | 0.563 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_camdr_cell_z | nan | 0.189 | 0.642 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_camdr_nb_z | nan | 0.149 | 0.924 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_core_cell_z | nan | 0.210 | 0.802 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_core_nb_z | nan | 0.148 | 0.928 | nan | nan | nan | nan | nan | nan | nan | nan |
| prog_adhesion_survival_cell_z | 0.004 | 0.620 | nan | 0.459 | 0.462 | 0.467 | 0.458 | 0.428 | 0.460 | 0.461 | 0.466 |
| prog_adhesion_survival_nb_z | 0.009 | 0.178 | nan | 0.859 | 0.875 | 0.877 | 0.869 | 0.787 | 0.836 | 0.874 | 0.869 |
| prog_cytotoxicity_cell_z | -0.009 | 0.814 | nan | 0.036 | 0.035 | 0.039 | 0.028 | 0.048 | 0.035 | 0.036 | 0.100 |
| prog_cytotoxicity_nb_z | 0.010 | 0.043 | nan | 0.158 | 0.160 | 0.160 | 0.162 | 0.134 | 0.140 | 0.159 | 0.154 |
| prog_extra_0_cell_z | -0.004 | 0.840 | nan | 0.093 | 0.093 | 0.088 | 0.096 | 0.070 | 0.100 | 0.093 | 0.090 |
| prog_extra_0_nb_z | 0.000 | 0.004 | nan | 0.448 | 0.446 | 0.451 | 0.442 | 0.369 | 0.402 | 0.448 | 0.449 |
| prog_extra_1_cell_z | 0.017 | 0.863 | nan | 0.092 | 0.093 | 0.092 | 0.092 | 0.094 | 0.091 | 0.092 | 0.095 |
| prog_extra_1_nb_z | 0.010 | 0.121 | nan | 0.430 | 0.428 | 0.429 | 0.420 | 0.416 | 0.408 | 0.429 | 0.419 |
| prog_pc_stress_survival_cell_z | 0.041 | 0.119 | nan | 0.118 | 0.077 | 0.073 | 0.063 | 0.122 | 0.089 | 0.086 | 0.124 |
| prog_pc_stress_survival_nb_z | 0.055 | 0.169 | nan | 0.624 | 0.617 | 0.636 | 0.619 | 0.539 | 0.649 | 0.620 | 0.597 |
| prog_t_exhaustion_cell_z | -0.025 | 0.237 | nan | 0.074 | 0.085 | 0.165 | 0.133 | 0.058 | 0.168 | 0.082 | 0.108 |
| prog_t_exhaustion_nb_z | -0.009 | -0.045 | nan | 0.334 | 0.328 | 0.647 | 0.312 | 0.283 | 0.619 | 0.324 | 0.357 |
| prog_wnt_camdr_cell_z | 0.009 | 0.548 | nan | 0.348 | 0.351 | 0.356 | 0.365 | 0.356 | 0.336 | 0.349 | 0.386 |
| prog_wnt_camdr_nb_z | 0.002 | -0.036 | nan | 0.694 | 0.720 | 0.749 | 0.730 | 0.626 | 0.717 | 0.723 | 0.769 |
| prog_wnt_core_cell_z | 0.007 | 0.615 | nan | 0.477 | 0.478 | 0.483 | 0.477 | 0.446 | 0.477 | 0.480 | 0.476 |
| prog_wnt_core_nb_z | -0.001 | 0.182 | nan | 0.891 | 0.893 | 0.885 | 0.884 | 0.824 | 0.856 | 0.894 | 0.877 |

Seed stability of the full model (mean and standard deviation of the per-patient median r):

- anchor_adhesion_survival_cell_z: nan +/- nan
- anchor_adhesion_survival_nb_z: nan +/- nan
- anchor_cytotoxicity_cell_z: nan +/- nan
- anchor_cytotoxicity_nb_z: nan +/- nan
- anchor_pc_stress_survival_cell_z: nan +/- nan
- anchor_pc_stress_survival_nb_z: nan +/- nan
- anchor_t_exhaustion_cell_z: nan +/- nan
- anchor_t_exhaustion_nb_z: nan +/- nan
- anchor_wnt_camdr_cell_z: nan +/- nan
- anchor_wnt_camdr_nb_z: nan +/- nan
- anchor_wnt_core_cell_z: nan +/- nan
- anchor_wnt_core_nb_z: nan +/- nan
- prog_adhesion_survival_cell_z: 0.355 +/- 0.005
- prog_adhesion_survival_nb_z: 0.831 +/- 0.021
- prog_cytotoxicity_cell_z: 0.029 +/- 0.003
- prog_cytotoxicity_nb_z: 0.141 +/- 0.002
- prog_extra_0_cell_z: 0.087 +/- 0.005
- prog_extra_0_nb_z: 0.387 +/- 0.007
- prog_extra_1_cell_z: 0.082 +/- 0.001
- prog_extra_1_nb_z: 0.384 +/- 0.005
- prog_pc_stress_survival_cell_z: 0.018 +/- 0.001
- prog_pc_stress_survival_nb_z: 0.151 +/- 0.009
- prog_t_exhaustion_cell_z: 0.032 +/- 0.013
- prog_t_exhaustion_nb_z: 0.205 +/- 0.018
- prog_wnt_camdr_cell_z: 0.337 +/- 0.007
- prog_wnt_camdr_nb_z: 0.724 +/- 0.009
- prog_wnt_core_cell_z: 0.379 +/- 0.000
- prog_wnt_core_nb_z: 0.861 +/- 0.020

## 3. Decision rules applied to `prog_wnt_camdr_nb_z`

- Main target: prog_wnt_camdr_nb_z. Full model per-patient median r = 0.718 (reliability-adjusted 0.723).
- ABOVE CHANCE: full (0.718) exceeds the shuffled-label baseline (0.002) by more than 0.05.
- MORPHOLOGY BEYOND IDENTITY: full (0.718) exceeds composition_only (-0.035); morphology carries program information beyond cell type. Report as the main finding.
- NO IDENTITY BOTTLENECK: oracle (0.760) is not above full (0.718) by 0.05.
- SPATIAL PRIOR USEFUL: full (0.718) exceeds no_neighbours (0.620).
- Across all targets: 13/16 exceed the shuffled baseline by 0.05.
- ABLATION no_program_tokens: full (0.718) vs without (0.710); the program-token cross-attention neither helps nor hurts 0.008 on prog_wnt_camdr_nb_z.
- ABLATION no_grid: full (0.718) vs without (0.725); the patch-block tokens neither helps nor hurts 0.007 on prog_wnt_camdr_nb_z.
- ABLATION no_smooth: full (0.718) vs without (0.715); the neighbourhood consistency term neither helps nor hurts 0.003 on prog_wnt_camdr_nb_z.
- ABLATION align: full (0.718) vs with the contrastive alignment loss switched on (0.689); switching the alignment loss on worsens prog_wnt_camdr_nb_z by 0.029.
- LEARNED VS FIXED TARGETS: learned prog_wnt_camdr_nb_z r = 0.718; fixed-set anchor_wnt_camdr_nb_z r = 0.921; learned programs are not more predictable from morphology.

## 4. What this run supports

- Nuclear morphology carries information about the resistance-associated program beyond what cell identity alone explains (full model above both the shuffled-label bound and the composition-only baseline).
- Patient-level program burden versus disease stage: Spearman rho = 0.970, p = 0.00132 (n = 6).

## 5. Where in the neighbourhood the signal lives (spatially regularized boosting)

Boosting over decision stumps on a grid of patch blocks, with the spatial regulariser of Xiang et al. (2009); lambda = 0 is the AdaBoost control on the same stumps. Accuracy is the per-patient median held-out metric; coherence is Moran's I of the importance map over the block grid (about 0 for scattered selection, positive for clustered selection).

| loss | lambda | metric | value | coherence (Moran's I) | active blocks |
|---|---|---|---|---|---|
| exponential | 0.0 | auc_patient_median | 0.849 | -0.031 | 81 |
| exponential | 0.1 | auc_patient_median | 0.840 | 0.003 | 95 |
| exponential | 0.5 | auc_patient_median | 0.845 | 0.131 | 85 |
| squared | 0.0 | pearson_patient_median | 0.474 | -0.013 | 84 |
| squared | 0.1 | pearson_patient_median | 0.507 | 0.021 | 97 |
| squared | 0.5 | pearson_patient_median | 0.508 | 0.231 | 81 |

Importance by block-centre distance from the nucleus (centre < 5 um, inner 5 to 10 um, outer >= 10 um); share of total, and share per block in brackets:

| model | plane | centre | inner ring | outer ring |
|---|---|---|---|---|
| exponential_lam0.0 | mean | 0.149 (0.037) | 0.251 (0.009) | 0.600 (0.019) |
| exponential_lam0.0 | std | 0.069 (0.017) | 0.380 (0.014) | 0.551 (0.017) |
| exponential_lam0.1 | mean | 0.168 (0.042) | 0.292 (0.010) | 0.540 (0.017) |
| exponential_lam0.1 | std | 0.139 (0.035) | 0.482 (0.017) | 0.379 (0.012) |
| exponential_lam0.5 | mean | 0.193 (0.048) | 0.470 (0.017) | 0.336 (0.011) |
| exponential_lam0.5 | std | 0.168 (0.042) | 0.705 (0.025) | 0.127 (0.004) |
| squared_lam0.0 | mean | 0.424 (0.106) | 0.161 (0.006) | 0.415 (0.013) |
| squared_lam0.0 | std | 0.199 (0.050) | 0.266 (0.009) | 0.536 (0.017) |
| squared_lam0.1 | mean | 0.372 (0.093) | 0.242 (0.009) | 0.386 (0.012) |
| squared_lam0.1 | std | 0.157 (0.039) | 0.319 (0.011) | 0.524 (0.016) |
| squared_lam0.5 | mean | 0.400 (0.100) | 0.474 (0.017) | 0.126 (0.004) |
| squared_lam0.5 | std | 0.159 (0.040) | 0.474 (0.017) | 0.367 (0.011) |

## 6. Association of the predicted program vector with disease stage (patient level)

PERMANOVA (Anderson 2001; Euclidean distance between patient-level burden vectors, permutation p-value) asks whether the vector of predicted program burdens differs between stages; the Jonckheere-Terpstra trend test asks, per program, whether burden rises along the ordered stages, with Benjamini-Hochberg q-values across programs. Both are distribution-free and suited to the cohort size. Rows with `vectors = measured` repeat the tests on the measured programs, so the predicted and the measured association can be compared directly.

| vectors | test | program | statistic | p_value | q_value | r2_or_rho | n_patients | n_groups | groups | note |
|---||---||---||---||---||---||---||---||---||---||---|
| predicted | permanova | all | 12.904 | 0.085 | nan | 0.896 | 6 | 3 | MM:2;SMM:2;control:2 | pseudo-F on 7 z-scored programs, Euclidean, 199 permutations |
| predicted | jt_trend | prog_wnt_core_nb_z | 0.000 | 0.020 | 0.035 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_adhesion_survival_nb_z | 0.000 | 0.020 | 0.035 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_pc_stress_survival_nb_z | 11.000 | 0.075 | 0.075 | 0.837 | 6 | 3 | MM:2;SMM:2;control:2 | increasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_t_exhaustion_nb_z | 0.000 | 0.050 | 0.058 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_cytotoxicity_nb_z | 0.000 | 0.025 | 0.035 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_extra_0_nb_z | 0.000 | 0.020 | 0.035 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| predicted | jt_trend | prog_extra_1_nb_z | 0.000 | 0.025 | 0.035 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | permanova | all | 10.500 | 0.085 | nan | 0.875 | 6 | 3 | MM:2;SMM:2;control:2 | pseudo-F on 7 z-scored programs, Euclidean, 199 permutations |
| measured | jt_trend | prog_wnt_core_nb_z | 2.000 | 0.220 | 0.268 | -0.717 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_adhesion_survival_nb_z | 0.000 | 0.020 | 0.044 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_pc_stress_survival_nb_z | 8.000 | 0.660 | 0.660 | 0.478 | 6 | 3 | MM:2;SMM:2;control:2 | increasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_t_exhaustion_nb_z | 2.000 | 0.230 | 0.268 | -0.717 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_cytotoxicity_nb_z | 0.000 | 0.025 | 0.044 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_extra_0_nb_z | 0.000 | 0.020 | 0.044 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |
| measured | jt_trend | prog_extra_1_nb_z | 0.000 | 0.025 | 0.044 | -0.956 | 6 | 3 | MM:2;SMM:2;control:2 | decreasing along the stage order; expected JT 6.0; 199 permutations |

With one trephine per patient the unit of analysis is the patient, so these tests have the cohort's patient count as their sample size and are reported as supporting evidence only.

## 7. Neighbourhood motifs enriched for the main program (byte-pair encoding over lineage sequences)

| motif | difference (with minus without) | permutation p | patients with motif | fraction of patients with positive difference |
|---|---|---|---|---|
| plasma|plasma|plasma|plasma | 1.110 | 1.000 | 2 | 0.000 |
| plasma|plasma | 0.958 | 1.000 | 6 | 0.333 |
| monocyte|B | -0.447 | 0.005 | 6 | 0.167 |
| monocyte|erythroid | -0.437 | 0.060 | 6 | 0.167 |
| B|B | -0.407 | 0.005 | 6 | 0.167 |
| erythroid|erythroid | -0.404 | 0.005 | 6 | 0.000 |
| monocyte|granulocyte | -0.399 | 0.274 | 6 | 0.333 |
| T|B | -0.362 | 0.055 | 6 | 0.333 |

Motif discovery is exploratory: byte-pair encoding is a tokenisation heuristic applied to ordered neighbour sequences, and enrichment is descriptive.

## 7a. External validation (modality-agnostic validators)

| validator | status | verdict | reason or headline |
|---|---|---|---|
| program_specificity | ran | pass | median_z = 49.624, median_r_target = 0.718 (bar: median_z > 2 on the main target and both split-gene halves correlate positively) |
| spatial_fidelity | ran | pass | median_moran_ratio = 0.984, cohens_d_pred = 0.736, cohens_d_measured = 0.875 (bar: 0.5 <= median Moran ratio <= 2 and the plasma-dense niche contrast has the measured sign, on the main target) |
| perturbation_robustness | ran | fail | min_spearman_main = 0.417, worst_perturbation_main = gamma_0.8 (bar: Spearman between perturbed and clean predictions > 0.9 on the main target under every perturbation) |
| independent_cohort | skipped |  | no independent cohort manifest; set validation.validation_cohort_manifest to a JSON manifest (Xenium or Visium HD rows with local paths, or {"demo": {...}} for a synthetic cohort). The Visium HD trephine cohort of Ohlstrom et al. 2026 (preprint) is embargoed as of writing (docs/DATA.md) |
| protein_bridge | skipped |  | set codex_table (cell-by-marker CSV/parquet with x, y) and codex_dapi (DAPI image of the same section); no public dataset pairs CODEX with Xenium on the same myeloma trephine |

Program specificity per target (median over patients): the prediction's correlation with the real target against expression-matched random gene sets, and with each random half of the program's genes.

| target | median r_target | median null mean | median z | median r_half_a | median r_half_b |
|---|---|---|---|---|---|
| prog_adhesion_survival_nb_z | 0.846 | 0.491 | 1.497 | 0.817 | 0.836 |
| prog_cytotoxicity_nb_z | 0.140 | 0.372 | 0.588 | 0.264 | 0.218 |
| prog_extra_0_nb_z | 0.392 | nan | nan | 0.548 | 0.561 |
| prog_extra_1_nb_z | 0.388 | 0.182 | 16.161 | 0.463 | 0.607 |
| prog_pc_stress_survival_nb_z | 0.157 | 0.179 | 0.726 | 0.150 | 0.158 |
| prog_t_exhaustion_nb_z | 0.218 | -0.166 | 8.400 | 0.033 | 0.187 |
| prog_wnt_camdr_nb_z | 0.718 | -0.191 | 49.624 | 0.712 | 0.650 |
| prog_wnt_core_nb_z | 0.875 | 0.248 | 2.424 | 0.848 | 0.867 |

Spatial fidelity (median over patients): Moran's I of the predicted and the measured map.

| target | Moran's I predicted | Moran's I measured | ratio |
|---|---|---|---|
| prog_adhesion_survival_nb_z | 0.964 | 0.960 | 1.004 |
| prog_cytotoxicity_nb_z | 0.940 | 0.865 | 1.093 |
| prog_extra_0_nb_z | 0.938 | 0.867 | 1.079 |
| prog_extra_1_nb_z | 0.967 | 0.873 | 1.114 |
| prog_pc_stress_survival_nb_z | 0.958 | 0.847 | 1.116 |
| prog_t_exhaustion_nb_z | 0.925 | 0.875 | 1.070 |
| prog_wnt_camdr_nb_z | 0.915 | 0.939 | 0.984 |
| prog_wnt_core_nb_z | 0.963 | 0.960 | 1.003 |

Plasma-cell-dense niche contrast (Cohen's d, dense versus other neighbourhoods):

| target | d predicted | d measured | sign agreement |
|---|---|---|---|
| prog_wnt_camdr_nb_z | 0.736 | 0.875 | True |
| prog_wnt_core_nb_z | -1.190 | -0.955 | True |
| prog_adhesion_survival_nb_z | -1.326 | -1.076 | True |
| prog_pc_stress_survival_nb_z | 1.083 | 0.157 | True |
| prog_t_exhaustion_nb_z | -1.025 | -0.253 | True |
| prog_cytotoxicity_nb_z | -0.396 | -0.327 | True |
| prog_extra_0_nb_z | -0.559 | -0.430 | True |
| prog_extra_1_nb_z | -0.056 | -0.077 | True |

Perturbation robustness on `prog_wnt_camdr_nb_z`: Spearman between perturbed and clean predictions on the same nuclei.

| perturbation | Spearman vs clean | mean absolute shift (z units) |
|---|---|---|
| gamma_0.8 | 0.417 | 0.273 |
| rot90 | 0.912 | 0.034 |

A skipped validator is a gap in the evidence, listed with its reason; the pass bars are the validators' own stated criteria (docs/research/VALIDATION_METHODS.md) and the `external_validation` release gate reads them.

## 8a. Calibration of the predictive uncertainty

Empirical coverage of the predictive intervals on held-out patients. A well calibrated model covers the target at the nominal level; coverage is reported per patient because cells within one patient are not exchangeable with cells of another.

| target | level | coverage_pooled | coverage_patient_median | coverage_patient_min | coverage_patient_max | mean_std | n_cells | n_patients | kind | q | ece |
|---||---||---||---||---||---||---||---||---||---||---||---|
| prog_wnt_camdr_cell_z | 0.500 | 0.515 | 0.431 | 0.366 | 0.744 | 0.824 | 18000.000 | 6.000 | gaussian | nan | 0.026 |
| prog_wnt_core_cell_z | 0.500 | 0.449 | 0.433 | 0.395 | 0.514 | 0.908 | 18000.000 | 6.000 | gaussian | nan | 0.032 |
| prog_adhesion_survival_cell_z | 0.500 | 0.457 | 0.450 | 0.418 | 0.502 | 0.911 | 18000.000 | 6.000 | gaussian | nan | 0.026 |
| prog_pc_stress_survival_cell_z | 0.500 | 0.435 | 0.431 | 0.418 | 0.460 | 1.016 | 18000.000 | 6.000 | gaussian | nan | 0.032 |
| prog_t_exhaustion_cell_z | 0.500 | 0.502 | 0.502 | 0.485 | 0.514 | 1.011 | 18000.000 | 6.000 | gaussian | nan | 0.021 |
| prog_cytotoxicity_cell_z | 0.500 | 0.583 | 0.584 | 0.566 | 0.600 | 1.020 | 18000.000 | 6.000 | gaussian | nan | 0.050 |
| prog_extra_0_cell_z | 0.500 | 0.537 | 0.531 | 0.499 | 0.588 | 1.009 | 18000.000 | 6.000 | gaussian | nan | 0.023 |
| prog_extra_1_cell_z | 0.500 | 0.517 | 0.512 | 0.479 | 0.565 | 1.011 | 18000.000 | 6.000 | gaussian | nan | 0.009 |
| prog_wnt_camdr_nb_z | 0.500 | 0.549 | 0.529 | 0.459 | 0.685 | 0.368 | 18000.000 | 6.000 | gaussian | nan | 0.030 |
| prog_wnt_core_nb_z | 0.500 | 0.498 | 0.521 | 0.377 | 0.563 | 0.481 | 18000.000 | 6.000 | gaussian | nan | 0.005 |
| prog_adhesion_survival_nb_z | 0.500 | 0.484 | 0.492 | 0.425 | 0.523 | 0.508 | 18000.000 | 6.000 | gaussian | nan | 0.012 |
| prog_pc_stress_survival_nb_z | 0.500 | 0.503 | 0.503 | 0.457 | 0.555 | 0.988 | 18000.000 | 6.000 | gaussian | nan | 0.002 |
| prog_t_exhaustion_nb_z | 0.500 | 0.505 | 0.512 | 0.454 | 0.543 | 0.951 | 18000.000 | 6.000 | gaussian | nan | 0.003 |
| prog_cytotoxicity_nb_z | 0.500 | 0.514 | 0.518 | 0.434 | 0.620 | 0.890 | 18000.000 | 6.000 | gaussian | nan | 0.008 |
| prog_extra_0_nb_z | 0.500 | 0.526 | 0.530 | 0.473 | 0.576 | 0.803 | 18000.000 | 6.000 | gaussian | nan | 0.016 |
| prog_extra_1_nb_z | 0.500 | 0.510 | 0.514 | 0.462 | 0.544 | 0.849 | 18000.000 | 6.000 | gaussian | nan | 0.010 |
| prog_wnt_camdr_cell_z | 0.900 | 0.916 | 0.934 | 0.846 | 0.965 | 0.824 | 18000.000 | 6.000 | gaussian | nan | 0.026 |
| prog_wnt_core_cell_z | 0.900 | 0.929 | 0.929 | 0.895 | 0.961 | 0.908 | 18000.000 | 6.000 | gaussian | nan | 0.032 |
| prog_adhesion_survival_cell_z | 0.900 | 0.920 | 0.918 | 0.888 | 0.963 | 0.911 | 18000.000 | 6.000 | gaussian | nan | 0.026 |
| prog_pc_stress_survival_cell_z | 0.900 | 0.891 | 0.887 | 0.866 | 0.921 | 1.016 | 18000.000 | 6.000 | gaussian | nan | 0.032 |
| prog_t_exhaustion_cell_z | 0.900 | 0.920 | 0.920 | 0.913 | 0.927 | 1.011 | 18000.000 | 6.000 | gaussian | nan | 0.021 |
| prog_cytotoxicity_cell_z | 0.900 | 0.901 | 0.897 | 0.869 | 0.936 | 1.020 | 18000.000 | 6.000 | gaussian | nan | 0.050 |
| prog_extra_0_cell_z | 0.900 | 0.902 | 0.898 | 0.875 | 0.933 | 1.009 | 18000.000 | 6.000 | gaussian | nan | 0.023 |
| prog_extra_1_cell_z | 0.900 | 0.903 | 0.901 | 0.885 | 0.925 | 1.011 | 18000.000 | 6.000 | gaussian | nan | 0.009 |
| prog_wnt_camdr_nb_z | 0.900 | 0.914 | 0.909 | 0.854 | 0.978 | 0.368 | 18000.000 | 6.000 | gaussian | nan | 0.030 |
| prog_wnt_core_nb_z | 0.900 | 0.913 | 0.926 | 0.854 | 0.953 | 0.481 | 18000.000 | 6.000 | gaussian | nan | 0.005 |
| prog_adhesion_survival_nb_z | 0.900 | 0.916 | 0.910 | 0.889 | 0.948 | 0.508 | 18000.000 | 6.000 | gaussian | nan | 0.012 |
| prog_pc_stress_survival_nb_z | 0.900 | 0.899 | 0.898 | 0.854 | 0.947 | 0.988 | 18000.000 | 6.000 | gaussian | nan | 0.002 |
| prog_t_exhaustion_nb_z | 0.900 | 0.899 | 0.907 | 0.845 | 0.931 | 0.951 | 18000.000 | 6.000 | gaussian | nan | 0.003 |
| prog_cytotoxicity_nb_z | 0.900 | 0.904 | 0.910 | 0.822 | 0.962 | 0.890 | 18000.000 | 6.000 | gaussian | nan | 0.008 |
| prog_extra_0_nb_z | 0.900 | 0.912 | 0.916 | 0.884 | 0.944 | 0.803 | 18000.000 | 6.000 | gaussian | nan | 0.016 |
| prog_extra_1_nb_z | 0.900 | 0.911 | 0.911 | 0.878 | 0.948 | 0.849 | 18000.000 | 6.000 | gaussian | nan | 0.010 |
| prog_adhesion_survival_cell_z | 0.900 | 0.881 | 0.913 | 0.784 | nan | nan | nan | nan | conformal | 1.506 | 0.026 |
| prog_adhesion_survival_nb_z | 0.900 | 0.902 | 0.914 | 0.874 | nan | nan | nan | nan | conformal | 1.574 | 0.012 |
| prog_cytotoxicity_cell_z | 0.900 | 0.902 | 0.899 | 0.866 | nan | nan | nan | nan | conformal | 1.627 | 0.050 |
| prog_cytotoxicity_nb_z | 0.900 | 0.885 | 0.891 | 0.812 | nan | nan | nan | nan | conformal | 1.657 | 0.008 |
| prog_extra_0_cell_z | 0.900 | 0.899 | 0.913 | 0.856 | nan | nan | nan | nan | conformal | 1.604 | 0.023 |
| prog_extra_0_nb_z | 0.900 | 0.909 | 0.910 | 0.874 | nan | nan | nan | nan | conformal | 1.617 | 0.016 |
| prog_extra_1_cell_z | 0.900 | 0.907 | 0.906 | 0.895 | nan | nan | nan | nan | conformal | 1.675 | 0.009 |
| prog_extra_1_nb_z | 0.900 | 0.908 | 0.916 | 0.875 | nan | nan | nan | nan | conformal | 1.648 | 0.010 |

## 8b. Gene backbone of each program

Genes whose learned weight survives stability selection over patient subsamples, with the literature evidence grade from resources/gene_annotations.json. A backbone gene whose evidence points toward sensitivity rather than resistance is shown with its grade so the reader can weigh it.

| program | gene | selection_fraction | weight_relative_to_max | evidence_grade | gene_class |
|---||---||---||---||---||---|
| wnt_camdr | BCL2 | 1.000 | 0.498 | established | survival_tf |
| wnt_camdr | CDH2 | 1.000 | 0.362 | single-study | adhesion |
| wnt_camdr | DKK1 | 1.000 | 0.365 | established | wnt_antagonist |
| wnt_camdr | FRZB | 1.000 | 0.283 | single-study | wnt_antagonist |
| wnt_camdr | FZD3 | 1.000 | 0.682 | single-study | wnt_receptor |
| wnt_camdr | IGF1 | 1.000 | 0.557 | established | cytokine_growth_factor |
| wnt_camdr | IRF4 | 1.000 | 0.691 | established | survival_tf |
| wnt_camdr | LRP6 | 1.000 | 0.722 | single-study | wnt_receptor |
| wnt_camdr | NCAM1 | 1.000 | 0.349 | single-study | adhesion |
| wnt_camdr | MIF | 1.000 | 0.748 | single-study | cytokine_growth_factor |
| wnt_core | DKK1 | 1.000 | 0.841 | established | wnt_antagonist |
| wnt_core | FRZB | 1.000 | 0.945 | single-study | wnt_antagonist |
| wnt_core | FZD3 | 1.000 | 0.547 | single-study | wnt_receptor |
| wnt_core | LRP6 | 1.000 | 0.545 | single-study | wnt_receptor |
| wnt_core | ROR2 | 1.000 | 0.496 | single-study | wnt_receptor |
| wnt_core | SFRP1 | 1.000 | 0.523 | established | wnt_antagonist |
| wnt_core | SFRP2 | 1.000 | 0.903 | established | wnt_antagonist |
| wnt_core | WNT10A | 1.000 | 0.510 | single-study | wnt_ligand |
| wnt_core | WNT5A | 1.000 | 0.440 | single-study | wnt_ligand |
| wnt_core | WNT5B | 1.000 | 1.000 | none | wnt_ligand |
| adhesion_survival | BCL2 | 1.000 | 0.919 | established | survival_tf |
| adhesion_survival | CDH2 | 1.000 | 1.000 | single-study | adhesion |
| adhesion_survival | IFNAR2 | 1.000 | 0.776 | single-study | cytokine_growth_factor |
| adhesion_survival | IGF1 | 1.000 | 0.827 | established | cytokine_growth_factor |
| adhesion_survival | IRF4 | 1.000 | 0.664 | established | survival_tf |
| adhesion_survival | MIF | 1.000 | 0.642 | single-study | cytokine_growth_factor |
| adhesion_survival | NCAM1 | 1.000 | 0.996 | single-study | adhesion |
| adhesion_survival | VEGFB | 1.000 | 0.812 | none | cytokine_growth_factor |
| pc_stress_survival | DDIT4 | 1.000 | 1.000 | none | stress_regulator |
| pc_stress_survival | DEPTOR | 1.000 | 0.665 | established | stress_regulator |
| pc_stress_survival | EIF2AK3 | 1.000 | 0.975 | single-study | stress_regulator |
| pc_stress_survival | IL16 | 1.000 | 0.844 | none | cytokine_growth_factor |
| pc_stress_survival | TLR10 | 1.000 | 0.837 | none | immune_receptor |
| pc_stress_survival | TNFRSF13B | 1.000 | 0.871 | single-study | immune_receptor |
| t_exhaustion | AHR | 1.000 | 0.910 | none | immunoregulatory |
| t_exhaustion | CD27 | 1.000 | 0.829 | none | costim_marker |
| t_exhaustion | CD55 | 1.000 | 1.000 | none | immunoregulatory |
| t_exhaustion | CTLA4 | 1.000 | 0.814 | single-study | checkpoint |
| t_exhaustion | FOXP3 | 1.000 | 0.787 | single-study | treg_marker |
| t_exhaustion | IL2RA | 1.000 | 0.910 | single-study | treg_marker |
| t_exhaustion | LAG3 | 1.000 | 0.844 | single-study | checkpoint |
| t_exhaustion | TNFAIP3 | 1.000 | 0.773 | none | immunoregulatory |
| cytotoxicity | CD8A | 1.000 | 0.327 | none | lineage_tf |
| cytotoxicity | CD8B | 1.000 | 0.306 | none | lineage_tf |
| cytotoxicity | DEFA3 | 1.000 | 0.384 | none | neutrophil_granule |
| cytotoxicity | CTSW | 1.000 | 0.957 | none | cytotoxic_effector |
| cytotoxicity | DEFA4 | 1.000 | 0.340 | none | neutrophil_granule |
| cytotoxicity | EOMES | 1.000 | 0.263 | none | lineage_tf |
| cytotoxicity | TBX21 | 1.000 | 0.275 | none | lineage_tf |
| cytotoxicity | GNLY | 1.000 | 0.940 | none | cytotoxic_effector |
| cytotoxicity | GZMA | 1.000 | 0.933 | none | cytotoxic_effector |
| cytotoxicity | GZMB | 1.000 | 0.940 | none | cytotoxic_effector |
| extra_0 | ACHE | 1.000 | 0.346 |  |  |
| extra_0 | ACTA2 | 1.000 | 1.000 |  |  |
| extra_0 | AHSP | 1.000 | 0.337 |  |  |
| extra_0 | AZU1 | 1.000 | 0.345 |  |  |
| extra_0 | BCL11A | 1.000 | 0.252 |  |  |
| extra_0 | BGLAP | 1.000 | 0.961 |  |  |
| extra_0 | BIRC3 | 1.000 | 0.247 |  |  |
| extra_0 | C5AR1 | 1.000 | 0.642 |  |  |
| extra_0 | CA1 | 1.000 | 0.392 |  |  |
| extra_0 | CA2 | 1.000 | 0.459 |  |  |
| extra_1 | ACHE | 1.000 | 0.780 |  |  |
| extra_1 | ACTA2 | 1.000 | 0.101 |  |  |
| extra_1 | AHR | 1.000 | 0.097 | none | immunoregulatory |
| extra_1 | AHSP | 1.000 | 0.784 |  |  |
| extra_1 | AZU1 | 1.000 | 0.776 |  |  |
| extra_1 | BCL11A | 1.000 | 0.882 |  |  |
| extra_1 | BGLAP | 1.000 | 0.145 |  |  |
| extra_1 | BIRC3 | 1.000 | 0.888 |  |  |
| extra_1 | C5AR1 | 1.000 | 0.468 |  |  |
| extra_1 | CA1 | 1.000 | 0.731 |  |  |

## 8c. Pathway annotation of the learned programs

Hypergeometric enrichment of each learned program's backbone against the curated gene sets, with the measured panel as background.

| program | gene_set | backbone_size | set_size_in_panel | overlap | p_value | overlapping_genes | q_value |
|---||---||---||---||---||---||---||---|
| adhesion_survival | adhesion_survival | 8 | 8 | 8 | 0.000 | BCL2, CDH2, IFNAR2, IGF1, IRF4, MIF, NCAM1, VEGFB | 0.000 |
| adhesion_survival | wnt_camdr | 8 | 18 | 8 | 0.000 | BCL2, CDH2, IFNAR2, IGF1, IRF4, MIF, NCAM1, VEGFB | 0.000 |
| adhesion_survival | wnt_core | 8 | 10 | 0 | 1.000 |  | 1.000 |
| adhesion_survival | pc_stress_survival | 8 | 6 | 0 | 1.000 |  | 1.000 |
| adhesion_survival | t_exhaustion | 8 | 8 | 0 | 1.000 |  | 1.000 |
| adhesion_survival | cytotoxicity | 8 | 17 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | cytotoxicity | 22 | 17 | 17 | 0.000 | CD8A, CD8B, CTSW, DEFA3, DEFA4, EOMES, GNLY, GZMA, GZMB, GZMH, GZMK, IFNG, KLRD1, KLRK1, NKG7, PRF1, TBX21 | 0.000 |
| cytotoxicity | wnt_camdr | 22 | 18 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | wnt_core | 22 | 10 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | adhesion_survival | 22 | 8 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | pc_stress_survival | 22 | 6 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | t_exhaustion | 22 | 8 | 0 | 1.000 |  | 1.000 |
| extra_0 | wnt_camdr | 120 | 18 | 10 | 1.000 | DKK1, FRZB, FZD3, IFNAR2, MIF, ROR2, SFRP1, SFRP2, WNT10A, WNT5A | 1.000 |
| extra_0 | wnt_core | 120 | 10 | 8 | 0.876 | DKK1, FRZB, FZD3, ROR2, SFRP1, SFRP2, WNT10A, WNT5A | 1.000 |
| extra_0 | adhesion_survival | 120 | 8 | 2 | 1.000 | IFNAR2, MIF | 1.000 |
| extra_0 | pc_stress_survival | 120 | 6 | 4 | 0.971 | DEPTOR, IL16, TLR10, TNFRSF13B | 1.000 |
| extra_0 | t_exhaustion | 120 | 8 | 5 | 0.990 | CD27, CTLA4, FOXP3, LAG3, TNFAIP3 | 1.000 |
| extra_0 | cytotoxicity | 120 | 17 | 12 | 0.989 | CD8A, CD8B, DEFA3, DEFA4, EOMES, GZMA, GZMH, GZMK, IFNG, KLRK1, NKG7, TBX21 | 1.000 |
| extra_1 | wnt_camdr | 129 | 18 | 15 | 0.983 | CDH2, DKK1, FRZB, FZD3, IGF1, IRF4, LRP6, MIF, NCAM1, ROR2, SFRP1, VEGFB, WNT10A, WNT5A, WNT5B | 1.000 |
| extra_1 | wnt_core | 129 | 10 | 9 | 0.870 | DKK1, FRZB, FZD3, LRP6, ROR2, SFRP1, WNT10A, WNT5A, WNT5B | 1.000 |
| extra_1 | adhesion_survival | 129 | 8 | 6 | 0.991 | CDH2, IGF1, IRF4, MIF, NCAM1, VEGFB | 1.000 |
| extra_1 | pc_stress_survival | 129 | 6 | 5 | 0.950 | DEPTOR, EIF2AK3, IL16, TLR10, TNFRSF13B | 1.000 |
| extra_1 | t_exhaustion | 129 | 8 | 6 | 0.991 | AHR, CD27, CTLA4, FOXP3, IL2RA, TNFAIP3 | 1.000 |
| extra_1 | cytotoxicity | 129 | 17 | 14 | 0.986 | CD8A, CD8B, DEFA3, DEFA4, EOMES, GNLY, GZMB, GZMH, GZMK, IFNG, KLRD1, KLRK1, NKG7, TBX21 | 1.000 |
| pc_stress_survival | pc_stress_survival | 6 | 6 | 6 | 0.000 | DDIT4, DEPTOR, EIF2AK3, IL16, TLR10, TNFRSF13B | 0.000 |
| pc_stress_survival | wnt_camdr | 6 | 18 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | wnt_core | 6 | 10 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | adhesion_survival | 6 | 8 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | t_exhaustion | 6 | 8 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | cytotoxicity | 6 | 17 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | t_exhaustion | 8 | 8 | 8 | 0.000 | AHR, CD27, CD55, CTLA4, FOXP3, IL2RA, LAG3, TNFAIP3 | 0.000 |
| t_exhaustion | wnt_camdr | 8 | 18 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | wnt_core | 8 | 10 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | adhesion_survival | 8 | 8 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | pc_stress_survival | 8 | 6 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | cytotoxicity | 8 | 17 | 0 | 1.000 |  | 1.000 |
| wnt_camdr | wnt_camdr | 20 | 18 | 17 | 0.000 | BCL2, CDH2, DKK1, FRZB, FZD3, IFNAR2, IGF1, IRF4, LRP6, MIF, NCAM1, SFRP1, SFRP2, VEGFB, WNT10A, WNT5A, WNT5B | 0.000 |
| wnt_camdr | wnt_core | 20 | 10 | 9 | 0.000 | DKK1, FRZB, FZD3, LRP6, SFRP1, SFRP2, WNT10A, WNT5A, WNT5B | 0.000 |
| wnt_camdr | adhesion_survival | 20 | 8 | 8 | 0.000 | BCL2, CDH2, IFNAR2, IGF1, IRF4, MIF, NCAM1, VEGFB | 0.000 |
| wnt_camdr | pc_stress_survival | 20 | 6 | 0 | 1.000 |  | 1.000 |

## 8d. Where the model looks (pathway attribution)

Cross-attention mass that each program token places on the centre patch, the neighbour blocks and the neighbour cells, averaged over held-out cells. Attention mass is descriptive, not causal.

| program | group | lineage | mass | n_cells | kind | target | input | dim | name | mean_abs_attribution | rank |
|---||---||---||---||---||---||---||---||---||---||---||---|
| wnt_camdr | centre_token | megakaryocyte | 0.019 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_centre | megakaryocyte | 0.068 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_inner_ring | megakaryocyte | 0.386 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_outer_ring | megakaryocyte | 0.418 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | neighbour_cells | megakaryocyte | 0.110 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | centre_token | megakaryocyte | 0.041 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_centre | megakaryocyte | 0.062 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_inner_ring | megakaryocyte | 0.367 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_outer_ring | megakaryocyte | 0.411 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | neighbour_cells | megakaryocyte | 0.119 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | centre_token | megakaryocyte | 0.058 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_centre | megakaryocyte | 0.055 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_inner_ring | megakaryocyte | 0.328 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_outer_ring | megakaryocyte | 0.333 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | neighbour_cells | megakaryocyte | 0.227 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | centre_token | megakaryocyte | 0.076 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_centre | megakaryocyte | 0.046 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_inner_ring | megakaryocyte | 0.310 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_outer_ring | megakaryocyte | 0.343 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | neighbour_cells | megakaryocyte | 0.226 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | centre_token | megakaryocyte | 0.019 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_centre | megakaryocyte | 0.045 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_inner_ring | megakaryocyte | 0.341 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_outer_ring | megakaryocyte | 0.391 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | neighbour_cells | megakaryocyte | 0.204 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | centre_token | megakaryocyte | 0.027 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_centre | megakaryocyte | 0.053 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_inner_ring | megakaryocyte | 0.377 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_outer_ring | megakaryocyte | 0.439 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | neighbour_cells | megakaryocyte | 0.103 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | centre_token | megakaryocyte | 0.022 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_centre | megakaryocyte | 0.059 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_inner_ring | megakaryocyte | 0.384 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_outer_ring | megakaryocyte | 0.431 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | neighbour_cells | megakaryocyte | 0.104 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | centre_token | megakaryocyte | 0.017 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_centre | megakaryocyte | 0.059 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_inner_ring | megakaryocyte | 0.388 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_outer_ring | megakaryocyte | 0.456 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | neighbour_cells | megakaryocyte | 0.081 | 259.000 | attention_mass | nan | nan | nan | nan | nan | nan |

## 9. Release gates

Each gate is read from one artefact of this run and nothing is recomputed; a gate whose artefact or column is absent is NOT EVALUATED, never passed. On the synthetic fixture the gates test the plumbing, not the biology.

| gate | status | value | threshold | source | meaning |
|---|---|---|---|---|---|
| above_chance | PASS | 0.716 | > 0.05 | `evaluation/cv_results.csv` | the full model beats the shuffled-label baseline on the main target by a margin |
| beyond_identity | PASS | 0.753 | > 0.05 | `evaluation/cv_results.csv` | morphology carries program information beyond the cell-type composition baseline |
| ci_excludes_zero | PASS | 0.371 | > 0 | `evaluation/cv_results.csv` | the lower 95 % patient-bootstrap bound of the full model's correlation is above zero |
| shuffled_at_chance | PASS | 0.002 | |r| < 0.05 | `evaluation/cv_results.csv` | the shuffled-label baseline sits at chance, so no leak feeds the model |
| calibrated | PASS | 0.914 | in [0.85, 0.95] | `evaluation/calibration.csv` | the 90 % Gaussian interval covers about 90 % of held-out cells (pooled; read the per-patient spread too) |
| backbone_stable | PASS | 20.000 | >= 3 genes | `interpretation/gene_backbone.csv` | the main program has at least three genes that survive stability selection over half-patient refits |
| inference_parity | PASS | 0.994 | > 0.95 and no inference error | `deployment/sanity_check.csv` | the morphology-only inference path reproduces the training-path predictions on a training slide |
| stage_association | PASS | 0.035 | PERMANOVA p < 0.05 or trend q < 0.1 | `evaluation/stage_association.csv` | the patient-level vector of predicted program burdens is associated with disease stage |
| external_validation | FAIL | program_specificity:pass; spatial_fidelity:pass; perturbation_robustness:fail; independent_cohort:skipped; protein_bridge:skipped | validators ran >= 1, every ran validator passed, failed = 0 | `external_validation/validation_summary.json` | every validator that could run on the available data met its own stated bar; skipped validators are listed with their reason |

8 of 9 gates passed, 1 failed, 0 not evaluated (`reporting/release_gates.csv`, `RESULTS.json`).

## 10. What this run does not establish

- No clinical endpoint (progression-free survival) was tested: no clinical table was supplied.
- The morphology channel is the native DAPI image (`stain_mode: native`); no H&E slide was used. Transfer to H&E requires the stain-harmonisation phase and an H&E cohort.
- Learned neighbourhood-scale targets average the cell-scale program usage over the cells kept for training (after subsampling), while the fixed-set neighbourhood targets were summed at ingest over every QC-passing cell; the two target families therefore measure neighbourhoods of different cell density (DESIGN_V3.md D28).
- The gene sets are investigator-defined gene-set abundances, not pathway activity. Caveats recorded with the gene annotations (`resources/gene_annotations.json`, docs/research/PATHWAY_INTERPRETATION.md):
  - wnt_camdr is an unsigned sum over ligands, receptors and secreted antagonists whose effects on pathway output have opposite signs, so a high score is NOT a Wnt pathway activity estimate.
  - SFRP1 and SFRP2 are silenced by promoter methylation in 35.5 and 52.6 percent of primary myeloma samples, so their detected counts most plausibly report stromal rather than malignant plasma cell biology.
  - Three wnt_camdr genes carry evidence pointing toward drug SENSITIVITY or better outcome rather than resistance: WNT10A (selinexor response), NCAM1 (CD56 loss associates with worse survival), BCL2 (t(11;14) venetoclax sensitivity).
  - Eight of the eighteen wnt_camdr genes are not Wnt pathway components (MIF, IFNAR2, IGF1, VEGFB, NCAM1, CDH2, IRF4, BCL2); they are the CAM-DR and niche-survival half of the program.
  - GSE299193 contains no drug exposure, response or survival data, so no score computed from it can be validated against resistance or progression-free survival.
  - Yip et al. 2025 does not report Wnt signalling, CAM-DR or progression-free survival; those findings are from the Ohlstrom et al. 2026 bioRxiv preprint (doi 10.64898/2026.03.23.713195), which is not peer reviewed.
- Single trephine per patient: patient-level and slide-level cross-validation coincide.

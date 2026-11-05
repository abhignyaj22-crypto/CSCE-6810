# Contributions established by this run

Run: `gse299193_v3`. Data: GSE299193, 21 patients, 552691 cells after QC and subsampling. Evaluation: patient-level 7-fold cross-validation.

## 1. A measured ceiling for every target

Split-half reliability (binomial thinning, Spearman-Brown corrected) gives the correlation an independent re-measurement of the same cells would reach. No image model can be expected to exceed its square root.

| target | median r_sb | attainable ceiling sqrt(r_sb) |
|---|---|---|
| anchor_adhesion_survival_cell_z | 0.257 | 0.507 |
| anchor_adhesion_survival_nb_z | 0.468 | 0.684 |
| anchor_cytotoxicity_cell_z | 0.417 | 0.646 |
| anchor_cytotoxicity_nb_z | 0.585 | 0.765 |
| anchor_pc_stress_survival_cell_z | 0.144 | 0.379 |
| anchor_pc_stress_survival_nb_z | 0.349 | 0.591 |
| anchor_t_exhaustion_cell_z | 0.099 | 0.315 |
| anchor_t_exhaustion_nb_z | 0.293 | 0.541 |
| anchor_wnt_camdr_cell_z | 0.270 | 0.520 |
| anchor_wnt_camdr_nb_z | 0.514 | 0.717 |
| anchor_wnt_core_cell_z | 0.146 | 0.382 |
| anchor_wnt_core_nb_z | 0.375 | 0.612 |
| prog_adhesion_survival_cell_z | 0.820 | 0.906 |
| prog_adhesion_survival_nb_z | 0.919 | 0.959 |
| prog_cytotoxicity_cell_z | 0.894 | 0.946 |
| prog_cytotoxicity_nb_z | 0.974 | 0.987 |
| prog_extra_0_cell_z | 0.949 | 0.974 |
| prog_extra_0_nb_z | 0.973 | 0.986 |
| prog_extra_1_cell_z | 0.897 | 0.947 |
| prog_extra_1_nb_z | 0.954 | 0.977 |
| prog_pc_stress_survival_cell_z | 0.524 | 0.724 |
| prog_pc_stress_survival_nb_z | 0.722 | 0.850 |
| prog_t_exhaustion_cell_z | 0.946 | 0.973 |
| prog_t_exhaustion_nb_z | 0.989 | 0.994 |
| prog_wnt_camdr_cell_z | 0.765 | 0.875 |
| prog_wnt_camdr_nb_z | 0.906 | 0.952 |
| prog_wnt_core_cell_z | 0.568 | 0.754 |
| prog_wnt_core_nb_z | 0.776 | 0.881 |

## 2. Held-out performance against baselines and ablations

| target | shuffled | composition_only | anchor_only_targets | align | no_smooth | no_comp | no_grid | no_neighbours | no_program_tokens | full | oracle | full 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_adhesion_survival_cell_z | nan | 0.236 | 0.122 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_adhesion_survival_nb_z | nan | 0.163 | 0.312 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_cytotoxicity_cell_z | nan | 0.299 | 0.077 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_cytotoxicity_nb_z | nan | 0.102 | 0.248 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_pc_stress_survival_cell_z | nan | 0.183 | 0.099 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_pc_stress_survival_nb_z | nan | 0.131 | 0.302 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_t_exhaustion_cell_z | nan | 0.054 | 0.049 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_t_exhaustion_nb_z | nan | 0.063 | 0.282 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_camdr_cell_z | nan | 0.239 | 0.138 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_camdr_nb_z | nan | 0.153 | 0.338 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_core_cell_z | nan | 0.114 | 0.082 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| anchor_wnt_core_nb_z | nan | 0.103 | 0.326 | nan | nan | nan | nan | nan | nan | nan | nan |  |
| prog_adhesion_survival_cell_z | -0.001 | 0.605 | nan | 0.214 | 0.237 | 0.211 | 0.182 | 0.186 | 0.160 | 0.213 | 0.627 | (0.182, 0.342) |
| prog_adhesion_survival_nb_z | -0.001 | 0.291 | nan | 0.277 | 0.277 | 0.283 | 0.308 | 0.228 | 0.209 | 0.304 | 0.597 | (0.227, 0.402) |
| prog_cytotoxicity_cell_z | -0.002 | 0.632 | nan | 0.312 | 0.301 | 0.292 | 0.267 | 0.241 | 0.217 | 0.287 | 0.695 | (0.27, 0.328) |
| prog_cytotoxicity_nb_z | -0.001 | 0.402 | nan | 0.417 | 0.421 | 0.403 | 0.381 | 0.249 | 0.252 | 0.385 | 0.734 | (0.345, 0.447) |
| prog_extra_0_cell_z | -0.002 | 0.299 | nan | 0.214 | 0.216 | 0.204 | 0.166 | 0.179 | 0.130 | 0.204 | 0.396 | (0.182, 0.219) |
| prog_extra_0_nb_z | -0.002 | 0.147 | nan | 0.271 | 0.281 | 0.261 | 0.250 | 0.195 | 0.180 | 0.263 | 0.418 | (0.249, 0.28) |
| prog_extra_1_cell_z | -0.000 | 0.521 | nan | 0.233 | 0.229 | 0.238 | 0.183 | 0.173 | 0.134 | 0.224 | 0.559 | (0.178, 0.262) |
| prog_extra_1_nb_z | 0.001 | 0.240 | nan | 0.300 | 0.284 | 0.302 | 0.274 | 0.184 | 0.175 | 0.286 | 0.544 | (0.215, 0.332) |
| prog_pc_stress_survival_cell_z | -0.001 | 0.381 | nan | 0.195 | 0.176 | 0.163 | 0.147 | 0.129 | 0.124 | 0.167 | 0.403 | (0.118, 0.21) |
| prog_pc_stress_survival_nb_z | -0.001 | 0.236 | nan | 0.273 | 0.273 | 0.282 | 0.296 | 0.165 | 0.145 | 0.274 | 0.432 | (0.154, 0.345) |
| prog_t_exhaustion_cell_z | -0.000 | 0.510 | nan | 0.538 | 0.550 | 0.529 | 0.471 | 0.512 | 0.476 | 0.515 | 0.679 | (0.479, 0.57) |
| prog_t_exhaustion_nb_z | -0.002 | 0.300 | nan | 0.610 | 0.622 | 0.599 | 0.601 | 0.539 | 0.554 | 0.600 | 0.709 | (0.535, 0.673) |
| prog_wnt_camdr_cell_z | 0.001 | 0.221 | nan | 0.316 | 0.313 | 0.311 | 0.298 | 0.254 | 0.240 | 0.307 | 0.458 | (0.244, 0.367) |
| prog_wnt_camdr_nb_z | 0.002 | 0.153 | nan | 0.494 | 0.486 | 0.486 | 0.498 | 0.358 | 0.351 | 0.478 | 0.591 | (0.321, 0.554) |
| prog_wnt_core_cell_z | 0.000 | 0.171 | nan | 0.214 | 0.214 | 0.235 | 0.193 | 0.151 | 0.134 | 0.212 | 0.328 | (0.171, 0.296) |
| prog_wnt_core_nb_z | -0.003 | 0.082 | nan | 0.342 | 0.371 | 0.367 | 0.325 | 0.217 | 0.203 | 0.347 | 0.390 | (0.269, 0.497) |

Reliability-adjusted (r divided by sqrt of the median noise ceiling):

| target | shuffled | composition_only | anchor_only_targets | align | no_smooth | no_comp | no_grid | no_neighbours | no_program_tokens | full | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_adhesion_survival_cell_z | nan | 0.486 | 0.254 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_adhesion_survival_nb_z | nan | 0.246 | 0.466 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_cytotoxicity_cell_z | nan | 0.495 | 0.141 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_cytotoxicity_nb_z | nan | 0.134 | 0.306 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_pc_stress_survival_cell_z | nan | 0.469 | 0.256 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_pc_stress_survival_nb_z | nan | 0.223 | 0.468 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_t_exhaustion_cell_z | nan | 0.172 | 0.141 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_t_exhaustion_nb_z | nan | 0.108 | 0.534 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_camdr_cell_z | nan | 0.466 | 0.320 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_camdr_nb_z | nan | 0.243 | 0.504 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_core_cell_z | nan | 0.318 | 0.306 | nan | nan | nan | nan | nan | nan | nan | nan |
| anchor_wnt_core_nb_z | nan | 0.200 | 0.520 | nan | nan | nan | nan | nan | nan | nan | nan |
| prog_adhesion_survival_cell_z | -0.001 | 0.657 | nan | 0.269 | 0.262 | 0.239 | 0.204 | 0.229 | 0.195 | 0.251 | 0.692 |
| prog_adhesion_survival_nb_z | -0.001 | 0.297 | nan | 0.318 | 0.310 | 0.304 | 0.322 | 0.247 | 0.228 | 0.315 | 0.641 |
| prog_cytotoxicity_cell_z | -0.002 | 0.673 | nan | 0.331 | 0.317 | 0.308 | 0.282 | 0.255 | 0.230 | 0.301 | 0.739 |
| prog_cytotoxicity_nb_z | -0.001 | 0.411 | nan | 0.421 | 0.425 | 0.414 | 0.391 | 0.251 | 0.254 | 0.395 | 0.740 |
| prog_extra_0_cell_z | -0.002 | 0.306 | nan | 0.225 | 0.222 | 0.208 | 0.170 | 0.182 | 0.132 | 0.208 | 0.407 |
| prog_extra_0_nb_z | -0.002 | 0.149 | nan | 0.277 | 0.287 | 0.265 | 0.252 | 0.199 | 0.183 | 0.265 | 0.428 |
| prog_extra_1_cell_z | -0.000 | 0.546 | nan | 0.241 | 0.242 | 0.247 | 0.189 | 0.188 | 0.139 | 0.243 | 0.592 |
| prog_extra_1_nb_z | 0.001 | 0.243 | nan | 0.307 | 0.288 | 0.309 | 0.280 | 0.187 | 0.177 | 0.294 | 0.561 |
| prog_pc_stress_survival_cell_z | -0.002 | 0.504 | nan | 0.269 | 0.240 | 0.214 | 0.205 | 0.195 | 0.171 | 0.235 | 0.523 |
| prog_pc_stress_survival_nb_z | -0.002 | 0.269 | nan | 0.330 | 0.317 | 0.328 | 0.353 | 0.196 | 0.182 | 0.319 | 0.505 |
| prog_t_exhaustion_cell_z | -0.000 | 0.523 | nan | 0.546 | 0.562 | 0.547 | 0.484 | 0.521 | 0.484 | 0.535 | 0.705 |
| prog_t_exhaustion_nb_z | -0.002 | 0.304 | nan | 0.616 | 0.625 | 0.605 | 0.607 | 0.541 | 0.557 | 0.605 | 0.726 |
| prog_wnt_camdr_cell_z | 0.001 | 0.306 | nan | 0.400 | 0.378 | 0.389 | 0.363 | 0.299 | 0.301 | 0.378 | 0.516 |
| prog_wnt_camdr_nb_z | 0.002 | 0.171 | nan | 0.508 | 0.515 | 0.518 | 0.512 | 0.393 | 0.385 | 0.522 | 0.650 |
| prog_wnt_core_cell_z | 0.000 | 0.242 | nan | 0.364 | 0.335 | 0.305 | 0.287 | 0.186 | 0.173 | 0.323 | 0.415 |
| prog_wnt_core_nb_z | -0.003 | 0.093 | nan | 0.447 | 0.438 | 0.437 | 0.383 | 0.246 | 0.257 | 0.425 | 0.499 |

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
- prog_adhesion_survival_cell_z: 0.206 +/- 0.008
- prog_adhesion_survival_nb_z: 0.285 +/- 0.017
- prog_cytotoxicity_cell_z: 0.302 +/- 0.014
- prog_cytotoxicity_nb_z: 0.391 +/- 0.012
- prog_extra_0_cell_z: 0.191 +/- 0.011
- prog_extra_0_nb_z: 0.272 +/- 0.010
- prog_extra_1_cell_z: 0.215 +/- 0.008
- prog_extra_1_nb_z: 0.274 +/- 0.012
- prog_pc_stress_survival_cell_z: 0.173 +/- 0.012
- prog_pc_stress_survival_nb_z: 0.261 +/- 0.012
- prog_t_exhaustion_cell_z: 0.530 +/- 0.015
- prog_t_exhaustion_nb_z: 0.603 +/- 0.006
- prog_wnt_camdr_cell_z: 0.309 +/- 0.006
- prog_wnt_camdr_nb_z: 0.483 +/- 0.012
- prog_wnt_core_cell_z: 0.221 +/- 0.008
- prog_wnt_core_nb_z: 0.344 +/- 0.005

## 3. Decision rules applied to `prog_wnt_camdr_nb_z`

- Main target: prog_wnt_camdr_nb_z. Full model per-patient median r = 0.478 (reliability-adjusted 0.522).
- ABOVE CHANCE: full (0.478) exceeds the shuffled-label baseline (0.002) by more than 0.05.
- MORPHOLOGY BEYOND IDENTITY: full (0.478) exceeds composition_only (0.153); morphology carries program information beyond cell type. Report as the main finding.
- IDENTITY BOTTLENECK: oracle (0.591) exceeds full (0.478); the next lever is the encoder (pathology-pretrained) or a larger patch, not the head.
- SPATIAL PRIOR USEFUL: full (0.478) exceeds no_neighbours (0.358).
- Across all targets: 16/16 exceed the shuffled baseline by 0.05.
- ABLATION no_program_tokens: full (0.478) vs without (0.351); the program-token cross-attention earns 0.127 on prog_wnt_camdr_nb_z.
- ABLATION no_grid: full (0.478) vs without (0.498); the patch-block tokens costs 0.020 on prog_wnt_camdr_nb_z.
- ABLATION no_smooth: full (0.478) vs without (0.486); the neighbourhood consistency term neither helps nor hurts 0.008 on prog_wnt_camdr_nb_z.
- ABLATION align: full (0.478) vs with the contrastive alignment loss switched on (0.494); switching the alignment loss on neither helps nor hurts prog_wnt_camdr_nb_z by 0.016.
- LEARNED VS FIXED TARGETS: learned prog_wnt_camdr_nb_z r = 0.478; fixed-set anchor_wnt_camdr_nb_z r = 0.338; learned programs are more predictable from morphology.

## 4. What this run supports

- Nuclear morphology carries information about the resistance-associated program beyond what cell identity alone explains (full model above both the shuffled-label bound and the composition-only baseline).
- The oracle gap shows cell identity is the bottleneck; the next lever is the encoder or patch size, not the head.
- Patient-level program burden versus disease stage: Spearman rho = 0.687, p = 0.000585 (n = 21).

## 5. Where in the neighbourhood the signal lives (spatially regularized boosting)

Boosting over decision stumps on a grid of patch blocks, with the spatial regulariser of Xiang et al. (2009); lambda = 0 is the AdaBoost control on the same stumps. Accuracy is the per-patient median held-out metric; coherence is Moran's I of the importance map over the block grid (about 0 for scattered selection, positive for clustered selection).

| loss | lambda | metric | value | coherence (Moran's I) | active blocks |
|---|---|---|---|---|---|
| exponential | 0.0 | auc_patient_median | 0.558 | -0.003 | 128 |
| exponential | 0.1 | auc_patient_median | 0.556 | 0.112 | 128 |
| exponential | 0.5 | auc_patient_median | 0.554 | 0.281 | 124 |
| squared | 0.0 | pearson_patient_median | 0.053 | 0.008 | 124 |
| squared | 0.1 | pearson_patient_median | 0.047 | 0.078 | 128 |
| squared | 0.5 | pearson_patient_median | 0.053 | 0.189 | 128 |

Importance by block-centre distance from the nucleus (centre < 5 um, inner 5 to 10 um, outer >= 10 um); share of total, and share per block in brackets:

| model | plane | centre | inner ring | outer ring |
|---|---|---|---|---|
| exponential_lam0.0 | mean | 0.058 (0.015) | 0.733 (0.026) | 0.208 (0.007) |
| exponential_lam0.0 | std | 0.064 (0.016) | 0.555 (0.020) | 0.381 (0.012) |
| exponential_lam0.1 | mean | 0.182 (0.046) | 0.591 (0.021) | 0.227 (0.007) |
| exponential_lam0.1 | std | 0.129 (0.032) | 0.580 (0.021) | 0.290 (0.009) |
| exponential_lam0.5 | mean | 0.263 (0.066) | 0.575 (0.021) | 0.162 (0.005) |
| exponential_lam0.5 | std | 0.204 (0.051) | 0.620 (0.022) | 0.177 (0.006) |
| squared_lam0.0 | mean | 0.267 (0.067) | 0.180 (0.006) | 0.553 (0.017) |
| squared_lam0.0 | std | 0.086 (0.022) | 0.468 (0.017) | 0.446 (0.014) |
| squared_lam0.1 | mean | 0.215 (0.054) | 0.246 (0.009) | 0.539 (0.017) |
| squared_lam0.1 | std | 0.090 (0.022) | 0.497 (0.018) | 0.414 (0.013) |
| squared_lam0.5 | mean | 0.167 (0.042) | 0.389 (0.014) | 0.444 (0.014) |
| squared_lam0.5 | std | 0.092 (0.023) | 0.528 (0.019) | 0.380 (0.012) |

## 6. Association of the predicted program vector with disease stage (patient level)

PERMANOVA (Anderson 2001; Euclidean distance between patient-level burden vectors, permutation p-value) asks whether the vector of predicted program burdens differs between stages; the Jonckheere-Terpstra trend test asks, per program, whether burden rises along the ordered stages, with Benjamini-Hochberg q-values across programs. Both are distribution-free and suited to the cohort size. Rows with `vectors = measured` repeat the tests on the measured programs, so the predicted and the measured association can be compared directly.

| vectors | test | program | statistic | p_value | q_value | r2_or_rho | n_patients | n_groups | groups | note |
|---||---||---||---||---||---||---||---||---||---||---|
| predicted | permanova | all | 4.035 | 0.002 | nan | 0.416 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | pseudo-F on 7 z-scored programs, Euclidean, 999 permutations |
| predicted | jt_trend | prog_wnt_core_nb_z | 128.000 | 0.001 | 0.004 | 0.762 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_adhesion_survival_nb_z | 128.000 | 0.001 | 0.004 | 0.756 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_pc_stress_survival_nb_z | 109.000 | 0.021 | 0.025 | 0.539 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_t_exhaustion_nb_z | 28.000 | 0.004 | 0.007 | -0.669 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | decreasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_cytotoxicity_nb_z | 37.000 | 0.019 | 0.025 | -0.526 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | decreasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_extra_0_nb_z | 79.000 | 0.792 | 0.792 | 0.066 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| predicted | jt_trend | prog_extra_1_nb_z | 117.000 | 0.003 | 0.007 | 0.665 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| measured | permanova | all | 2.161 | 0.032 | nan | 0.276 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | pseudo-F on 7 z-scored programs, Euclidean, 999 permutations |
| measured | jt_trend | prog_wnt_core_nb_z | 124.000 | 0.002 | 0.005 | 0.709 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_adhesion_survival_nb_z | 130.000 | 0.001 | 0.004 | 0.762 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_pc_stress_survival_nb_z | 137.000 | 0.001 | 0.004 | 0.831 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_t_exhaustion_nb_z | 39.000 | 0.020 | 0.028 | -0.532 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | decreasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_cytotoxicity_nb_z | 32.000 | 0.006 | 0.011 | -0.612 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | decreasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_extra_0_nb_z | 68.000 | 0.734 | 0.856 | -0.087 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | decreasing along the stage order; expected JT 74.0; 999 permutations |
| measured | jt_trend | prog_extra_1_nb_z | 77.000 | 0.887 | 0.887 | 0.039 | 21 | 4 | MGUS:2;MM:10;SMM:5;control:4 | increasing along the stage order; expected JT 74.0; 999 permutations |

With one trephine per patient the unit of analysis is the patient, so these tests have the cohort's patient count as their sample size and are reported as supporting evidence only.

## 7. Neighbourhood motifs enriched for the main program (byte-pair encoding over lineage sequences)

| motif | difference (with minus without) | permutation p | patients with motif | fraction of patients with positive difference |
|---|---|---|---|---|
| plasma|plasma|plasma|plasma|plasma|plasma|plasma | 2.165 | 0.005 | 10 | 1.000 |
| plasma|plasma|plasma|plasma | 1.574 | 0.005 | 19 | 1.000 |
| plasma|plasma|plasma | 1.366 | 0.005 | 20 | 1.000 |
| plasma|plasma | 1.091 | 0.005 | 21 | 1.000 |
| granulocyte|granulocyte | -0.502 | 0.005 | 21 | 0.000 |
| granulocyte|granulocyte|granulocyte|granulocyte | -0.486 | 0.005 | 20 | 0.000 |
| granulocyte|monocyte | -0.464 | 0.005 | 21 | 0.000 |
| granulocyte|NK | -0.457 | 0.005 | 21 | 0.000 |

Motif discovery is exploratory: byte-pair encoding is a tokenisation heuristic applied to ordered neighbour sequences, and enrichment is descriptive.

## 7a. External validation (modality-agnostic validators)

| validator | status | verdict | reason or headline |
|---|---|---|---|
| program_specificity | ran | pass | median_z = 12.518, median_r_target = 0.478 (bar: median_z > 2 on the main target and both split-gene halves correlate positively) |
| spatial_fidelity | ran | pass | median_moran_ratio = 0.968, cohens_d_pred = 1.081, cohens_d_measured = 1.368 (bar: 0.5 <= median Moran ratio <= 2 and the plasma-dense niche contrast has the measured sign, on the main target) |
| perturbation_robustness | ran | fail | min_spearman_main = 0.476, worst_perturbation_main = noise_0.02 (bar: Spearman between perturbed and clean predictions > 0.9 on the main target under every perturbation) |
| independent_cohort | skipped |  | no independent cohort manifest; set validation.validation_cohort_manifest to a JSON manifest (Xenium or Visium HD rows with local paths, or {"demo": {...}} for a synthetic cohort). The Visium HD trephine cohort of Ohlstrom et al. 2026 (preprint) is embargoed as of writing (docs/DATA.md) |
| protein_bridge | skipped |  | set codex_table (cell-by-marker CSV/parquet with x, y) and codex_dapi (DAPI image of the same section); no public dataset pairs CODEX with Xenium on the same myeloma trephine |

Program specificity per target (median over patients): the prediction's correlation with the real target against expression-matched random gene sets, and with each random half of the program's genes.

| target | median r_target | median null mean | median z | median r_half_a | median r_half_b |
|---|---|---|---|---|---|
| prog_adhesion_survival_nb_z | 0.304 | 0.036 | 5.899 | 0.360 | 0.365 |
| prog_cytotoxicity_nb_z | 0.385 | -0.261 | 11.549 | 0.228 | 0.315 |
| prog_extra_0_nb_z | 0.263 | 0.094 | 2.944 | 0.265 | 0.265 |
| prog_extra_1_nb_z | 0.286 | -0.117 | 8.817 | 0.269 | 0.206 |
| prog_pc_stress_survival_nb_z | 0.274 | 0.061 | 4.086 | 0.369 | 0.303 |
| prog_t_exhaustion_nb_z | 0.600 | -0.239 | 19.527 | 0.520 | 0.559 |
| prog_wnt_camdr_nb_z | 0.478 | -0.082 | 12.518 | 0.354 | 0.365 |
| prog_wnt_core_nb_z | 0.347 | 0.123 | 3.783 | 0.424 | 0.388 |

Spatial fidelity (median over patients): Moran's I of the predicted and the measured map.

| target | Moran's I predicted | Moran's I measured | ratio |
|---|---|---|---|
| prog_adhesion_survival_nb_z | 0.797 | 0.803 | 1.002 |
| prog_cytotoxicity_nb_z | 0.818 | 0.854 | 0.959 |
| prog_extra_0_nb_z | 0.805 | 0.793 | 1.000 |
| prog_extra_1_nb_z | 0.808 | 0.763 | 1.080 |
| prog_pc_stress_survival_nb_z | 0.789 | 0.746 | 1.056 |
| prog_t_exhaustion_nb_z | 0.886 | 0.896 | 1.005 |
| prog_wnt_camdr_nb_z | 0.826 | 0.888 | 0.968 |
| prog_wnt_core_nb_z | 0.826 | 0.744 | 1.043 |

Plasma-cell-dense niche contrast (Cohen's d, dense versus other neighbourhoods):

| target | d predicted | d measured | sign agreement |
|---|---|---|---|
| prog_wnt_camdr_nb_z | 1.081 | 1.368 | True |
| prog_wnt_core_nb_z | 1.107 | 1.245 | True |
| prog_adhesion_survival_nb_z | 1.526 | 1.608 | True |
| prog_pc_stress_survival_nb_z | 1.025 | 1.073 | True |
| prog_t_exhaustion_nb_z | -0.298 | -0.550 | True |
| prog_cytotoxicity_nb_z | -0.944 | -1.055 | True |
| prog_extra_0_nb_z | 0.165 | -0.203 | False |
| prog_extra_1_nb_z | 0.195 | -0.339 | False |

Perturbation robustness on `prog_wnt_camdr_nb_z`: Spearman between perturbed and clean predictions on the same nuclei.

| perturbation | Spearman vs clean | mean absolute shift (z units) |
|---|---|---|
| gamma_0.8 | 0.960 | 0.083 |
| gamma_1.25 | 0.945 | 0.090 |
| blur_1.0 | 0.616 | 0.149 |
| rot90 | 0.801 | 0.113 |
| noise_0.02 | 0.476 | 0.169 |
| scale_0.9 | 0.958 | 0.059 |

A skipped validator is a gap in the evidence, listed with its reason; the pass bars are the validators' own stated criteria (docs/research/VALIDATION_METHODS.md) and the `external_validation` release gate reads them.

## 8a. Calibration of the predictive uncertainty

Empirical coverage of the predictive intervals on held-out patients. A well calibrated model covers the target at the nominal level; coverage is reported per patient because cells within one patient are not exchangeable with cells of another.

| target | level | coverage_pooled | coverage_patient_median | coverage_patient_min | coverage_patient_max | mean_std | n_cells | n_patients | kind | q | ece |
|---||---||---||---||---||---||---||---||---||---||---||---|
| prog_wnt_camdr_cell_z | 0.500 | 0.598 | 0.620 | 0.236 | 0.906 | 0.582 | 552691.000 | 21.000 | gaussian | nan | 0.052 |
| prog_wnt_core_cell_z | 0.500 | 0.507 | 0.496 | 0.135 | 0.910 | 0.546 | 552691.000 | 21.000 | gaussian | nan | 0.049 |
| prog_adhesion_survival_cell_z | 0.500 | 0.585 | 0.580 | 0.220 | 0.970 | 0.905 | 552691.000 | 21.000 | gaussian | nan | 0.053 |
| prog_pc_stress_survival_cell_z | 0.500 | 0.544 | 0.539 | 0.134 | 0.928 | 0.663 | 552691.000 | 21.000 | gaussian | nan | 0.044 |
| prog_t_exhaustion_cell_z | 0.500 | 0.669 | 0.663 | 0.334 | 0.928 | 0.927 | 552691.000 | 21.000 | gaussian | nan | 0.089 |
| prog_cytotoxicity_cell_z | 0.500 | 0.506 | 0.472 | 0.126 | 0.882 | 0.935 | 552691.000 | 21.000 | gaussian | nan | 0.022 |
| prog_extra_0_cell_z | 0.500 | 0.659 | 0.633 | 0.501 | 0.927 | 1.129 | 552691.000 | 21.000 | gaussian | nan | 0.096 |
| prog_extra_1_cell_z | 0.500 | 0.546 | 0.519 | 0.383 | 0.897 | 1.030 | 552691.000 | 21.000 | gaussian | nan | 0.043 |
| prog_wnt_camdr_nb_z | 0.500 | 0.454 | 0.471 | 0.094 | 0.906 | 0.362 | 552691.000 | 21.000 | gaussian | nan | 0.042 |
| prog_wnt_core_nb_z | 0.500 | 0.416 | 0.453 | 0.024 | 0.841 | 0.281 | 552691.000 | 21.000 | gaussian | nan | 0.073 |
| prog_adhesion_survival_nb_z | 0.500 | 0.355 | 0.444 | 0.074 | 0.592 | 0.573 | 552691.000 | 21.000 | gaussian | nan | 0.100 |
| prog_pc_stress_survival_nb_z | 0.500 | 0.421 | 0.472 | 0.033 | 0.698 | 0.370 | 552691.000 | 21.000 | gaussian | nan | 0.063 |
| prog_t_exhaustion_nb_z | 0.500 | 0.503 | 0.503 | 0.279 | 0.805 | 0.779 | 552691.000 | 21.000 | gaussian | nan | 0.012 |
| prog_cytotoxicity_nb_z | 0.500 | 0.383 | 0.417 | 0.027 | 0.795 | 0.745 | 552691.000 | 21.000 | gaussian | nan | 0.093 |
| prog_extra_0_nb_z | 0.500 | 0.503 | 0.518 | 0.221 | 0.878 | 0.989 | 552691.000 | 21.000 | gaussian | nan | 0.004 |
| prog_extra_1_nb_z | 0.500 | 0.371 | 0.345 | 0.086 | 0.733 | 0.770 | 552691.000 | 21.000 | gaussian | nan | 0.098 |
| prog_wnt_camdr_cell_z | 0.900 | 0.892 | 0.950 | 0.477 | 0.990 | 0.582 | 552691.000 | 21.000 | gaussian | nan | 0.052 |
| prog_wnt_core_cell_z | 0.900 | 0.894 | 0.936 | 0.234 | 0.997 | 0.546 | 552691.000 | 21.000 | gaussian | nan | 0.049 |
| prog_adhesion_survival_cell_z | 0.900 | 0.920 | 0.968 | 0.561 | 0.998 | 0.905 | 552691.000 | 21.000 | gaussian | nan | 0.053 |
| prog_pc_stress_survival_cell_z | 0.900 | 0.888 | 0.944 | 0.268 | 0.993 | 0.663 | 552691.000 | 21.000 | gaussian | nan | 0.044 |
| prog_t_exhaustion_cell_z | 0.900 | 0.946 | 0.940 | 0.899 | 0.995 | 0.927 | 552691.000 | 21.000 | gaussian | nan | 0.089 |
| prog_cytotoxicity_cell_z | 0.900 | 0.898 | 0.933 | 0.618 | 0.998 | 0.935 | 552691.000 | 21.000 | gaussian | nan | 0.022 |
| prog_extra_0_cell_z | 0.900 | 0.945 | 0.960 | 0.869 | 0.992 | 1.129 | 552691.000 | 21.000 | gaussian | nan | 0.096 |
| prog_extra_1_cell_z | 0.900 | 0.924 | 0.951 | 0.706 | 0.997 | 1.030 | 552691.000 | 21.000 | gaussian | nan | 0.043 |
| prog_wnt_camdr_nb_z | 0.900 | 0.829 | 0.940 | 0.217 | 0.996 | 0.362 | 552691.000 | 21.000 | gaussian | nan | 0.042 |
| prog_wnt_core_nb_z | 0.900 | 0.792 | 0.882 | 0.054 | 0.989 | 0.281 | 552691.000 | 21.000 | gaussian | nan | 0.073 |
| prog_adhesion_survival_nb_z | 0.900 | 0.811 | 0.905 | 0.217 | 0.999 | 0.573 | 552691.000 | 21.000 | gaussian | nan | 0.100 |
| prog_pc_stress_survival_nb_z | 0.900 | 0.819 | 0.921 | 0.072 | 0.998 | 0.370 | 552691.000 | 21.000 | gaussian | nan | 0.063 |
| prog_t_exhaustion_nb_z | 0.900 | 0.909 | 0.906 | 0.836 | 0.998 | 0.779 | 552691.000 | 21.000 | gaussian | nan | 0.012 |
| prog_cytotoxicity_nb_z | 0.900 | 0.783 | 0.878 | 0.269 | 0.998 | 0.745 | 552691.000 | 21.000 | gaussian | nan | 0.093 |
| prog_extra_0_nb_z | 0.900 | 0.907 | 0.916 | 0.778 | 0.994 | 0.989 | 552691.000 | 21.000 | gaussian | nan | 0.004 |
| prog_extra_1_nb_z | 0.900 | 0.791 | 0.846 | 0.450 | 0.986 | 0.770 | 552691.000 | 21.000 | gaussian | nan | 0.098 |
| prog_adhesion_survival_cell_z | 0.900 | 0.827 | 0.909 | 0.293 | nan | nan | nan | nan | conformal | 0.893 | 0.053 |
| prog_adhesion_survival_nb_z | 0.900 | 0.769 | 0.794 | 0.212 | nan | nan | nan | nan | conformal | 1.556 | 0.100 |
| prog_cytotoxicity_cell_z | 0.900 | 0.842 | 0.820 | 0.534 | nan | nan | nan | nan | conformal | 1.344 | 0.022 |
| prog_cytotoxicity_nb_z | 0.900 | 0.767 | 0.815 | 0.206 | nan | nan | nan | nan | conformal | 1.397 | 0.093 |
| prog_extra_0_cell_z | 0.900 | 0.885 | 0.880 | 0.734 | nan | nan | nan | nan | conformal | 1.123 | 0.096 |
| prog_extra_0_nb_z | 0.900 | 0.874 | 0.876 | 0.649 | nan | nan | nan | nan | conformal | 1.439 | 0.004 |
| prog_extra_1_cell_z | 0.900 | 0.843 | 0.788 | 0.564 | nan | nan | nan | nan | conformal | 0.982 | 0.043 |
| prog_extra_1_nb_z | 0.900 | 0.720 | 0.815 | 0.381 | nan | nan | nan | nan | conformal | 1.493 | 0.098 |

## 8b. Gene backbone of each program

Genes whose learned weight survives stability selection over patient subsamples, with the literature evidence grade from resources/gene_annotations.json. A backbone gene whose evidence points toward sensitivity rather than resistance is shown with its grade so the reader can weigh it.

| program | gene | selection_fraction | weight_relative_to_max | evidence_grade | gene_class |
|---||---||---||---||---||---|
| wnt_camdr | HNRNPA1L2 | 0.900 | 0.444 |  |  |
| wnt_core | EIF3A | 1.000 | 0.151 |  |  |
| wnt_core | HSPA8 | 1.000 | 0.377 |  |  |
| wnt_core | HNRNPA1L2 | 1.000 | 0.128 |  |  |
| wnt_core | ATP5F1A | 1.000 | 0.358 |  |  |
| wnt_core | FBL | 0.950 | 0.641 |  |  |
| wnt_core | GANAB | 0.950 | 0.451 |  |  |
| wnt_core | HNRNPH1 | 0.950 | 0.653 |  |  |
| wnt_core | PABPC4 | 0.950 | 0.167 |  |  |
| wnt_core | OGT | 0.950 | 0.546 |  |  |
| wnt_core | SEPTIN6 | 0.950 | 0.821 |  |  |
| adhesion_survival | ATP5F1A | 1.000 | 0.278 |  |  |
| adhesion_survival | CD38 | 1.000 | 0.118 |  |  |
| adhesion_survival | GANAB | 1.000 | 0.325 |  |  |
| adhesion_survival | EWSR1 | 1.000 | 0.156 |  |  |
| adhesion_survival | CYBA | 1.000 | 0.347 |  |  |
| adhesion_survival | XBP1 | 1.000 | 0.578 |  |  |
| adhesion_survival | PTK2B | 1.000 | 0.231 |  |  |
| adhesion_survival | SND1 | 1.000 | 0.454 |  |  |
| adhesion_survival | MAN1A1 | 1.000 | 0.171 |  |  |
| adhesion_survival | HNRNPH1 | 1.000 | 0.226 |  |  |
| pc_stress_survival | ADAR | 0.950 | 0.209 |  |  |
| pc_stress_survival | ADD1 | 0.950 | 0.109 |  |  |
| pc_stress_survival | CFLAR | 0.950 | 0.519 |  |  |
| pc_stress_survival | CKAP4 | 0.950 | 0.074 |  |  |
| pc_stress_survival | DDX6 | 0.950 | 0.127 |  |  |
| pc_stress_survival | DNM2 | 0.950 | 0.038 |  |  |
| pc_stress_survival | EEF1G | 0.950 | 1.000 |  |  |
| pc_stress_survival | CYLD | 0.950 | 0.056 |  |  |
| pc_stress_survival | CYBA | 0.950 | 0.208 |  |  |
| pc_stress_survival | CXCR4 | 0.950 | 0.114 |  |  |
| t_exhaustion | SOX2-OT | 0.950 | 0.362 |  |  |
| t_exhaustion | EEF1G | 0.950 | 0.177 |  |  |
| t_exhaustion | CAT | 0.900 | 0.395 |  |  |
| t_exhaustion | SLC4A1 | 0.900 | 1.000 |  |  |
| t_exhaustion | MKI67 | 0.900 | 0.178 |  |  |
| t_exhaustion | UBTF | 0.900 | 0.122 |  |  |
| t_exhaustion | DEFA1 | 0.850 | 0.008 |  |  |
| t_exhaustion | ABCA7 | 0.800 | 0.077 |  |  |
| t_exhaustion | AP2A1 | 0.800 | 0.134 |  |  |
| t_exhaustion | ANK1 | 0.800 | 0.327 |  |  |
| cytotoxicity | DEFA1 | 0.950 | 1.000 |  |  |
| cytotoxicity | DDX39B | 0.900 | 0.182 |  |  |
| cytotoxicity | EEF1G | 0.900 | 0.368 |  |  |
| cytotoxicity | CAT | 0.850 | 0.161 |  |  |
| cytotoxicity | ATF4 | 0.850 | 0.118 |  |  |
| cytotoxicity | LENG8 | 0.850 | 0.144 |  |  |
| cytotoxicity | SOST | 0.850 | 0.461 |  |  |
| cytotoxicity | SNHG15 | 0.850 | 0.286 |  |  |
| cytotoxicity | NSUN5 | 0.800 | 0.121 |  |  |
| cytotoxicity | LARP4B | 0.800 | 0.061 |  |  |
| extra_0 | SAFB | 1.000 | 0.188 |  |  |
| extra_0 | XRCC5 | 1.000 | 0.353 |  |  |
| extra_0 | RREB1 | 1.000 | 0.314 |  |  |
| extra_0 | CPSF6 | 1.000 | 0.312 |  |  |
| extra_0 | CAT | 1.000 | 0.320 |  |  |
| extra_0 | RBM15 | 1.000 | 0.297 |  |  |
| extra_0 | SF3B1 | 1.000 | 0.132 |  |  |
| extra_0 | ARHGEF2 | 1.000 | 0.168 |  |  |
| extra_0 | DNM2 | 1.000 | 0.224 |  |  |
| extra_0 | NONO | 1.000 | 0.392 |  |  |
| extra_1 | EEF1G | 1.000 | 0.530 |  |  |
| extra_1 | PRKAR1A | 0.950 | 0.331 |  |  |
| extra_1 | NOTCH2NLA | 0.900 | 0.175 |  |  |
| extra_1 | SNHG15 | 0.900 | 0.252 |  |  |
| extra_1 | LENG8 | 0.900 | 0.248 |  |  |
| extra_1 | DDX39B | 0.900 | 0.363 |  |  |
| extra_1 | HNRNPA1L2 | 0.900 | 0.105 |  |  |
| extra_1 | ATF4 | 0.850 | 0.172 |  |  |
| extra_1 | NAIP | 0.850 | 0.257 |  |  |
| extra_1 | DNM2 | 0.850 | 0.217 |  |  |

## 8c. Pathway annotation of the learned programs

Hypergeometric enrichment of each learned program's backbone against the curated gene sets, with the measured panel as background.

| program | gene_set | backbone_size | set_size_in_panel | overlap | p_value | overlapping_genes | q_value |
|---||---||---||---||---||---||---||---|
| adhesion_survival | pc_stress_survival | 106 | 5 | 2 | 0.004 | EIF2AK3, IL16 | 0.199 |
| adhesion_survival | wnt_camdr | 106 | 15 | 1 | 0.272 | IRF4 | 1.000 |
| adhesion_survival | wnt_core | 106 | 8 | 0 | 1.000 |  | 1.000 |
| adhesion_survival | adhesion_survival | 106 | 7 | 1 | 0.137 | IRF4 | 1.000 |
| adhesion_survival | t_exhaustion | 106 | 7 | 0 | 1.000 |  | 1.000 |
| adhesion_survival | cytotoxicity | 106 | 12 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | wnt_camdr | 12 | 15 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | wnt_core | 12 | 8 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | adhesion_survival | 12 | 7 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | pc_stress_survival | 12 | 5 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | t_exhaustion | 12 | 7 | 0 | 1.000 |  | 1.000 |
| cytotoxicity | cytotoxicity | 12 | 12 | 0 | 1.000 |  | 1.000 |
| extra_0 | wnt_camdr | 671 | 15 | 0 | 1.000 |  | 1.000 |
| extra_0 | wnt_core | 671 | 8 | 0 | 1.000 |  | 1.000 |
| extra_0 | adhesion_survival | 671 | 7 | 0 | 1.000 |  | 1.000 |
| extra_0 | pc_stress_survival | 671 | 5 | 0 | 1.000 |  | 1.000 |
| extra_0 | t_exhaustion | 671 | 7 | 0 | 1.000 |  | 1.000 |
| extra_0 | cytotoxicity | 671 | 12 | 0 | 1.000 |  | 1.000 |
| extra_1 | wnt_camdr | 34 | 15 | 0 | 1.000 |  | 1.000 |
| extra_1 | wnt_core | 34 | 8 | 0 | 1.000 |  | 1.000 |
| extra_1 | adhesion_survival | 34 | 7 | 0 | 1.000 |  | 1.000 |
| extra_1 | pc_stress_survival | 34 | 5 | 0 | 1.000 |  | 1.000 |
| extra_1 | t_exhaustion | 34 | 7 | 0 | 1.000 |  | 1.000 |
| extra_1 | cytotoxicity | 34 | 12 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | pc_stress_survival | 91 | 5 | 1 | 0.087 | IL16 | 0.831 |
| pc_stress_survival | wnt_camdr | 91 | 15 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | wnt_core | 91 | 8 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | adhesion_survival | 91 | 7 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | t_exhaustion | 91 | 7 | 0 | 1.000 |  | 1.000 |
| pc_stress_survival | cytotoxicity | 91 | 12 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | wnt_camdr | 88 | 15 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | wnt_core | 88 | 8 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | adhesion_survival | 88 | 7 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | pc_stress_survival | 88 | 5 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | t_exhaustion | 88 | 7 | 0 | 1.000 |  | 1.000 |
| t_exhaustion | cytotoxicity | 88 | 12 | 0 | 1.000 |  | 1.000 |
| wnt_camdr | wnt_camdr | 1 | 15 | 0 | 1.000 |  | 1.000 |
| wnt_camdr | wnt_core | 1 | 8 | 0 | 1.000 |  | 1.000 |
| wnt_camdr | adhesion_survival | 1 | 7 | 0 | 1.000 |  | 1.000 |
| wnt_camdr | pc_stress_survival | 1 | 5 | 0 | 1.000 |  | 1.000 |

## 8d. Where the model looks (pathway attribution)

Cross-attention mass that each program token places on the centre patch, the neighbour blocks and the neighbour cells, averaged over held-out cells. Attention mass is descriptive, not causal.

| program | group | lineage | mass | n_cells | kind | target | input | dim | name | mean_abs_attribution | rank |
|---||---||---||---||---||---||---||---||---||---||---||---|
| wnt_camdr | centre_token | all | 0.025 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_centre | all | 0.044 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_inner_ring | all | 0.286 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | grid_outer_ring | all | 0.469 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_camdr | neighbour_cells | all | 0.175 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | centre_token | all | 0.021 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_centre | all | 0.043 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_inner_ring | all | 0.293 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | grid_outer_ring | all | 0.489 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| wnt_core | neighbour_cells | all | 0.154 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | centre_token | all | 0.037 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_centre | all | 0.038 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_inner_ring | all | 0.230 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | grid_outer_ring | all | 0.371 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| adhesion_survival | neighbour_cells | all | 0.324 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | centre_token | all | 0.030 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_centre | all | 0.045 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_inner_ring | all | 0.278 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | grid_outer_ring | all | 0.436 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| pc_stress_survival | neighbour_cells | all | 0.210 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | centre_token | all | 0.031 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_centre | all | 0.070 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_inner_ring | all | 0.324 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | grid_outer_ring | all | 0.366 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| t_exhaustion | neighbour_cells | all | 0.210 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | centre_token | all | 0.030 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_centre | all | 0.038 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_inner_ring | all | 0.261 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | grid_outer_ring | all | 0.463 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| cytotoxicity | neighbour_cells | all | 0.209 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | centre_token | all | 0.030 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_centre | all | 0.050 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_inner_ring | all | 0.294 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | grid_outer_ring | all | 0.431 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_0 | neighbour_cells | all | 0.196 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | centre_token | all | 0.035 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_centre | all | 0.035 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_inner_ring | all | 0.231 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | grid_outer_ring | all | 0.387 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |
| extra_1 | neighbour_cells | all | 0.311 | 20000.000 | attention_mass | nan | nan | nan | nan | nan | nan |

## 9. Release gates

Each gate is read from one artefact of this run and nothing is recomputed; a gate whose artefact or column is absent is NOT EVALUATED, never passed. On the synthetic fixture the gates test the plumbing, not the biology.

| gate | status | value | threshold | source | meaning |
|---|---|---|---|---|---|
| above_chance | PASS | 0.476 | > 0.05 | `evaluation/cv_results.csv` | the full model beats the shuffled-label baseline on the main target by a margin |
| beyond_identity | PASS | 0.325 | > 0.05 | `evaluation/cv_results.csv` | morphology carries program information beyond the cell-type composition baseline |
| ci_excludes_zero | PASS | 0.321 | > 0 | `evaluation/cv_results.csv` | the lower 95 % patient-bootstrap bound of the full model's correlation is above zero |
| shuffled_at_chance | PASS | 0.002 | |r| < 0.05 | `evaluation/cv_results.csv` | the shuffled-label baseline sits at chance, so no leak feeds the model |
| calibrated | FAIL | 0.829 | in [0.85, 0.95] | `evaluation/calibration.csv` | the 90 % Gaussian interval covers about 90 % of held-out cells (pooled; read the per-patient spread too) |
| backbone_stable | FAIL | 1.000 | >= 3 genes | `interpretation/gene_backbone.csv` | the main program has at least three genes that survive stability selection over half-patient refits |
| inference_parity | FAIL | 0.815 | > 0.95 and no inference error | `deployment/sanity_check.csv` | the morphology-only inference path reproduces the training-path predictions on a training slide |
| stage_association | PASS | 0.002 | PERMANOVA p < 0.05 or trend q < 0.1 | `evaluation/stage_association.csv` | the patient-level vector of predicted program burdens is associated with disease stage |
| external_validation | FAIL | program_specificity:pass; spatial_fidelity:pass; perturbation_robustness:fail; independent_cohort:skipped; protein_bridge:skipped | validators ran >= 1, every ran validator passed, failed = 0 | `external_validation/validation_summary.json` | every validator that could run on the available data met its own stated bar; skipped validators are listed with their reason |

5 of 9 gates passed, 4 failed, 0 not evaluated (`reporting/release_gates.csv`, `RESULTS.json`).

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

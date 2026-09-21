# Results and populations

All paths below are relative to `experiments/`. Saved failures are evidence, not records to discard.

| Evidence | Files | Scope and interpretation |
|---|---|---|
| Core proportionality and tortuosity | `results_E_maze_s01234.json` | 15 grid-by-training-seed units. Aggregate and per-scene coefficients are distinct estimands |
| Eligible ruler CVs, policy and figures | `audit_full_population.json`, `policy_discrete.py` | 3,810 rows: 762 scene identities with five model seeds. Clamped scenes excluded, missing knees retained for coverage |
| Uncertainty | `bootstrap_ci.json`, `bootstrap_ci.py` | 2,000 scene-cluster replicates, all five seed rows carried together; constants refitted within replicate |
| Weighted-target transfer | `results_L_clean_G32.json`, `branch_L_clean.py`, `branch_L_transfer.py` | Three seeds. Each heterogeneity level's conditional ratios use its own attained rows and population SD. Common-subset ordering is checked separately |
| Weighted 40k sensitivity | `diag_L_het09_s0_40k.json`, `diag_L_het09_40k.py` | One seed, post-hoc duration diagnostic, not additional independent evidence |
| Original G48 addendum | `results_E_ext_G48.json`, `diag_G48_s4.json` | Original seeds 3/4, including the aggregate failure |
| Five-seed G48 extension | `camera_ready/g48_v1/`, `camera_ready/reference_g48_v1/` | New seeds 0/1/2 plus replay of archived 3/4. Three aggregate attainments out of five |
| Wider G32 control | `camera_ready/width_v1/` | Three trainings, C=24, hidden=192. Calibration uses even scene indices and evaluation uses odd indices |
| Paired G48 interventions | `camera_ready/round2_paired_v1/`, `camera_ready/round2_pool_provenance_v1.json` | Baseline, width and duration for seeds 0/1/2. Original, diagnostic and held-out confirmation pools. Common original-pool variance for comparison |
| G48 pre-intervention diagnosis | `camera_ready/round2_diagnosis_v1/` | Diagnostic pool and horizon/quality checks, not independent new training seeds |
| Sobel perception | `results_M_sobel.json` | Three seeds per grid; G32 coefficient statistics retain two runs, with the non-finite evaluation floor disclosed |
| VGLC maps | `eval_vglc.json`, `eval_vglc.py` | Human-authored external layouts, not external model training. No raw map files bundled |
| Tail definition sensitivity | `probe_p100.json`, `probe_p100.py` | Retained seeds 3/4. Farthest-cell versus p95 values are differences of medians, not medians of paired delays |
| Screened Poisson | `results_N_screened.json`, `results_N_screened_s12.json`, `branch_N_gates.json` | Single gamma negative probe. One G32 seed reverses ruler ordering; held-out policy savings are negative |
| Full-path timing | `camera_ready/latency_full_population/results.json` | 762 scenes, one retained model seed, six timing repeats per scene/arm. Static and CPU-origin changing-input paths. Means, not medians; setup reported separately |
| Derived revision summary | `camera_ready/analysis_full_population.json` | Produced by `summarize_full_population.py`. The archived orchestrator uses local status files; independent saved-row checks do not need those files |

Current figure sources are in `paper/camera_ready_636/`. The generator reads the same saved core audit and bootstrap files. It does not infer data from the printed plots.

## Definitions to preserve

The knee is the first step beginning a ten-step threshold crossing after a five-point median filter, at theta=0.05. Mean and p95-tail error have separate knees. An attained knee does not promise permanent stability. Policy coverage counts whether a budget reaches the recorded knee, whereas actual-output diagnostics test the error at the stopping step.

The core eligible population removes scenes with Dgeo>2G. The revision training-control analysis additionally requires Dgeo>2. The full-path timing population intentionally matches the core audit and retains Dgeo<=2. Do not merge these denominators silently. Paired intervention NPZ curves use native pool variances; their comparable JSON summaries rescale to the original pool variance. `verify_curves.py` applies that conversion before checking them.

## Reproducibility depth

`verify_release.py` checks hashes and selected table/claim numbers, including conditional ratios, policy point estimates and latency means. Saved bootstrap endpoints are checked for consistency, but that command does not generate the 2,000 replicates anew. `verify_curves.py` independently reconstructs every archived revision per-scene mean/tail knee from raw curves. Neither route constitutes a fresh training replication or hardware-independent timing validation.

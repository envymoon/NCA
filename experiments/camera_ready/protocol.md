# Camera-ready experimental protocol

Specified 2026-09-12 after review, before these runs. This is not part of the original preregistration. Public excerpt: local file-management instructions and private review summaries are omitted. See SOURCE_PROVENANCE.json for the original protocol digest.

## Experiment CR-width: a controlled capacity increase

Train learned-perception NCA at G=32, C=24, hidden=192, seeds 0/1/2. Baseline is
C=12, hidden=96 with the same archived seeds. This is a width control, not a new
architectural family. Keep the receptive field, residual update, optimizer, data pools,
20,000 iterations, batch 12, segment 24, learning rate 0.001, clip 1, cap 152,
theta 0.05, five-point median, ten-step persistence, and p95 tail error fixed.
Restore the reference model's post-initialization random-generator states before
training the wider model so batch indices and sampled horizons match the original
small-model schedule for the same seed.

Primary reports: each seed's convergence ceiling, aggregate attainment, per-scene
tail T*/Dgeo and CV on all tail-converged eligible scenes, and paired mean/tail
population sensitivity. Compare geo/free/euc on exactly the same rows. Never remove
an entire seed's scene results because its aggregate mean curve fails.
Fit flat and geo policies on even scene indices and evaluate both on odd indices.
Report coverage differences with cost differences. Positive savings at lower coverage
are not a matched-quality win. Store actual error at the chosen budget as a separate
diagnostic because an earlier persistent crossing need not survive late drift.

## Experiment CR-G48: complete the existing seed set

Add the missing G=48 seeds 0/1/2 under the frozen protocol above with C=12, hidden=96.
The original seeds 3/4 remain part of the five-seed outcome, including seed 4's failure.
Re-evaluate the existing G48 checkpoints with the new full-row exporter before launching
training and reconcile their reported aggregate floors and knees.
Keep cap=152. Report the fraction of scenes with inadequate horizon headroom, clamp
exclusions, non-finite trajectories, and convergence failures. This extends finite-range
evidence and replication; it cannot establish asymptotic scaling or zero-shot size transfer.
Do not add G64 or tune the cap in response to disappointing results in this round.

## Experiment CR-latency: measure the omitted executable costs

Use the existing fused kernel and frozen pooled f=.95 policy (Tflat=39, c=1.29),
G=16/24/32 and archived seed 3. No retraining or calibration. Check native BFS against
the same 256 scenes per grid and verify kernel output with changed inputs and varying
budgets before accepting any measurements.

Measure two paired paths, alternating/randomizing arm order with a fixed timing seed:

* Static geometry: resident input, cached per-scene budget, graph lookup, replay, completion wait.
* CPU-origin changing geometry: CPU input packing and pinned staging, input transfer,
  BFS and integer budget selection for the geo arm, graph lookup, replay, completion wait.
  The flat arm pays the same input preparation/transfer and uses its fixed budget.

For each eligible scene, retain individual paired timings and report scene-weighted
means, medians, quantiles, and a paired scene-bootstrap interval for the saving.
Warm up both paths. Report component diagnostics separately; the primary total is
timed directly, not obtained by adding component medians.

Measure graph construction and first-use costs, graph-bank size and memory where
observable. Separate shared kernel compilation from arm-specific capture/setup.
Show amortization for 1/10/100/1000 frames and a first-use graph miss. A bank of budgets
seen in the measurement pool is explicitly a workload-specific cache, not evidence
that unseen budgets are free. Engine scheduling, GPU-origin mask readback, and actual
renderer integration remain outside this experiment. No claim that shared overhead
cancels from the saving percentage.

## Numerical and provenance checks

Before long runs, verify the submission PDF and archived manifests, CUDA availability,
pool reconstruction, reference-model forward/gradient behavior, and the strict persistence
definition. Run a short training/memory calibration on synthetic pilot inputs. The pilot
does not contribute paper results. Freeze this protocol before any new trained outcome.
Save per-unit progress and an exclusive run lock. Never overwrite an existing result.
Non-finite loss or gradients terminate that seed and are recorded, with the last valid
checkpoint retained for diagnosis. No automatic learning-rate changes or replacement seeds.
An infrastructure error stops the queue. Train sequentially on the laptop GPU and keep
timing runs separate from training.

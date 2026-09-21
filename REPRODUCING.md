# Reproducing the artifact

## CPU checks

Run from the repository root:

```sh
python verify_release.py
python -m pip install numpy
python verify_curves.py
```

These commands read archived data without overwriting it. Manifest validation intentionally fails if an archived file changes. A clean checkout of the pinned tag restores a separate verification copy.

## GPU environment

Original revision records identify PyTorch 2.12.1+cu130, CUDA 13.0, Triton 3.7.1 and an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM, under Linux/WSL2. NumPy is required. Timing also needs a C compiler (`gcc`). Use the official PyTorch installation appropriate to your CUDA environment. Different versions or hardware may change random streams, floating-point behavior and timings; version substitution is not bit-exact reproduction.

The pool generator uses CUDA random generators, not CPU-generated random pools. Training uses Adam lr=.001, batch 12, 20,000 updates, gradient clipping at 1, checkpoint segments of 24 steps, and an anytime horizon sampled inclusively from 4 to 152 for revision controls. Source and result files record the remaining settings. TF32 is enabled for control training and disabled for the full-path kernel parity reference.

Use a fresh output directory for every command. Do not select an archived result directory as output. Paths supplied to `--out-dir` below are resolved relative to `experiments/camera_ready/` unless absolute.

```sh
python experiments/camera_ready/train_controls.py --mode smoke --seeds 0 --out-dir ../../runs/smoke
python experiments/camera_ready/train_controls.py --mode g48 --seeds 0 1 2 --out-dir ../../runs/g48
python experiments/camera_ready/train_controls.py --mode width --seeds 0 1 2 --out-dir ../../runs/width
python experiments/camera_ready/train_controls.py --mode reference --seeds 3 4 --out-dir ../../runs/reference
```

`reference` loads the original G48 seeds 3/4 rather than retraining them. The first two full training commands can take hours. No new training was run merely to prepare this release.

### Paired interventions

This publication wrapper uses the original training and evaluation helpers with portable input paths and refuses an existing output directory:

```sh
python reproduce_paired.py --mode replay --out-dir runs/paired-replay
python reproduce_paired.py --mode train --out-dir runs/paired-retrain
```

`replay` evaluates nine archived models on three pools without training. `train` keeps the archived baseline models, retrains the three wider models, and resumes the three baseline optimizer/RNG checkpoints from 20k to 40k updates. It does not recreate the baseline from scratch. To test a freshly trained baseline, use `--baseline-dir` pointing to the fresh G48 output directory. The wrapper was syntax-checked for release; no new GPU replay/training was executed during release preparation.

### Full-path timing

```sh
python experiments/camera_ready/latency_full_population.py --smoke --out-dir ../../runs/timing-smoke
python experiments/camera_ready/latency_full_population.py --out-dir ../../runs/timing
```

Timing includes graph selection, replay and synchronization. The changing-input arm additionally includes CPU packing, pinned staging and H2D transfer, plus mask conversion, source extraction and BFS for the geodesic arm. Graph-bank setup is separate. Cold process startup, renderer scheduling and GPU-origin readback are not included. Hardware load affects latency; these are not portable speed guarantees.

## Original analyses and figures

Original reproduction commands are preserved in `original_submission/RELEASE_README.md`. Several historical scripts write fixed filenames. Run them only in a disposable copy of the release, not over this archive. The actual computation inputs are under `experiments/`. The original policy script and `bootstrap_ci.py` expect that directory as their working directory.

To regenerate the current vector figures in a disposable copy, install Matplotlib and run `paper/camera_ready_636/make_figures_camera_ready.py`. This replaces that copy's figure files. To run VGLC evaluation, obtain the external data from https://github.com/TheVGLC/TheVGLC at `data/TheVGLC/` and run `eval_vglc.py` from `experiments/`, as documented in its header. The script has no data-path command-line option. No third-party map data are shipped here.

Archived orchestration programs such as `round2_paired.py`, `audit_completed.py`, `audit_round2.py` and `summarize_full_population.py` retain historical local-status checks. Use the portable wrappers above for the documented routes. Their presence in the release does not imply that every original launch script is independently portable.

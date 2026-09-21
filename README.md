# Geodesic Step Budgeting for Neural Cellular Automata

Ian Dang, University of California, Davis. ICTAI 2026, paper 636.

Formal repository: https://github.com/envymoon/NCA

Release: **ictai-2026-camera-ready**. Use the tag rather than a changing `main` checkout.

This artifact studies a scene-dependent iteration budget for distance-like NCA tasks. It is not an alternative fast distance solver: BFS already solves the unweighted reference task. The tested benefit is fewer learned-update passes at matched **offline knee-attainment coverage**. That event does not certify output quality at the stopping step, especially when trajectories drift.

## Start here

Python 3.10 or later is sufficient for the saved-number checks. No GPU or third-party Python packages are needed for the first command.

```sh
python verify_release.py
```

It checks the complete release manifest and recomputes selected manuscript values from saved per-scene records. It does not retrain models or independently recreate every historic analysis. For a separate CPU oracle over the saved revision curves:

```sh
python -m pip install numpy
python verify_curves.py
```

See [REPRODUCING.md](REPRODUCING.md) for GPU evaluation, training and timing. [RESULTS_INDEX.md](RESULTS_INDEX.md) maps the paper's evidence to files and populations.

## Versions

- `experiments/` contains the code and evidence used by the camera-ready paper, including original core results and additional controls in `experiments/camera_ready/`.
- `original_submission/` preserves the complete 65-file local submission artifact and its original manifest, byte for byte. It includes superseded timing and earlier explanatory wording.
- `legacy_github/` preserves the preceding GitHub snapshot, commit `1b4ebbe944aeb46135a22ee8d663ac94504edfca`. Its comments were stripped for the anonymous release. This is a separate source snapshot, not another experiment.
- `SOURCE_PROVENANCE.json` records original paths and hashes. [CODE_PROVENANCE.md](CODE_PROVENANCE.md) explains the two historical source-hash differences. Original manifests are not rewritten.

The current latency claims use **full-path** measurements in `experiments/camera_ready/latency_full_population/results.json`. The earlier Branch O timing omitted parts of the changing-input path and is retained only as history.

## Evidence limits

The core comparison uses grids 16/24/32 and five trainings per grid. Only three of five G48 baseline trainings attain the aggregate threshold. Ratio statistics condition on attainment; budget coverage retains non-attained scenes as failures. A wider model and fixed Sobel perception test limited changes of capacity and perception. They do not establish architecture-independent calibration. Weighted targets, game maps and a screened-Poisson negative result delineate transfer limits. These experiments do not establish an asymptotic scaling law or a competitive real-world application.

## License and external materials

The author's code and accompanying artifact documentation are MIT licensed; see [LICENSE](LICENSE). This code license is not a license for the conference manuscript or for third-party data. Included checkpoints and generated numerical records are supplied as research artifacts. External assets and software retain their own terms. VGLC map files are not bundled; obtain them from https://github.com/TheVGLC/TheVGLC and follow the relevant data terms and attribution requirements. Saved measurements derived from those maps are included.

# Anonymous ICTAI artifact

This archive contains only artifacts used by the submitted paper. It deliberately omits
development logs, superseded experiments, and the withdrawn learned-halting pilot.

## Reproduce the CPU-only paper analyses

From this directory:

```text
python full_population_audit.py results_E_maze_s01234.json \
  --device cpu --reuse-rows audit_full_population.json \
  --out audit_full_population.json
python bootstrap_ci.py
python eval_vglc.py --reanalyze-existing
python make_figures.py
```

The first command reuses the bit-exact pool reconstruction already stored in the audit
JSON; it does not regenerate CUDA random streams. `policy_discrete.py` is the single
definition of the deployed policy: `T=max(1,ceil(c*D))`, with `c` calibrated on a 0.01
grid to the flat policy's unconditional coverage.

GPU scripts record their protocol in their module docstrings and output JSON. The VGLC
evaluation requires a local checkout of The Video Game Level Corpus at the relative path
documented in `eval_vglc.py`. Retained checkpoints are included for the evaluation-only
probes. SHA-256 hashes in `RELEASE_MANIFEST.sha256` cover every other file in the archive.
`branch_L_clean.py` imports the weighted shortest-path pool builder from the included
`branch_L_transfer.py`.

## Reproduce the fused-kernel latency measurement

Branch O requires Linux/WSL2, an NVIDIA GPU, PyTorch with CUDA, Triton, and `gcc`.
The archived run used Python 3.10, PyTorch 2.12.1+cu130, Triton 3.7.1, CUDA 13.0,
and an RTX 4060 Laptop GPU. From this directory:

```text
python3 -u branch_O_shader.py --out branch_O_shader_v2.json
```

The script records 25 blocked groups and 100 host-observed frame samples per budget,
their raw values and p10/median/p90 summaries, randomized budget order, numerical
parity, and the full measurement boundary. Inputs are already GPU-resident; graph
capture/setup, changed-mask transfer, graph selection, and renderer integration are
not included. Exact latency is hardware- and runtime-dependent.

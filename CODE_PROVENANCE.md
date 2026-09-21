# Source provenance

The old `FROZEN_MANIFEST.v2.sha256` identifies an earlier source snapshot. Against the later original-release files, 26 of its 28 entries match byte for byte. The two differences are fully explained:

| Source | Change after the old snapshot | Numerical effect |
|---|---|---|
| `branch_E_anytime.py` | Top-level horizon description and one printed message now say that the shared cap comes from the maximum training-pool distance | None. The cap formula, sampling and training code are unchanged |
| `branch_F_tortuosity.py` | Generator docstring now states that scenes missing the reachability threshold after the final retry are retained | None. The generator already behaved this way |

`experiments/camera_ready/source_provenance_636.json` contains exact diffs and SHA-256 values. Reversing only these text changes reconstructs the old expected hashes exactly. Removing docstrings and normalizing that one known printed string yields identical Python ASTs. Verify independently with:

```sh
python experiments/camera_ready/audit_source_provenance_636.py
```

This resolves the source mismatch without changing old manifests or rerunning training. It is not a claim that the current files literally match all 28 old entries. The original submission artifact has its own manifest, which does match all 65 entries.

Historical source comments are not the current paper's claims. In particular, unmasked NCA state can propagate through obstacle cells, so geodesic distance is **not** an architectural light-cone lower bound. It is an empirical task-aligned ruler in these experiments. The shared horizon is chosen from training-pool geometry, not independent of every distance statistic.

The public `experiments/camera_ready/protocol.md` is an excerpt of the pre-run internal protocol. It omits local file-management instructions and private review summaries. Scientific experimental instructions remain. Recorded run hashes still identify the original protocol, not this excerpt. Both digests and the transformation are recorded in `SOURCE_PROVENANCE.json`.

The wrappers `reproduce_paired.py`, `verify_release.py` and `verify_curves.py` were added for publication. They are not the programs originally launched to collect the archived results. Original orchestration scripts are retained, but some expect the author's local status files and are not portable entry points.

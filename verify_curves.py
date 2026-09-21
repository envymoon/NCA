"""Independent CPU-only knee reconstruction from released raw revision curves."""
import json
import math
from pathlib import Path
import statistics as st
import numpy as np

HERE = Path(__file__).resolve().parent / 'experiments/camera_ready'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def knee(curve):
    xs = []
    for value in curve:
        if not math.isfinite(float(value)):
            break
        xs.append(float(value))
    if len(xs) < 10:
        return None, None
    smooth = [st.median(xs[max(0, i-2):i+3]) for i in range(len(xs))]
    first = next((i+1 for i in range(len(xs)-9)
                  if all(x <= .05 for x in smooth[i:i+10])), None)
    return first, min(smooth)


def check(rows, arrays, factor=1.):
    count = 0
    for r in rows:
        for metric in ('mean', 'tail'):
            k, floor = knee(arrays[metric][r['pool_idx']].astype(np.float64) * factor)
            assert k == r['knee_' + metric], (r['pool_idx'], metric, k, r['knee_' + metric])
            expected = r['floor_' + metric]
            assert (floor is None and expected is None) or abs(floor - expected) <= 1e-9
            count += 1
    return count


total = 0
for folder in ('width_v1', 'g48_v1', 'reference_g48_v1'):
    for rec in read(HERE / folder / 'results.json')['per_seed']:
        with np.load(HERE / folder / f"s{rec['seed']}" / 'curves.npz') as curves:
            total += check(rec['rows'], curves)
            np.testing.assert_allclose(curves['mean'].mean(axis=0, dtype=np.float64),
                                       rec['aggregate_curve'], rtol=2e-7, atol=1e-8)
        k, floor = knee(rec['aggregate_curve'])
        assert k == rec['aggregate_knee'] and abs(floor-rec['aggregate_floor']) < 1e-9
provenance = read(HERE / 'round2_pool_provenance_v1.json')
for unit in read(HERE / 'round2_paired_v1/results.json')['per_unit']:
    directory = HERE / 'round2_paired_v1' / f"{unit['arm']}_s{unit['seed']}"
    for pool in ('original', 'diagnostic', 'confirmation'):
        summary = read(directory / (pool + '_result.json'))
        factor = provenance[pool]['native_variance'] / provenance['original']['native_variance']
        with np.load(directory / (pool + '_curves.npz')) as curves:
            total += check(summary['rows'], curves, factor)
print(f'Independent revision curve checks: {total} scene-metric curves passed.')
print('Read-only check. No GPU execution, fresh model training or new independent seeds.')

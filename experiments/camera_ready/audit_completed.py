"""CPU-only independent checks of completed camera-ready evidence.

Does not import the training, knee, policy, or timing implementations.
Writes only new audit artifacts; does not modify experiments or manuscript inputs.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(path):
    return Path('C:/' + path[7:]) if path.startswith('/mnt/c/') else Path(path)


def knee(values):
    xs = []
    for v in values:
        if not math.isfinite(float(v)):
            break
        xs.append(float(v))
    if len(xs) < 10:
        return None, None
    smoothed = [st.median(xs[max(0, i-2):i+3]) for i in range(len(xs))]
    first = next((i+1 for i in range(len(xs)-9)
                  if all(v <= .05 for v in smoothed[i:i+10])), None)
    return first, min(smoothed)


def close(a, b, tol=1e-9):
    assert a is not None and b is not None and abs(a-b) <= tol, (a, b)


def budget(c, d):
    return max(1, math.ceil(c * max(1., d)))


def main():
    state = read(HERE / 'suite_v2_status.json')
    assert state['status'] == 'completed'
    assert read(HERE / 'final_status.json')['status'] == 'completed'
    for path, expected in state['protected_files'].items():
        assert sha(local(path)) == expected, path
    for path, expected in state['script_hashes'].items():
        assert sha(HERE / path) == expected, path
    report = {'protected_files_verified': len(state['protected_files']),
              'launch_scripts_verified': len(state['script_hashes']), 'units': [],
              'interpretation': 'post-review diagnostic checks, not additional independent training seeds'}
    row_count = 0
    for folder in ('width_v1', 'g48_v1', 'reference_g48_v1'):
        data = read(HERE / folder / 'results.json')
        for rec in data['per_seed']:
            unit = HERE / folder / ('s' + str(rec['seed']))
            curves = np.load(unit / 'curves.npz')
            assert curves['mean'].shape == curves['tail'].shape == (256, 152)
            assert np.isfinite(curves['mean']).all() and np.isfinite(curves['tail']).all()
            rows = rec['rows']
            assert [r['pool_idx'] for r in rows] == list(range(256))
            for r in rows:
                assert r['eligible'] == (2 < r['dgeo'] <= 2*r['grid'])
                for key in ('mean', 'tail'):
                    k, f = knee(curves[key][r['pool_idx']])
                    assert k == r['knee_' + key]
                    close(f, r['floor_' + key])
                    row_count += 1
            k, f = knee(rec['aggregate_curve'])
            assert k == rec['aggregate_knee']
            close(f, rec['aggregate_floor'])
            np.testing.assert_allclose(curves['mean'].mean(axis=0, dtype=np.float64),
                                       rec['aggregate_curve'], rtol=2e-7, atol=1e-8)
            eligible = [r for r in rows if r['eligible']]
            fit = [r for r in eligible if r['pool_idx'] % 2 == 0]
            hold = [r for r in eligible if r['pool_idx'] % 2 == 1]
            conv = sorted(r['knee_tail'] for r in fit if r['knee_tail'] is not None)
            flat = conv[math.ceil(.95 * len(conv))-1]
            target = sum(r['knee_tail'] is not None and r['knee_tail'] <= flat for r in fit)
            tick = next(t for t in range(1, 10001) if sum(
                r['knee_tail'] is not None and r['knee_tail'] <= budget(round(t/100, 10), r['dgeo'])
                for r in fit) >= target)
            c = tick/100
            p = rec['heldout_policy_tail']
            assert p['flat_T'] == flat and p['geo_c'] == c
            for arm in ('flat', 'geo'):
                bs = [flat if arm == 'flat' else budget(c, r['dgeo']) for r in hold]
                close(st.mean(bs), p[arm+'_cost'])
                served = sum(r['knee_tail'] is not None and r['knee_tail'] <= b for r,b in zip(hold,bs))
                close(served/len(hold), p[arm+'_coverage'])
                actual = sum(curves['tail'][r['pool_idx'], b-1] <= .05 for r,b in zip(hold,bs))
                close(actual/len(hold), p['actual_output_diagnostic'][arm]['raw_quality_coverage_conservative'])
            for key in ('mean', 'tail'):
                valid = [r for r in eligible if r['knee_'+key] is not None]
                metric = rec['per_metric']['knee_'+key]
                assert len(valid) == metric['converged']
                for ruler in ('dgeo','dfree','deuc'):
                    vals = [r['knee_'+key]/max(1.,r[ruler]) for r in valid]
                    close(st.mean(vals), metric['ratios'][ruler]['mean'])
                    close(st.stdev(vals)/st.mean(vals), metric['ratios'][ruler]['cv'])
            raw = np.asarray(rec['aggregate_curve'])
            best = int(raw.argmin())
            errors = curves['mean'][:, best].astype(float)
            largest = np.sort(errors)[-max(1, math.ceil(.05*len(errors))):]
            report['units'].append({'folder': folder, 'seed': rec['seed'],
                'aggregate_knee': k, 'raw_minimum': float(raw[best]), 'raw_minimum_T': best+1,
                'raw_final': float(raw[-1]), 'fraction_of_error_from_worst_5pct_at_raw_min': float(largest.sum()/errors.sum()),
                'median_scene_mean_error_at_raw_min': float(np.median(errors)),
                'n_nonfinite_scenes': rec['n_nonfinite_scenes'],
                'n_geometric_headroom_failures': rec['n_Dgeo_plus_persistence_exceeds_cap'],
                'n_eligible': len(eligible),
                'tail_ever_attained': rec['per_metric']['knee_tail']['converged']})
    report['independently_recomputed_scene_metric_curves'] = row_count
    timing = read(HERE / 'latency_full_population' / 'results.json')
    pooled = {m: [] for m in ('static','dynamic')}
    report['timing'] = {}
    for grid, g in timing['per_grid'].items():
        assert g['full_curve_parity']['knee_match'] == 256
        assert not g['bfs_mismatches']
        assert g['changed_input_parity']['checks'] == 2*g['n_eligible']
        entry = {}
        for mode in pooled:
            rows = g[mode]['rows']
            assert len(rows) == g['n_eligible']
            assert len({r['pool_idx'] for r in rows}) == len(rows)
            pooled[mode].extend(rows)
            for r in rows:
                assert r['geo_T'] == budget(1.29, r['dgeo']) and r['flat_T'] == 39
                assert len(r['flat_us']) == len(r['geo_us']) == 6
                assert all(math.isfinite(v) and v>0 for a in ('flat','geo') for v in r[a+'_us'])
            means = {a: st.mean(st.mean(r[a+'_us']) for r in rows) for a in ('flat','geo')}
            saving = means['flat'] - means['geo']
            close(1-means['geo']/means['flat'], g[mode]['summary']['saving_fraction'])
            overhead = g['setup']['geo']['prepare_capture_first_replay_us'] - g['setup']['flat']['prepare_capture_first_replay_us']
            if mode == 'static':
                overhead += g['static_scene_preparation']['geo_us']['mean']-g['static_scene_preparation']['flat_us']['mean']
            entry[mode] = {'saving_percent': 100*saving/means['flat'],
                          'workload_bank_break_even_frames': math.ceil(overhead/saving) if saving>0 else None,
                          'bank_amortization_note': 'all observed budgets prebuilt; not a single-scene lazy cache or cold-process startup'}
        report['timing'][grid] = entry
    for mode, rows in pooled.items():
        assert len(rows) == 762
        fs = st.mean(st.mean(r['flat_us']) for r in rows)
        gs = st.mean(st.mean(r['geo_us']) for r in rows)
        close(1-gs/fs, timing['pooled'][mode]['saving_fraction'])
    report['passed'] = True
    output = HERE / 'completion_audit_v1.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    manifest = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp'
                and p.name != 'completion_manifest_v1.json'}
    with (HERE / 'completion_manifest_v1.json').open('x', encoding='utf-8') as f:
        json.dump({'files': manifest, 'note': 'New post-review archive only. Original manifests unchanged.'}, f, indent=2)
    print(json.dumps(report, indent=2))
    print('Archived files:', len(manifest))


if __name__ == '__main__':
    main()

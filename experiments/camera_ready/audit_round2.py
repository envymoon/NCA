"""Independent CPU arithmetic audit of all paired outcomes, without training imports."""
import json
import math
from pathlib import Path
import statistics as st
import numpy as np
from audit_completed import read, sha, local, knee, close, budget

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'round2_paired_v1'


def main():
    state = read(ROOT / 'status.json')
    assert state['status'] == 'completed' and len(state['completed']) == 9
    assert sha(HERE / 'round2_paired.py') == state['script_sha256']
    assert sha(HERE / 'round2_decision.json') == state['decision_sha256']
    protected = read(HERE / 'suite_v2_status.json')['protected_files']
    for path, expected in protected.items():
        assert sha(local(path)) == expected, path
    archive = read(HERE / 'completion_manifest_v1.json')['files']
    for path, expected in archive.items():
        assert sha(HERE / path) == expected, path
    data = read(ROOT / 'results.json')
    assert data['protected_files_unchanged'] is True
    assert {k:v for k,v in data.items() if k != 'protected_files_unchanged'} == read(ROOT / 'results.partial.json')
    assert {(u['arm'], u['seed']) for u in data['per_unit']} == {
        (a, s) for a in ('baseline', 'width', 'duration') for s in range(3)}
    prov = read(HERE / 'round2_pool_provenance_v1.json')
    report = {'protected_verified': len(protected), 'previous_archive_verified': len(archive),
              'units': [], 'scene_metric_curves_recomputed': 0}
    for u in data['per_unit']:
        unit = ROOT / f"{u['arm']}_s{u['seed']}"
        assert u == read(unit / 'result.json')
        assert u['training']['status'] in ('completed', 'archived_baseline')
        assert u['parameters'] == (20527 if u['arm'] == 'width' else 5683)
        if u['arm'] == 'width':
            old = read(HERE / 'g48_v1' / f"s{u['seed']}" / 'progress.json')
            assert u['training']['first_32_sampling_sha256'] == old['first_32_sampling_sha256']
        if u['arm'] == 'duration':
            assert u['training']['iterations'] == 40000
            assert u['training']['source_checkpoint_sha256'] == sha(HERE / 'g48_v1' / f"s{u['seed']}" / 'last_periodic.pt')
        c = flat = None
        for pool in ('original', 'diagnostic', 'confirmation'):
            r = read(unit / (pool + '_result.json'))
            assert {k:v for k,v in r.items() if k != 'rows'} == u['evaluation'][pool]
            assert r['input_sha256'] == prov[pool]['input_sha256']
            close(r['normalization_variance'], prov['original']['native_variance'])
            curves = np.load(unit / (pool + '_curves.npz'))
            factor = prov[pool]['native_variance'] / prov['original']['native_variance']
            fixed = {key: curves[key].astype(np.float64) * factor for key in ('mean', 'tail')}
            assert [x['pool_idx'] for x in r['rows']] == list(range(256))
            for key in fixed:
                assert fixed[key].shape == (256, 152) and np.isfinite(fixed[key]).all()
                if u['arm'] == 'baseline' and pool == 'original':
                    old = np.load(HERE / 'g48_v1' / f"s{u['seed']}" / 'curves.npz')
                    np.testing.assert_array_equal(curves[key], old[key])
                for row in r['rows']:
                    assert row['eligible'] == (2 < row['dgeo'] <= 96)
                    k, f = knee(fixed[key][row['pool_idx']])
                    assert k == row['knee_' + key]
                    close(f, row['floor_' + key])
                    report['scene_metric_curves_recomputed'] += 1
            k, f = knee(fixed['mean'].mean(axis=0))
            assert k == r['aggregate_knee']
            close(f, r['aggregate_floor'])
            if k is not None:
                close(k / prov[pool]['Dgeo_mean'], r['aggregate_c1'])
            rows = [x for x in r['rows'] if x['eligible']]
            valid = [x for x in rows if x['knee_tail'] is not None]
            assert len(rows) == r['n_eligible'] and len(valid) == r['tail_attained']
            close(len(valid)/len(rows), r['tail_ceiling'])
            for ruler in ('dgeo', 'dfree', 'deuc'):
                vals = [x['knee_tail']/max(1., x[ruler]) for x in valid]
                metric = r['tail_conditional_ratios'][ruler]
                close(st.mean(vals), metric['mean'])
                close(st.stdev(vals), metric['std'])
                close(st.stdev(vals)/st.mean(vals), metric['cv'])
            p = r['frozen_original_calibration']
            if pool == 'original':
                fit = [x for x in rows if x['pool_idx'] % 2 == 0]
                attained = sorted(x['knee_tail'] for x in fit if x['knee_tail'] is not None)
                flat = attained[math.ceil(.95*len(attained))-1]
                target = sum(x['knee_tail'] is not None and x['knee_tail'] <= flat for x in fit)
                tick = next(t for t in range(1, 10001) if sum(
                    x['knee_tail'] is not None and x['knee_tail'] <= budget(t/100, x['dgeo'])
                    for x in fit) >= target)
                c = tick/100
                rows = [x for x in rows if x['pool_idx'] % 2 == 1]
            assert c == p['geo_c'] and flat == p['flat_T'] and len(rows) == p['n']
            for arm in ('flat', 'geo'):
                bs = [flat if arm == 'flat' else budget(c, x['dgeo']) for x in rows]
                actual = sum(b <= 152 and fixed['tail'][x['pool_idx'], b-1] <= .05 for x,b in zip(rows, bs))
                served = sum(x['knee_tail'] is not None and x['knee_tail'] <= b for x,b in zip(rows, bs))
                close(st.mean(bs), p[arm]['mean_steps'])
                close(actual/len(rows), p[arm]['actual_tail_quality_coverage'])
                close(served/len(rows), p[arm]['knee_service_coverage'])
                assert sum(b > 152 for b in bs) == p[arm]['beyond_horizon']
            close(1-p['geo']['mean_steps']/p['flat']['mean_steps'], p['saving_fraction'])
            close(100*(p['geo']['actual_tail_quality_coverage']-p['flat']['actual_tail_quality_coverage']), p['actual_coverage_difference_pp'])
            report['units'].append({'arm':u['arm'], 'seed':u['seed'], 'pool':pool,
                'aggregate_knee':k, 'aggregate_floor':f, 'tail_attained':len(valid),
                'n_eligible':r['n_eligible'], 'geo_CV':r['tail_conditional_ratios']['dgeo']['cv'],
                'flat_quality':p['flat']['actual_tail_quality_coverage'],
                'geo_quality':p['geo']['actual_tail_quality_coverage'],
                'flat_steps':p['flat']['mean_steps'], 'geo_steps':p['geo']['mean_steps'],
                'step_saving':p['saving_fraction']})
    report['passed'] = True
    with (HERE / 'round2_audit_v1.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    paths = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and (p.relative_to(HERE).parts[0].startswith('round2') or p.name == 'audit_round2.py')]
    with (HERE / 'round2_manifest_v1.json').open('x', encoding='utf-8') as f:
        json.dump({'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}}, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

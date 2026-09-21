"""Collect completed post-review results without modifying any manuscript."""
import argparse
import json

from common import EXP, HERE, ROOT, digest, save, stamp, stats


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def brief(rec):
    tail = rec.get('per_metric', {}).get('knee_tail', {})
    ratios = tail.get('ratios', {})
    return {'seed': rec['seed'], 'status': rec['status'],
            'training_status': rec.get('training', {}).get('status'),
            'aggregate_knee': rec.get('aggregate_knee'),
            'aggregate_c1': rec.get('aggregate_c1'),
            'aggregate_floor': rec.get('aggregate_floor'),
            'tail_ceiling': tail.get('ceiling', 0.),
            'tail_c1_geo': ratios.get('dgeo', {}).get('mean'),
            'tail_CV': {key: value['cv'] for key, value in ratios.items()},
            'heldout_policy_tail': rec.get('heldout_policy_tail')}


def baseline(rows, seed):
    part = [r for r in rows if r['grid'] == 32 and r['seed'] == seed and not r['clamped'] and r['dgeo'] > 2]
    conv = [r for r in part if r['knee_tail'] is not None]
    return {'seed': seed, 'tail_ceiling': len(conv) / len(part),
            'tail_ratios': {ruler: stats([r['knee_tail'] / max(1., r[ruler]) for r in conv])
                            for ruler in ('dgeo', 'dfree', 'deuc')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite-status', default='suite_status.json')
    ap.add_argument('--latency-dir', default='latency_v1')
    a = ap.parse_args()
    width = read(HERE / 'width_v1' / 'results.json')
    new48 = read(HERE / 'g48_v1' / 'results.json')
    old48 = read(HERE / 'reference_g48_v1' / 'results.json')
    timing = read(HERE / a.latency_dir / 'results.json')
    audit = read(EXP / 'audit_full_population.json')
    width_brief = [brief(r) for r in width['per_seed']]
    all48 = sorted(new48['per_seed'] + old48['per_seed'], key=lambda r: r['seed'])
    assert [r['seed'] for r in all48] == [0, 1, 2, 3, 4]
    state = read(HERE / a.suite_status)
    changed = [p for p, expected in state['protected_files'].items() if digest(p) != expected]
    if changed:
        raise RuntimeError('protected inputs changed: ' + ', '.join(changed))
    summary = {'created_utc': stamp(), 'protected_files_unchanged': True,
               'width': {'per_seed': width_brief,
                         'paired_small_model_reference': [baseline(audit['rows'], s) for s in (0, 1, 2)],
                         'tail_ceiling_across_all_seeds': stats([r['tail_ceiling'] for r in width_brief]),
                         'tail_c1_conditional': stats([r['tail_c1_geo'] for r in width_brief]),
                         'interpretation': 'width control at G32 only; same-scene CV comparisons; no universal c=1 claim'},
               'G48': {'per_seed': [brief(r) for r in all48],
                       'n_seeds': 5,
                       'n_aggregate_attained': sum(r.get('aggregate_knee') is not None for r in all48),
                       'aggregate_c1_conditional_on_attainment': stats([r.get('aggregate_c1') for r in all48]),
                       'interpretation': 'original s4 failure retained; post-review completion of seed set, not asymptotic evidence'},
               'latency': {'pooled': timing['pooled'],
                           'per_grid': {g: {'static': v['static']['summary'], 'dynamic': v['dynamic']['summary'],
                                            'setup': v['setup'], 'amortization': v['workload_bank_amortization']}
                                        for g, v in timing['per_grid'].items()},
                           'scope': timing['scope']}}
    save(HERE / 'analysis_full_population.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()



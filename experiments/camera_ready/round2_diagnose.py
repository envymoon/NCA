"""Frozen-checkpoint diagnosis; new outputs only, no optimization."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from common import HERE, EXP, save, stamp, digest
import numpy as np
import torch
import branch_F_tortuosity as BF
import train_controls as TC

OUT = HERE / 'round2_diagnosis_v1'


def analyze(means, tails, rows, horizon):
    m, t = means[:, :horizon], tails[:, :horizon]
    ag = m.astype(np.float64).mean(axis=0)
    km, floor = TC.knee_of(ag)
    eligible = [r['pool_idx'] for r in rows if r['eligible']]
    ks = [TC.knee_of(x)[0] for x in t]
    actual = {str(k): float(np.mean(np.isfinite(t[eligible, k-1]) &
                                    (t[eligible, k-1] <= .05)))
              for k in (48, 96, 152, 304) if k <= horizon}
    bins = {}
    for attr, cuts in [('dgeo', [48,72]), ('tau', [1.5,2.5]),
                       ('low_degree_fraction', [.25,.5])]:
        groups = []
        for low, high in zip([-float('inf')]+cuts, cuts+[float('inf')]):
            ids = [i for i in eligible if low < rows[i][attr] <= high]
            groups.append({'lower_exclusive': low if np.isfinite(low) else None,
                           'upper_inclusive': high if np.isfinite(high) else None,
                           'n': len(ids), 'tail_attained': sum(ks[i] is not None for i in ids),
                           'tail_attainment_fraction': sum(ks[i] is not None for i in ids)/len(ids) if ids else None})
        bins[attr] = groups
    return {'horizon': horizon, 'aggregate_knee': km, 'aggregate_floor': floor,
            'aggregate_raw_min': float(np.nanmin(ag)), 'aggregate_raw_min_T': int(np.nanargmin(ag))+1,
            'aggregate_final': float(ag[-1]), 'n_eligible': len(eligible),
            'tail_attained': sum(ks[i] is not None for i in eligible),
            'tail_attainment_fraction': sum(ks[i] is not None for i in eligible)/len(eligible),
            'nonfinite_scenes': int(np.sum(~np.isfinite(m).all(axis=1) | ~np.isfinite(t).all(axis=1))),
            'actual_tail_quality_coverage': actual, 'diagnostic_bins': bins}


def model_path(seed):
    if seed < 3:
        return HERE / 'g48_v1' / f's{seed}' / 'final.pt'
    return EXP / f'results_E_ext_G48.json.G48.s{seed}.pt'


def run():
    OUT.mkdir(exist_ok=False)
    state = {'status':'running', 'started_utc':stamp(), 'pid':os.getpid(),
             'protocol_sha256':digest(HERE/'round2_protocol.md'), 'completed':[]}
    save(OUT/'status.json', state)
    try:
        previous = json.loads((HERE/'suite_v2_status.json').read_text())
        for p, h in previous['protected_files'].items():
            assert digest(p) == h, p
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        # Synthetic oracle: constant bad error must never attain; sustained good does.
        assert TC.knee_of(np.ones(304))[0] is None
        assert TC.knee_of(np.zeros(304))[0] == 1
        assert TC.knee_of(np.r_[np.ones(299), np.zeros(5)])[0] is None
        TC.CAP = 304
        pools = {name:BF.build_pool_maze(n,48,48,'cuda',torch.Generator(device='cuda').manual_seed(sd))
                 for name,n,sd in [('train',512,0), ('original',256,1), ('diagnostic',256,20260913)]}
        ref_var = pools['original'][1][pools['original'][2]>0].var().item()+1e-12
        output = {'reference_variance':ref_var, 'confirmation_pool_evaluated':False, 'units':[]}
        for seed in range(5):
            model = BF.NCA().cuda()
            path = model_path(seed)
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
            for name, pool in pools.items():
                print(f'START seed={seed} pool={name}',flush=True)
                rec, means, tails = TC.evaluate(model,pool,48,seed,chunk=32)
                reach = pool[2][:,0]>0
                degree = torch.zeros_like(reach,dtype=torch.int32)
                degree[:,1:] += reach[:,:-1]; degree[:,:-1] += reach[:,1:]
                degree[:,:,1:] += reach[:,:,:-1]; degree[:,:,:-1] += reach[:,:,1:]
                fractions = (((degree<=2)&reach).sum((1,2))/reach.sum((1,2)).clamp(min=1)).cpu().tolist()
                rows = rec['rows']
                for r, fraction in zip(rows,fractions):
                    r['low_degree_fraction'] = fraction
                delta = None
                if name == 'original':
                    old = HERE / ('g48_v1' if seed<3 else 'reference_g48_v1') / f's{seed}' / 'curves.npz'
                    with np.load(old) as archived:
                        delta = max(float(np.max(np.abs(means[:,:152]-archived['mean']))),
                                    float(np.max(np.abs(tails[:,:152]-archived['tail']))))
                        np.testing.assert_allclose(means[:,:152],archived['mean'],rtol=1e-5,atol=1e-6)
                        np.testing.assert_allclose(tails[:,:152],archived['tail'],rtol=1e-5,atol=1e-6)
                factor = rec['target_variance']/ref_var
                unit = {'seed':seed, 'pool':name, 'checkpoint_sha256':digest(path),
                        'input_sha256':__import__('hashlib').sha256(pool[0].cpu().numpy().tobytes()).hexdigest(),
                        'native_variance':rec['target_variance'], 'prefix_max_difference':delta,
                        'common_normalization':{str(h):analyze(means*factor,tails*factor,rows,h) for h in (152,304)},
                        'native_normalization':{str(h):analyze(means,tails,rows,h) for h in (152,304)}}
                tag = f'{name}_s{seed}'
                np.savez_compressed(OUT/(tag+'_curves.npz'),mean=means,tail=tails)
                save(OUT/(tag+'_rows.json'),rows)
                output['units'].append(unit)
                save(OUT/'results.partial.json',output)
                brief = unit['common_normalization']['152']
                print(f'DONE seed={seed} pool={name} floor={brief["aggregate_floor"]:.5f} tail={brief["tail_attained"]}/{brief["n_eligible"]}',flush=True)
                state['completed'].append(tag)
                save(OUT/'status.json',state)
            del model
            gc.collect(); torch.cuda.empty_cache()
        save(OUT/'results.json',output)
        state['status']='completed'
    except BaseException as exc:
        state['status']='failed'; state['error']=str(exc)
        traceback.print_exc()
    state['finished_utc']=stamp()
    save(OUT/'status.json',state)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--detach',action='store_true'); a=ap.parse_args()
    if a.detach:
        with (HERE/'round2_diagnosis_v1.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':child.pid}))
    else:
        run()

"""Post-review width and G48 controls, with full-population outcome export."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from common import EXP, HERE, digest, provenance, save, stamp, stats
import numpy as np
import torch
import branch_F_tortuosity as BF
from policy_discrete import eval_policy, fit_matched_policy, geometric_budget

CAP = 152
THETA = .05


def initialize(seed, channels, hidden):
    """Pair training draws with the original default-model initialization."""
    BF.set_seed(seed)
    reference = BF.NCA().cuda()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    if (channels, hidden) == (12, 96):
        return reference
    del reference
    BF.set_seed(seed)
    model = BF.NCA(C=channels, hidden=hidden).cuda()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    return model


def knee_of(values):
    # Preserve a valid early crossing even if the trajectory becomes nonfinite later.
    bad = np.flatnonzero(~np.isfinite(values))
    end = int(bad[0]) if len(bad) else len(values)
    if end < 10:
        return None, None
    return BF.abs_knee({t + 1: float(v) for t, v in enumerate(values[:end])},
                       theta=THETA, persist=10, win=5)


def policy_readout(rows, errors):
    eligible = [r for r in rows if r['eligible']]
    fit = [r for r in eligible if r['pool_idx'] % 2 == 0]
    hold = [r for r in eligible if r['pool_idx'] % 2 == 1]
    if not any(r['knee_tail'] is not None for r in fit) or not hold:
        return {'status': 'cannot_calibrate'}
    c, flat, fit_coverage = fit_matched_policy(fit, 'dgeo', .95, 'knee_tail')
    cg, vg = eval_policy(hold, 'dgeo', c, 'knee_tail')
    cf, vf = eval_policy(hold, 'flat', flat, 'knee_tail')
    actual = {}
    for arm in ('flat', 'geo'):
        served = unobserved = 0
        for row in hold:
            budget = flat if arm == 'flat' else geometric_budget(c, row['dgeo'])
            if budget > CAP:
                unobserved += 1
                continue
            value = errors[row['pool_idx'], budget - 1]
            served += bool(np.isfinite(value) and value <= THETA)
        actual[arm] = {'raw_quality_coverage_conservative': served / len(hold),
                       'budgets_beyond_measured_horizon': unobserved}
    return {'status': 'ok', 'f': .95, 'calibration': 'even pool_idx',
            'evaluation': 'odd pool_idx', 'flat_T': flat, 'geo_c': c,
            'fit_coverage': fit_coverage, 'geo_cost': cg, 'flat_cost': cf,
            'geo_coverage': vg, 'flat_coverage': vf,
            'coverage_difference_pp': 100 * (vg - vf),
            'saving_fraction': 1 - cg / cf,
            'point_weak_pareto_dominance': vg >= vf and cg <= cf and (vg > vf or cg < cf),
            'actual_output_diagnostic': actual}


@torch.no_grad()
def evaluate(model, pool, grid, seed, chunk=32):
    inp, target, mask, dg, df, de, tau = pool
    variance = target[mask > 0].var().item() + 1e-12
    count = len(inp)
    means = np.empty((count, CAP), dtype=np.float32)
    tails = np.empty_like(means)
    sums = {t: 0.0 for t in range(1, CAP + 1)}
    model.eval()
    for start in range(0, count, chunk):
        sl = slice(start, min(start + chunk, count))
        outputs = model.run(inp[sl], CAP, collect=set(sums))
        for t, pred in outputs.items():
            num = (((pred - target[sl]) ** 2) * mask[sl]).sum(dim=(1, 2, 3))
            den = mask[sl].sum(dim=(1, 2, 3)).clamp(min=1)
            error = (num / den / variance).cpu()
            sums[t] += error.sum().item()
            means[sl, t - 1] = error.numpy()
            tails[sl, t - 1] = BF.tail_nmse(pred, target[sl], mask[sl], variance, q=.95).cpu().numpy()
        del outputs
    full_curve = np.asarray([sums[t] / count for t in sums])
    agg_knee, floor = knee_of(full_curve)
    rows = []
    for i in range(count):
        km, fm = knee_of(means[i])
        kt, ft = knee_of(tails[i])
        rows.append({'grid': grid, 'seed': seed, 'pool_idx': i,
                     'dgeo': float(dg[i]), 'dfree': float(df[i]), 'deuc': float(de[i]),
                     'tau': float(tau[i]), 'clamped': bool(dg[i] > 2 * grid),
                     'eligible': bool(dg[i] <= 2 * grid and dg[i] > 2),
                     'knee_mean': km, 'knee_tail': kt, 'floor_mean': fm, 'floor_tail': ft,
                     'nonfinite_mean_steps': int((~np.isfinite(means[i])).sum()),
                     'nonfinite_tail_steps': int((~np.isfinite(tails[i])).sum())})
    eligible = [r for r in rows if r['eligible']]
    per_metric = {}
    for key in ('knee_mean', 'knee_tail'):
        converged = [r for r in eligible if r[key] is not None]
        per_metric[key] = {'converged': len(converged), 'n': len(eligible),
                           'ceiling': len(converged) / len(eligible),
                           'ratios': {ruler: stats([r[key] / max(1., r[ruler]) for r in converged])
                                      for ruler in ('dgeo', 'dfree', 'deuc')}}
    both = [r for r in eligible if r['knee_mean'] is not None and r['knee_tail'] is not None]
    joint = {key: {ruler: stats([r[key] / max(1., r[ruler]) for r in both])
                    for ruler in ('dgeo', 'dfree', 'deuc')}
             for key in ('knee_mean', 'knee_tail')}
    eligible_curve = means[[r['pool_idx'] for r in eligible]].mean(axis=0)
    eknee, efloor = knee_of(eligible_curve)
    rec = {'aggregate_knee': agg_knee, 'aggregate_floor': floor,
           'aggregate_c1': agg_knee / dg.mean().item() if agg_knee is not None else None,
           'aggregate_eligible_knee': eknee, 'aggregate_eligible_floor': efloor,
           'status': 'attained' if agg_knee is not None else 'finite_threshold_failure',
           'target_variance': variance, 'n_eligible': len(eligible),
           'n_clamped': sum(r['clamped'] for r in rows),
           'n_Dgeo_plus_persistence_exceeds_cap': sum(r['dgeo'] + 9 > CAP for r in eligible),
           'n_nonfinite_scenes': sum(r['nonfinite_mean_steps'] > 0 or r['nonfinite_tail_steps'] > 0 for r in rows),
           'per_metric': per_metric, 'n_converged_both': len(both),
           'joint_population_sensitivity': joint, 'rows': rows,
           'heldout_policy_tail': policy_readout(rows, tails),
           'aggregate_curve': full_curve.tolist(),
           'pool_Dgeo_mean': dg.mean().item(),
           'pool_tau_quantiles': torch.quantile(tau, torch.tensor([.1, .5, .9])).tolist()}
    if rec['n_nonfinite_scenes']:
        rec['status'] = 'contains_nonfinite_evaluation_trajectories'
    return rec, means, tails


def train(seed, channels, hidden, pool, iterations, directory):
    inp, target, mask, *_ = pool
    variance = target[mask > 0].var().item() + 1e-12
    model = initialize(seed, channels, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    samples_hash = hashlib.sha256()
    started = time.perf_counter()
    last_save = 0
    for it in range(iterations):
        idx = torch.randint(0, len(inp), (12,), device='cuda')
        T = int(torch.randint(4, CAP + 1, (1,)).item())
        if it < 32:
            samples_hash.update(idx.cpu().numpy().tobytes())
            samples_hash.update(str(T).encode('ascii'))
        out = model.run_train(inp[idx], T, seg=24)
        loss = BF.masked_nmse(out, target[idx], mask[idx], variance)
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            return None, {'status': 'nonfinite_training_loss', 'iteration': it + 1,
                          'T': T, 'last_periodic_checkpoint_iteration': last_save,
                          'seconds': time.perf_counter() - started}
        loss.backward()
        try:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError as exc:
            if 'non-finite' not in str(exc):
                raise
            return None, {'status': 'nonfinite_training_gradient', 'iteration': it + 1,
                          'T': T, 'last_periodic_checkpoint_iteration': last_save,
                          'seconds': time.perf_counter() - started}
        optimizer.step()
        if (it + 1) % 1000 == 0 or it + 1 == iterations:
            elapsed = time.perf_counter() - started
            rec = {'iteration': it + 1, 'iterations': iterations, 'seed': seed,
                   'loss': loss.item(), 'gradient_norm_before_clip': norm.item(),
                   'seconds': elapsed, 'estimated_remaining_seconds': elapsed * (iterations - it - 1) / (it + 1),
                   'first_32_sampling_sha256': samples_hash.hexdigest()}
            save(directory / 'progress.json', rec)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                        'metadata': rec}, directory / 'last_periodic.pt')
            last_save = it + 1
            print(f"s{seed} C={channels} iter={it+1}/{iterations} loss={loss.item():.5f} "
                  f"elapsed={elapsed:.0f}s", flush=True)
    return model, {'status': 'completed', 'iterations': iterations,
                   'seconds': time.perf_counter() - started,
                   'first_32_sampling_sha256': samples_hash.hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['width', 'g48', 'reference', 'smoke'], required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    folder = HERE / a.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / 'STARTED.json'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'utc': stamp(), 'args': vars(a), 'sources': provenance(),
                   'script_sha256': digest(__file__)}, f, indent=2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    G = 48 if a.mode in ('g48', 'reference') else 32
    C, H = (24, 192) if a.mode in ('width', 'smoke') else (12, 96)
    ntrain, neval = (24, 16) if a.mode == 'smoke' else (512, 256)
    pool = BF.build_pool_maze(ntrain, G, G, 'cuda', torch.Generator(device='cuda').manual_seed(0))
    val = BF.build_pool_maze(neval, G, G, 'cuda', torch.Generator(device='cuda').manual_seed(1))
    pool_hash = hashlib.sha256(val[0].cpu().numpy().tobytes()).hexdigest()
    all_results = {'mode': a.mode, 'grid': G, 'channels': C, 'hidden': H,
                   'cap': CAP, 'theta': THETA, 'pool_sha256': pool_hash,
                   'requested_seeds': a.seeds, 'per_seed': []}
    for sd in a.seeds:
        unit = folder / f's{sd}'
        unit.mkdir()
        if a.mode == 'reference':
            path = EXP / f'results_E_ext_G48.json.G48.s{sd}.pt'
            model = BF.NCA().cuda()
            model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
            training = {'status': 'archived_evaluation_only', 'checkpoint_sha256': digest(path)}
        else:
            model, training = train(sd, C, H, pool, 3 if a.mode == 'smoke' else 20000, unit)
        if model is None:
            rec = {'seed': sd, 'training': training, 'status': training['status'],
                   'unconditional_model_success': False}
        else:
            if a.mode != 'reference':
                torch.save(model.state_dict(), unit / 'final.pt')
            rec, means, tails = evaluate(model, val, G, sd, chunk=32 if G == 48 else 64)
            rec.update({'seed': sd, 'training': training,
                        'parameters': sum(p.numel() for p in model.parameters())})
            np.savez_compressed(unit / 'curves.npz', mean=means, tail=tails)
            if a.mode == 'reference':
                archived = json.loads((EXP / 'results_E_ext_G48.json').read_text())
                old = next(r for r in archived['per_seed'] if r['seed'] == sd)
                rec['reference_check'] = {'expected_knee': old['agg_knee'],
                                           'knee_match': rec['aggregate_knee'] == old['agg_knee'],
                                           'floor_delta': rec['aggregate_floor'] - old['floor']}
                if not rec['reference_check']['knee_match'] or abs(rec['reference_check']['floor_delta']) > 1e-6:
                    save(unit / 'result.json', rec)
                    raise RuntimeError('archived G48 reference mismatch')
            del model
        save(unit / 'result.json', rec)
        all_results['per_seed'].append(rec)
        save(folder / 'results.partial.json', all_results)
        print(f"{a.mode} seed={sd}: {rec['status']}, aggregate knee={rec.get('aggregate_knee')}, "
              f"floor={rec.get('aggregate_floor')}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    all_results['finished_utc'] = stamp()
    save(folder / 'results.json', all_results)


if __name__ == '__main__':
    main()

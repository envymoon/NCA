"""Paired latency with CPU-origin changed inputs and graph setup accounting.

This remains an isolated CUDA workload. It does not measure a renderer or GPU-origin
mask readback. The f=.95 constants are read from the archived audit without refitting.
"""
import argparse
import gc
import json
import math
import random
import statistics as st
import tempfile
import time

from common import EXP, HERE, digest, provenance, save, stamp
import numpy as np
import torch
import triton
import branch_F_tortuosity as BF
import branch_O_shader as O
from policy_discrete import geometric_budget


def now():
    return time.perf_counter_ns()


def summary(values):
    xs = np.asarray(values, dtype=np.float64)
    return {'n': len(values), 'mean': float(xs.mean()), 'median': float(np.median(xs)),
            'p10': float(np.quantile(xs, .1)), 'p90': float(np.quantile(xs, .9))}


def paired_summary(rows, bootstrap=2000):
    # The scene is the unit. Repeats for the same scene are not independent samples.
    flat = np.asarray([st.mean(r['flat_us']) for r in rows])
    geo = np.asarray([st.mean(r['geo_us']) for r in rows])
    rng = np.random.default_rng(20260912)
    reps = []
    by_grid = {}
    for i, row in enumerate(rows):
        by_grid.setdefault(row['grid'], []).append(i)
    for _ in range(bootstrap):
        idx = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in by_grid.values()])
        reps.append(1 - geo[idx].mean() / flat[idx].mean())
    return {'n_scenes': len(rows), 'flat_us': summary(flat), 'geo_us': summary(geo),
            'saving_us': float((flat - geo).mean()),
            'saving_fraction': float(1 - geo.mean() / flat.mean()),
            'saving_fraction_paired_scene_bootstrap_95ci': np.quantile(reps, [.025, .975]).tolist(),
            'bootstrap_replicates': bootstrap,
            'estimand': 'ratio of scene-weighted arithmetic means, all repeats retained'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    folder = HERE / a.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'STARTED.json').open('x') as f:
        json.dump({'started_utc': stamp(), 'sources': provenance(), 'script_sha256': digest(__file__)}, f, indent=2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = json.loads((EXP / O.AUDIT).read_text())
    policy = next(p for p in audit['policy_full'] if p['f'] == .95)
    Tf, c = policy['flat_T'], policy['geo_c']
    assert (Tf, c) == (39, 1.29)
    grids = [16] if a.smoke else [16, 24, 32]
    nscene, repeats = (8, 2) if a.smoke else (256, 6)
    result = {'started_utc': stamp(), 'smoke_only': a.smoke, 'flat_T': Tf, 'geo_c': c,
              'f': .95, 'checkpoint_seed': 3, 'repeats_per_scene_arm': repeats,
              'sources': provenance(), 'kernel_sha256': digest(EXP / 'branch_O_shader.py'),
              'environment': {'torch': torch.__version__, 'triton': triton.__version__,
                              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
              'scope': {'static': 'resident scene, cached integer budget, graph lookup, replay, synchronization',
                        'dynamic': 'CPU float32 seed/mask input, packing, pinned staging, H2D copy, graph lookup, replay, synchronization; geo also mask conversion, source extraction, native BFS, ceil policy',
                        'setup': 'within an initialized CUDA process; model loading and shared kernel first compilation measured separately; graph bank has workload-specific observed budgets',
                        'excluded': 'renderer/engine scheduling, GPU-origin readback, disk I/O per frame, cold process/CUDA context initialization, display presentation',
                        'primary_estimator': 'ratio of scene-weighted arithmetic mean times, directly timed paired frames',
                        'eligibility': 'Dgeo <= 2G, identical to the frozen audit and original Branch O; Dgeo <= 2 retained',
                        'parity_reference_precision': 'TF32 disabled; IEEE FP32'},
              'per_grid': {}}
    pool_rows = {'static': [], 'dynamic': []}
    with tempfile.TemporaryDirectory(prefix='cr_latency_') as temp:
        lib = O.build_bfs(temp)
        for G in grids:
            rng = random.Random(20260912 + G)
            pool = BF.build_pool_maze(nscene, G, G, 'cuda', torch.Generator(device='cuda').manual_seed(1))
            vinp, vtgt, vlm, dg, *_ = pool
            cpu = vinp.cpu()
            cpu_np = cpu.numpy()
            eligible = [i for i in range(nscene) if dg[i] <= 2 * G]
            budgets = {i: geometric_budget(c, dg[i]) for i in eligible}
            loading = now()
            path = EXP / O.CKPT.format(G=G, S=3)
            sd = torch.load(path, map_location='cuda', weights_only=True)
            sh = O.Shader(sd, G, G, 1, 'cuda')
            torch.cuda.synchronize()
            shared_load_us = (now() - loading) / 1e3
            model = BF.NCA().cuda()
            model.load_state_dict(sd)
            model.eval()
            sh.set_scenes(vinp[:1])
            compiling = now()
            sh.frame_ops(1)
            torch.cuda.synchronize()
            shared_first_kernel_use_us = (now() - compiling) / 1e3
            gres = {'n_eligible': len(eligible), 'n_pool': nscene,
                    'checkpoint_sha256': digest(path), 'shared_model_buffer_load_us': shared_load_us,
                    'shared_first_kernel_use_us': shared_first_kernel_use_us,
                    'common_resident_tensor_bytes': sh.mem_bytes()}
            # Strong parity check from the submitted implementation, on this environment.
            pshader = O.Shader(sd, G, G, nscene, 'cuda')
            parity = O.parity_phase(model, pshader, vinp, vtgt, vlm,
                                    vtgt[vlm > 0].var().item() + 1e-12, O.CAP, 'cuda')
            gres['full_curve_parity'] = parity
            assert parity['knee_match'] == nscene, f'knee parity failure G={G}'
            del pshader
            gc.collect()
            torch.cuda.empty_cache()

            def bfs(i):
                # These conversions belong to the geo path's timed CPU work.
                obstacle = (cpu_np[i, 1] > .5).astype(np.uint8)
                pos = int(cpu_np[i, 0].argmax())
                return O.bfs_native(lib, obstacle, pos // G, pos % G)

            mismatches = [i for i in range(nscene) if bfs(i) != int(dg[i])]
            assert not mismatches, f'BFS mismatch G={G}'
            gres['bfs_mismatches'] = mismatches
            pinned = torch.empty(G * G, 2, pin_memory=True)

            def transfer(i):
                packed = cpu[i].permute(1, 2, 0).reshape(G * G, 2).contiguous()
                pinned.copy_(packed)
                sh.INP.copy_(pinned, non_blocking=True)

            # Each arm pays for its own graph bank. All captures use warmed kernels.
            sets = {'flat': [Tf], 'geo': sorted(set(budgets.values()))}
            banks, setup = {}, {}
            arm_order = ['flat', 'geo']
            rng.shuffle(arm_order)
            for arm in arm_order:
                before = torch.cuda.mem_get_info()[0]
                t0 = now()
                banks[arm] = {T: O.capture_frame(sh, T) for T in sets[arm]}
                # Ensure first replay/instantiation costs belong to setup.
                for graph in banks[arm].values():
                    graph.replay()
                torch.cuda.synchronize()
                elapsed = (now() - t0) / 1e3
                after = torch.cuda.mem_get_info()[0]
                setup[arm] = {'prepare_capture_first_replay_us': elapsed,
                              'n_graphs': len(sets[arm]), 'budgets': sets[arm],
                              'driver_free_memory_delta_bytes': before - after,
                              'memory_note': 'coarse process/device diagnostic, not exact graph metadata bytes'}
            gres['setup'] = setup
            gres['setup_arm_order'] = arm_order

            preparation = []
            components = {'bfs_and_policy_us': [], 'pack_transfer_complete_us': [],
                          'graph_selection_us': []}
            for i in eligible:
                rec = {'scene': i}
                arms = ['flat', 'geo']
                rng.shuffle(arms)
                for arm in arms:
                    t0 = now()
                    T = geometric_budget(c, bfs(i)) if arm == 'geo' else Tf
                    transfer(i)
                    selected = banks[arm][T]
                    torch.cuda.synchronize()
                    rec[arm + '_us'] = (now() - t0) / 1e3
                preparation.append(rec)
                t0 = now()
                T = geometric_budget(c, bfs(i))
                components['bfs_and_policy_us'].append((now() - t0) / 1e3)
                t0 = now()
                transfer(i)
                torch.cuda.synchronize()
                components['pack_transfer_complete_us'].append((now() - t0) / 1e3)
                t0 = now()
                selected = banks['geo'][T]
                components['graph_selection_us'].append((now() - t0) / 1e3)
            gres['static_scene_preparation'] = {'rows': preparation,
                                                'flat_us': summary([r['flat_us'] for r in preparation]),
                                                'geo_us': summary([r['geo_us'] for r in preparation])}
            gres['component_diagnostics'] = {name: {'summary': summary(xs), 'samples': xs}
                                             for name, xs in components.items()}

            # Validate the actual changing-input path at both selected budgets.
            changed_deltas = []
            for i in eligible:
                for arm, T in [('flat', Tf), ('geo', budgets[i])]:
                    transfer(i)
                    banks[arm][T].replay()
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        reference = model.run(vinp[i:i + 1], T)
                        delta = (sh.readout_img() - reference).abs().max().item()
                    if not math.isfinite(delta) or not torch.allclose(sh.readout_img(), reference, atol=5e-4, rtol=1e-4):
                        raise RuntimeError(f'changed-input parity failure G={G} scene={i} arm={arm}: {delta}')
                    changed_deltas.append(delta)
            gres['changed_input_parity'] = {'checks': len(changed_deltas), 'max_abs_readout_delta': max(changed_deltas)}

            def frame(i, arm, dynamic):
                start = now()
                if dynamic:
                    T = geometric_budget(c, bfs(i)) if arm == 'geo' else Tf
                    transfer(i)
                else:
                    T = budgets[i] if arm == 'geo' else Tf
                graph = banks[arm][T]
                graph.replay()
                torch.cuda.synchronize()
                return (now() - start) / 1e3

            for mode in ('static', 'dynamic'):
                dynamic = mode == 'dynamic'
                rows = {i: {'grid': G, 'pool_idx': i, 'dgeo': float(dg[i]),
                            'flat_T': Tf, 'geo_T': budgets[i], 'flat_us': [], 'geo_us': []}
                        for i in eligible}
                for i in eligible[:min(8, len(eligible))]:
                    transfer(i)
                    torch.cuda.synchronize()
                    for _ in range(3):
                        frame(i, 'flat', dynamic)
                        frame(i, 'geo', dynamic)
                for repeat in range(repeats):
                    order = eligible[:]
                    rng.shuffle(order)
                    for i in order:
                        if not dynamic:
                            transfer(i)
                            torch.cuda.synchronize()
                        arms = ['flat', 'geo']
                        rng.shuffle(arms)
                        for arm in arms:
                            rows[i][arm + '_us'].append(frame(i, arm, dynamic))
                    print(f'latency G={G} {mode} repeat={repeat+1}/{repeats}', flush=True)
                records = list(rows.values())
                gres[mode] = {'summary': paired_summary(records), 'rows': records}
                pool_rows[mode].extend(records)

            # Direct startup measurements for a new single-scene cache entry.
            lazy = []
            for i in eligible[:min(8, len(eligible))]:
                rec = {'scene': i}
                arms = ['flat', 'geo']
                rng.shuffle(arms)
                for arm in arms:
                    t0 = now()
                    T = geometric_budget(c, bfs(i)) if arm == 'geo' else Tf
                    transfer(i)
                    graph = O.capture_frame(sh, T)
                    graph.replay()
                    torch.cuda.synchronize()
                    rec[arm + '_first_frame_us'] = (now() - t0) / 1e3
                    del graph
                lazy.append(rec)
            gres['lazy_cache_miss_frames'] = lazy
            gres['workload_bank_amortization'] = []
            for frames in (1, 10, 100, 1000):
                for mode in ('static', 'dynamic'):
                    s = gres[mode]['summary']
                    shared = shared_load_us + shared_first_kernel_use_us
                    flat_total = shared + setup['flat']['prepare_capture_first_replay_us'] + frames * s['flat_us']['mean']
                    geo_total = shared + setup['geo']['prepare_capture_first_replay_us'] + frames * s['geo_us']['mean']
                    if mode == 'static':
                        flat_total += gres['static_scene_preparation']['flat_us']['mean']
                        geo_total += gres['static_scene_preparation']['geo_us']['mean']
                    gres['workload_bank_amortization'].append({'frames': frames, 'mode': mode,
                                                              'flat_total_us': flat_total, 'geo_total_us': geo_total,
                                                              'saving_fraction': 1 - geo_total / flat_total,
                                                              'scope': 'shared load/first kernel use plus workload-specific bank setup plus measured frames; static includes one mean scene preparation'})
            result['per_grid'][str(G)] = gres
            save(folder / 'results.partial.json', result)
            for mode in ('static', 'dynamic'):
                print(f"G={G} {mode}: saving={100*gres[mode]['summary']['saving_fraction']:.2f}%", flush=True)
            del banks, sh, model, vinp, vtgt, vlm, pool, cpu, cpu_np, pinned
            gc.collect()
            torch.cuda.empty_cache()
    result['pooled'] = {mode: paired_summary(rows) for mode, rows in pool_rows.items()}
    result['finished_utc'] = stamp()
    save(folder / 'results.json', result)
    print('latency complete', flush=True)


if __name__ == '__main__':
    main()



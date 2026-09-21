"""Portable entry point reusing the archived paired-experiment helpers."""
import argparse
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CR = ROOT / 'experiments/camera_ready'
sys.path.insert(0, str(CR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('replay', 'train'), required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--baseline-dir', type=Path, default=CR / 'g48_v1')
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import branch_F_tortuosity as BF
    import train_controls as TC
    import round2_paired as RP
    from common import save, digest
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    TC.CAP = 152
    trainpool = BF.build_pool_maze(512, 48, 48, 'cuda', torch.Generator(device='cuda').manual_seed(0))
    pools = {name: BF.build_pool_maze(256, 48, 48, 'cuda', torch.Generator(device='cuda').manual_seed(seed))
             for name, seed in [('original', 1), ('diagnostic', 20260913), ('confirmation', 20260914)]}
    records = []
    for arm in ('baseline', 'width', 'duration'):
        for seed in (0, 1, 2):
            unit = out / f'{arm}_s{seed}'
            unit.mkdir()
            if arm == 'baseline' or args.mode == 'replay':
                path = ((args.baseline_dir / f's{seed}') if arm == 'baseline' else
                        CR / 'round2_paired_v1' / f'{arm}_s{seed}') / 'final.pt'
                model = BF.NCA(C=24, hidden=192).cuda() if arm == 'width' else BF.NCA().cuda()
                model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
                training = {'status': 'archived_evaluation_only', 'checkpoint_sha256': digest(path)}
            elif arm == 'width':
                model, training = TC.train(seed, 24, 192, trainpool, 20000, unit)
            else:
                path = args.baseline_dir / f's{seed}/last_periodic.pt'
                ckpt = torch.load(path, map_location='cuda', weights_only=True)
                assert ckpt['metadata']['iteration'] == 20000
                model = BF.NCA().cuda()
                model.load_state_dict(ckpt['model'])
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                optimizer.load_state_dict(ckpt['optimizer'])
                assert all(g['lr'] == .001 for g in optimizer.param_groups)
                torch.set_rng_state(ckpt['cpu_rng'].cpu())
                torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu())
                model, training = RP.advance(model, optimizer, trainpool, 20000, 40000, unit, seed)
            rec = {'arm': arm, 'seed': seed, 'training': training}
            if model is not None:
                rec['evaluation'] = RP.evaluate_unit(model, pools, seed, unit)
                if args.mode == 'train' and arm != 'baseline':
                    torch.save(model.state_dict(), unit / 'final.pt')
                del model
            records.append(rec)
            save(unit / 'result.json', rec)
            save(out / 'results.partial.json', {'per_unit': records})
            gc.collect()
            torch.cuda.empty_cache()
    save(out / 'results.json', {'mode': args.mode, 'per_unit': records})


if __name__ == '__main__':
    # common inserts experiments/ into sys.path without importing GPU packages.
    import common
    main()

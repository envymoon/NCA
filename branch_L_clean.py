from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics as st
import sys
import time
from pathlib import Path

import torch

import branch_F_tortuosity as BF
from branch_L_transfer import build_pool_medium


SCHEMA_VERSION = "branch_L_clean.v1"


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def safe_agg(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "vals": []}
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "std": st.pstdev(vals) if len(vals) >= 2 else None,
        "vals": vals,
    }


def cv(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) < 2 or st.mean(vals) == 0:
        return None
    return st.pstdev(vals) / st.mean(vals)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def compatible_nca3(seed, device):
    BF.set_seed(seed)
    base = BF.NCA(in_ch=2).to(device)
    extended = BF.NCA(in_ch=3).to(device)
    with torch.no_grad():
        active_perception = base.perceive.weight.shape[0]
        extended.perceive.weight[:active_perception].copy_(base.perceive.weight)
        extended.w1.weight[:, :active_perception].copy_(base.w1.weight)
        extended.w1.bias.copy_(base.w1.bias)
        extended.w2.weight.copy_(base.w2.weight)
        extended.w2.bias.copy_(base.w2.bias)
        extended.readout.weight.copy_(base.readout.weight)
        extended.readout.bias.copy_(base.readout.bias)

    BF.set_seed(seed + 100_000)
    return extended


def assert_control_compatibility(device):
    model2 = BF.NCA(in_ch=2).to(device)

    model3 = BF.NCA(in_ch=3).to(device)
    with torch.no_grad():
        n = model2.perceive.weight.shape[0]
        model3.perceive.weight[:n].copy_(model2.perceive.weight)
        model3.w1.weight[:, :n].copy_(model2.w1.weight)
        model3.w1.bias.copy_(model2.w1.bias)
        model3.w2.weight.copy_(model2.w2.weight)
        model3.w2.bias.copy_(model2.w2.bias)
        model3.readout.weight.copy_(model2.readout.weight)
        model3.readout.bias.copy_(model2.readout.bias)
    x2 = torch.randn(2, 2, 7, 7, device=device)
    x3 = torch.cat([x2, torch.zeros(2, 1, 7, 7, device=device)], dim=1)
    with torch.no_grad():
        delta = (model2.run(x2, 5) - model3.run(x3, 5)).abs().max().item()
    if delta > 1e-6:
        raise AssertionError(f"two/three-channel control initialization mismatch: {delta}")


def make_paired_pools(args, device, small=False):
    n_train = min(args.pool, 8) if small else args.pool
    n_val = min(args.test_per, 8) if small else args.test_per
    pools, vpools = {}, {}
    for het in args.hets:
        pools[het] = build_pool_medium(
            n_train, args.grid, args.grid, device, het,
            torch.Generator(device=device).manual_seed(args.train_scene_seed),
            walls=tuple(args.walls), noise=args.noise,
        )
        vpools[het] = build_pool_medium(
            n_val, args.grid, args.grid, device, het,
            torch.Generator(device=device).manual_seed(args.val_scene_seed),
            walls=tuple(args.walls), noise=args.noise,
        )

    ref = args.hets[0]
    for name, collection in (("train", pools), ("validation", vpools)):
        for het in args.hets[1:]:

            if not torch.equal(collection[ref][0][:, :2], collection[het][0][:, :2]):
                raise AssertionError(f"{name} masks are not paired for het={het}")
            for idx, label in ((3, "Dgeo"), (4, "Dfree"), (5, "Deuc"), (7, "tau")):
                if not torch.equal(collection[ref][idx], collection[het][idx]):
                    raise AssertionError(f"{name} {label} is not paired for het={het}")


    if 0.0 in pools:
        inp, tgt, lm, *_ = pools[0.0]
        d = BF.bfs8(inp[:, :1], inp[:, 1:2], max_iter=8 * (2 * args.grid))
        reach = (d < BF.INF / 2).float()
        expected = torch.clamp(
            torch.where(reach > 0, d, torch.full_like(d, 2.0 * args.grid)),
            max=2.0 * args.grid,
        ) / (2.0 * args.grid)
        err = (((tgt - expected).abs()) * lm).max().item()
        if err > 1e-6:
            raise AssertionError(f"het=0 does not reduce to BFS distance target: max error={err}")
    return pools, vpools


def per_scene_metrics(pred, tgt, mask, var, tail_q):
    num = (((pred - tgt) ** 2) * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp(min=1)
    mean = num / den / var
    tail = BF.tail_nmse(pred, tgt, mask, var, q=tail_q)
    return mean, tail


def knee_from_series(series, theta):
    knee, floor = BF.abs_knee(series, theta=theta)
    return knee, finite_or_none(floor), finite_or_none(BF.curve_drift(series))


def evaluate(model, vinp, vtgt, vlm, vvar, args, eligible):
    times = range(1, args.cap + 1)
    mean_by_t = {t: [] for t in times}
    tail_by_t = {t: [] for t in times}
    with torch.no_grad():
        for start in range(0, len(vinp), args.eval_batch):
            sl = slice(start, min(len(vinp), start + args.eval_batch))
            outs = model.run(vinp[sl], args.cap, collect=set(times))
            for t in times:
                mean, tail = per_scene_metrics(outs[t], vtgt[sl], vlm[sl], vvar, args.tail_q)
                mean_by_t[t].append(mean.cpu())
                tail_by_t[t].append(tail.cpu())
    mean_by_t = {t: torch.cat(v) for t, v in mean_by_t.items()}
    tail_by_t = {t: torch.cat(v) for t, v in tail_by_t.items()}

    eligible_cpu = eligible.cpu()
    aggregate_mean = {t: mean_by_t[t][eligible_cpu].mean().item() for t in times}
    aggregate_tail = {t: tail_by_t[t][eligible_cpu].mean().item() for t in times}
    mean_knee, mean_floor, mean_drift = knee_from_series(aggregate_mean, args.theta)
    tail_knee, tail_floor, tail_drift = knee_from_series(aggregate_tail, args.theta)

    scene_knees = []
    for i in range(len(vinp)):
        mk, mf, md = knee_from_series({t: mean_by_t[t][i].item() for t in times}, args.theta)
        tk, tf, td = knee_from_series({t: tail_by_t[t][i].item() for t in times}, args.theta)
        scene_knees.append({
            "scene_id": i,
            "eligible": bool(eligible_cpu[i]),
            "mean_knee": mk,
            "mean_floor": mf,
            "mean_drift": md,
            "tail_knee": tk,
            "tail_floor": tf,
            "tail_drift": td,
        })
    return {
        "aggregate_mean_curve": {str(k): finite_or_none(v) for k, v in aggregate_mean.items()},
        "aggregate_tail_curve": {str(k): finite_or_none(v) for k, v in aggregate_tail.items()},
        "aggregate_mean_knee": mean_knee,
        "aggregate_mean_floor": mean_floor,
        "aggregate_mean_drift": mean_drift,
        "aggregate_tail_knee": tail_knee,
        "aggregate_tail_floor": tail_floor,
        "aggregate_tail_drift": tail_drift,
        "scene_knees": scene_knees,
    }


def summarize_seed(eval_result, dgeo, dfree, deuc, eligible):
    rows = eval_result["scene_knees"]
    eligible_ids = [i for i, ok in enumerate(eligible.tolist()) if ok]
    converged = [i for i in eligible_ids if rows[i]["tail_knee"] is not None]
    ratios = {
        "geo": [rows[i]["tail_knee"] / max(float(dgeo[i]), 1.0) for i in converged],
        "free": [rows[i]["tail_knee"] / max(float(dfree[i]), 1.0) for i in converged],
        "euc": [rows[i]["tail_knee"] / max(float(deuc[i]), 1.0) for i in converged],
    }
    return {
        "eligible_scenes": len(eligible_ids),
        "tail_converged_scenes": len(converged),
        "tail_unconditional_coverage": len(converged) / len(eligible_ids) if eligible_ids else None,
        "tail_ratio_cv_conditional": {k: finite_or_none(cv(v)) if cv(v) is not None else None
                                      for k, v in ratios.items()},
        "tail_ratio": {k: safe_agg(v) for k, v in ratios.items()},
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--hets", type=float, nargs="+", default=[0.0, 0.9])
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--test_per", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=152)
    ap.add_argument("--theta", type=float, default=0.05)
    ap.add_argument("--tail_q", type=float, default=0.95)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--walls", type=int, nargs=2, default=[1, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train_scene_seed", type=int, default=0)
    ap.add_argument("--val_scene_seed", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results_L_clean_G32.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preflight_only", action="store_true")
    return ap.parse_args()


def validate_args(args):
    if args.grid != 32:
        raise ValueError("submission protocol is locked to grid=32")
    if args.cap != 152:
        raise ValueError("submission protocol is locked to cap=152 (the final E cap)")
    if sorted(args.hets) != [0.0, 0.9]:
        raise ValueError("submission protocol is locked to paired het={0.0,0.9}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("model seeds must be unique")
    if not (0 < args.tail_q < 1):
        raise ValueError("tail_q must be in (0,1)")


def main():
    args = parse_args()
    validate_args(args)
    device = BF.get_device(args.device)
    assert_control_compatibility(device)

    if args.preflight_only:
        make_paired_pools(args, device, small=True)
        print("PREFLIGHT PASS: paired scenes, het=0 BFS degeneration, and control initialization verified")
        return

    out = Path(args.out)
    partial = Path(str(out) + ".partial")
    if not args.overwrite and (out.exists() or partial.exists()):
        raise FileExistsError(f"refusing to overwrite {out} or {partial}; choose a new --out")

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    pools, vpools = make_paired_pools(args, device)
    result = {
        "schema": SCHEMA_VERSION,
        "status": "running",
        "started_utc": started,
        "script_sha256": script_sha256(),
        "argv": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        },
        "protocol": {
            "grid": args.grid, "hets": args.hets, "model_seeds": args.seeds,
            "pool": args.pool, "test_per": args.test_per, "iters": args.iters,
            "batch": args.batch, "seg": args.seg, "lr": args.lr, "fixed_cap": args.cap,
            "theta": args.theta, "tail_q": args.tail_q, "walls": args.walls,
            "noise": args.noise, "train_scene_seed": args.train_scene_seed,
            "val_scene_seed": args.val_scene_seed, "paired_scenes_across_het": True,
            "control_init_compatible_with_E_two_channel": True,
            "primary_population": "Dgeo>2 and target not clamped",
            "tail_definition": "q-quantile of per-cell squared error, not worst-q-tail mean",
        },
        "levels": [],
    }
    print(f"Branch L clean | device={device} | cap={args.cap} | hets={args.hets} | seeds={args.seeds}")

    for het in args.hets:
        inp, tgt, lm, dgeo, dfree, deuc, hstar, tau, sweeps = pools[het]
        vinp, vtgt, vlm, vdgeo, vdfree, vdeuc, vhstar, vtau, vsweeps = vpools[het]
        var = tgt[lm > 0].var().item() + 1e-12
        vvar = vtgt[vlm > 0].var().item() + 1e-12
        clamped = ((((vtgt >= 1.0 - 1e-6).float() * vlm).sum(dim=(1, 2, 3))) > 0).cpu()
        eligible = ((vdgeo > 2).cpu() & ~clamped)
        level = {
            "het": het,
            "mean_Dgeo": vdgeo.mean().item(),
            "mean_Dfree": vdfree.mean().item(),
            "mean_Deuc": vdeuc.mean().item(),
            "mean_Hstar": vhstar.mean().item(),
            "mean_Hstar_over_Dgeo": (vhstar / vdgeo.clamp(min=1)).mean().item(),
            "bf_sweeps_train": sweeps,
            "bf_sweeps_validation": vsweeps,
            "clamped_scene_ids": clamped.nonzero(as_tuple=False).flatten().tolist(),
            "eligible_scene_ids": eligible.nonzero(as_tuple=False).flatten().tolist(),
            "scene_geometry": [
                {"scene_id": i, "Dgeo": float(vdgeo[i]), "Dfree": float(vdfree[i]),
                 "Deuc": float(vdeuc[i]), "Hstar": float(vhstar[i]), "tau": float(vtau[i]),
                 "clamped": bool(clamped[i]), "eligible": bool(eligible[i])}
                for i in range(len(vinp))
            ],
            "runs": [],
        }
        result["levels"].append(level)
        print(f"het={het:.1f}: Dgeo mean={level['mean_Dgeo']:.2f}; "
              f"eligible={int(eligible.sum())}/{len(eligible)}; clamped={int(clamped.sum())}")

        for seed in args.seeds:
            t0 = time.time()
            model = compatible_nca3(seed, device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            for _ in range(args.iters):
                idx = torch.randint(0, len(inp), (args.batch,), device=device)
                horizon = int(torch.randint(4, args.cap + 1, (1,), device=device).item())
                pred = model.run_train(inp[idx], horizon, seg=args.seg)
                loss = BF.masked_nmse(pred, tgt[idx], lm[idx], var)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            weight_path = f"{args.out}.het{het:.1f}.s{seed}.pt"
            torch.save({
                "schema": SCHEMA_VERSION, "script_sha256": result["script_sha256"],
                "het": het, "seed": seed, "protocol": result["protocol"],
                "state_dict": model.state_dict(),
            }, weight_path)
            ev = evaluate(model, vinp, vtgt, vlm, vvar, args, eligible)
            summary = summarize_seed(ev, vdgeo, vdfree, vdeuc, eligible)
            run = {
                "model_seed": seed,
                "weight_path": weight_path,
                "elapsed_seconds": time.time() - t0,
                **summary,
                **ev,
            }
            level["runs"].append(run)
            cov = summary["tail_unconditional_coverage"]
            cv_geo = summary["tail_ratio_cv_conditional"]["geo"]
            print(f"  seed={seed}: aggregate mean knee={ev['aggregate_mean_knee']} "
                  f"tail knee={ev['aggregate_tail_knee']} coverage={cov:.3f} "
                  f"conditional geo CV={cv_geo if cv_geo is not None else 'n/a'} "
                  f"[{time.time()-t0:.0f}s]")
            write_json(partial, result)

    result["status"] = "complete"
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out, result)
    if partial.exists():
        partial.unlink()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

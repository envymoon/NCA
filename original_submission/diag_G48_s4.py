"""Evaluation-only diagnostic for the G=48 seed-4 aggregate non-convergence
(2026-07-18, run after the ACT baseline; no training, no new claims).

CONTEXT. Run A (ext_G48.py) recorded seed 4 as aggregate NON-CONVERGENT:
eval-pool floor 0.0583 vs theta = 0.05 (a 17% miss), while seed 3 converged
(T* = 49, c1_geo = 0.92). The working hypothesis is threshold-adjacent
training variance, not a code / cap / pool defect. This script tests exactly
that and nothing else:

  1. Rebuilds both pools bit-exact (train gen seed 0 / eval gen seed 1, CUDA
     generator, walls=(1,8)) and re-evaluates BOTH saved checkpoints. Seed 3
     must reproduce its run-A readout (agg knee 49, floor 0.0213) exactly;
     any mismatch would indicate a checkpoint / pool / eval inconsistency and
     would be the only finding that could justify a re-train.
  2. Saves the FULL aggregate curves (full-pool and eligible-only) plus the
     raw minimum and its location, so "how close and where" is on record.
  3. Computes seed 4's per-scene mean/tail knees, censoring counts and ruler
     CVs -- the quantities ext_G48.py never computed because it (correctly)
     stopped at the aggregate gate fixed before launch. Diagnostic only: these
     numbers do NOT enter that readout.
  4. Compares train-pool vs eval-pool floors for both seeds (same-distribution
     pools; a large gap would suggest pool pathology, a small one training
     variance).
  5. Extended horizon T <= 304 (2x cap) for seed 4, eval pool, diagnostic
     only: does the curve cross theta late, or is 0.058 a true floor?

DIAGNOSTIC ONLY. Nothing here changes the run-A conclusion fixed before launch
(1/2 seeds attained the frozen threshold). WRITES ONLY diag_G48_s4.json and
its log; both go to ADDENDUM_MANIFEST.sha256 afterwards.

Run:  cd experiments && python diag_G48_s4.py
"""
import json
import statistics as st
import time

import torch

import branch_F_tortuosity as BF

G = 48
CAP = 152          # frozen
EXT_CAP = 304      # diagnostic-only horizon, 2x frozen cap
THETA = 0.05
TAIL_Q = 0.95
CKPT = "results_E_ext_G48.json.G48.s{s}.pt"
OUT = "diag_G48_s4.json"
RUN_A = "results_E_ext_G48.json"


def cv(xs):
    return st.pstdev(xs) / st.mean(xs) if xs and st.mean(xs) else float("nan")


def eval_pool(model, pinp, ptgt, plm, var, cap, chunk=32, per_scene=False):
    """Aggregate NMSE curve over a pool; optionally per-scene mean/tail curves."""
    Tlist = list(range(1, cap + 1))
    curve = {t: 0.0 for t in Tlist}
    ps = {t: [] for t in Tlist} if per_scene else None
    pst = {t: [] for t in Tlist} if per_scene else None
    N = pinp.shape[0]
    with torch.no_grad():
        for s in range(0, N, chunk):
            sl = slice(s, min(N, s + chunk))
            outs = model.run(pinp[sl], cap, collect=set(Tlist))
            for t in Tlist:
                num = (((outs[t] - ptgt[sl]) ** 2) * plm[sl]).sum(dim=(1, 2, 3))
                den = plm[sl].sum(dim=(1, 2, 3)).clamp(min=1)
                e = (num / den / var).cpu()
                curve[t] += e.sum().item()
                if per_scene:
                    ps[t].append(e)
                    pst[t].append(BF.tail_nmse(outs[t], ptgt[sl], plm[sl], var,
                                               q=TAIL_Q).cpu())
    curve = {t: v / N for t, v in curve.items()}
    if per_scene:
        ps = {t: torch.cat(ps[t]) for t in Tlist}
        pst = {t: torch.cat(pst[t]) for t in Tlist}
    return curve, ps, pst


def curve_stats(curve, theta=THETA):
    knee, floor = BF.abs_knee(curve, theta=theta)
    tmin = min(curve, key=curve.get)
    return {"knee": knee, "floor": floor,
            "raw_min": curve[tmin], "argmin_T": tmin}


def main():
    dev = BF.get_device("cuda")
    t0 = time.time()
    inp, tgt, lm, Dgeo, _, _, _ = BF.build_pool_maze(
        512, G, G, dev, torch.Generator(device=dev).manual_seed(0), walls=(1, 8))
    var = tgt[lm > 0].var().item() + 1e-12
    vinp, vtgt, vlm, vDgeo, vDfree, vDeuc, vtau = BF.build_pool_maze(
        256, G, G, dev, torch.Generator(device=dev).manual_seed(1), walls=(1, 8))
    vvar = vtgt[vlm > 0].var().item() + 1e-12
    clamped = set(i for i in range(256) if float(vDgeo[i]) > 2 * G)
    elig_idx = torch.tensor([i for i in range(256)
                             if i not in clamped and float(vDgeo[i]) > 2],
                            dtype=torch.long)
    run_a = json.load(open(RUN_A))
    res = {"purpose": "evaluation-only diagnostic of G=48 seed-4 aggregate "
           "non-convergence; does not modify the run-A readout fixed before launch",
           "theta": THETA, "cap_frozen": CAP, "ext_cap_diag": EXT_CAP,
           "run_A_reference": {r["seed"]: {"agg_knee": r["agg_knee"],
                                           "floor": r["floor"]}
                               for r in run_a["per_seed"]},
           "seeds": {}}
    print(f"diag G=48  dev={dev}  pools rebuilt [{time.time()-t0:.0f}s]")

    for sd in (3, 4):
        model = BF.NCA().to(dev)
        model.load_state_dict(torch.load(CKPT.format(s=sd), map_location=dev))
        model.eval()
        r = {}

        # eval pool, frozen cap, full + eligible aggregates, per-scene curves
        curve, ps, pst = eval_pool(model, vinp, vtgt, vlm, vvar, CAP,
                                   per_scene=True)
        r["eval_agg"] = curve_stats(curve)
        r["eval_agg"]["curve"] = [curve[t] for t in range(1, CAP + 1)]
        curve_el = {t: ps[t][elig_idx].mean().item() for t in range(1, CAP + 1)}
        r["eval_agg_eligible"] = curve_stats(curve_el)
        ref = res["run_A_reference"].get(sd) or res["run_A_reference"].get(str(sd))
        r["floor_delta_vs_run_A"] = r["eval_agg"]["floor"] - ref["floor"]
        r["reproduces_run_A"] = (r["eval_agg"]["knee"] == ref["agg_knee"]
                                 and abs(r["floor_delta_vs_run_A"]) < 1e-6)
        print(f"  seed {sd} eval-pool: knee={r['eval_agg']['knee']}  "
              f"floor={r['eval_agg']['floor']:.4f}  "
              f"raw_min={r['eval_agg']['raw_min']:.4f}@T={r['eval_agg']['argmin_T']}  "
              f"reproduces run A: {r['reproduces_run_A']}  [{time.time()-t0:.0f}s]")

        # per-scene knees / censoring / ruler CVs (diagnostic for seed 4;
        # cross-check against run A for seed 3)
        sk, skt, sg, sf, se = [], [], [], [], []
        n_cens_mean = n_cens_tail = 0
        for i in range(256):
            if i in clamped or float(vDgeo[i]) <= 2:
                continue
            ki, _ = BF.abs_knee({t: ps[t][i].item() for t in range(1, CAP + 1)},
                                theta=THETA)
            kt, _ = BF.abs_knee({t: pst[t][i].item() for t in range(1, CAP + 1)},
                                theta=THETA)
            n_cens_mean += ki is None
            n_cens_tail += kt is None
            if ki is None or kt is None:
                continue
            sk.append(float(ki)); skt.append(float(kt))
            sg.append(float(vDgeo[i])); sf.append(max(float(vDfree[i]), 1.0))
            se.append(max(float(vDeuc[i]), 1.0))
        n_elig = int(len(elig_idx))
        r["per_scene"] = {
            "n_eligible": n_elig, "n_converged_both": len(sk),
            "n_censored_meanknee": n_cens_mean, "n_censored_tailknee": n_cens_tail,
            "ceiling_uncond": len(sk) / max(n_elig, 1),
            "c1_geo_scene_mean": st.mean([k / d for k, d in zip(sk, sg)])
                                 if sk else None,
            "cv_mean": {"geo": cv([k / d for k, d in zip(sk, sg)]),
                        "free": cv([k / d for k, d in zip(sk, sf)]),
                        "euc": cv([k / d for k, d in zip(sk, se)])},
            "cv_tail": {"geo": cv([k / d for k, d in zip(skt, sg)]),
                        "free": cv([k / d for k, d in zip(skt, sf)]),
                        "euc": cv([k / d for k, d in zip(skt, se)])},
        }
        p = r["per_scene"]
        print(f"    per-scene: converged {len(sk)}/{n_elig}  censored mean/tail "
              f"{n_cens_mean}/{n_cens_tail}  CV tail geo/free/euc "
              f"{100*p['cv_tail']['geo']:.0f}/{100*p['cv_tail']['free']:.0f}/"
              f"{100*p['cv_tail']['euc']:.0f}%")

        # train-pool floor (same distribution; large eval-train gap would
        # suggest pool pathology rather than training variance)
        curve_tr, _, _ = eval_pool(model, inp, tgt, lm, var, CAP, chunk=16)
        r["train_agg"] = curve_stats(curve_tr)
        print(f"    train-pool: knee={r['train_agg']['knee']}  "
              f"floor={r['train_agg']['floor']:.4f}  "
              f"raw_min={r['train_agg']['raw_min']:.4f}"
              f"@T={r['train_agg']['argmin_T']}  [{time.time()-t0:.0f}s]")

        # extended horizon, seed 4 only: late crossing vs true floor
        if sd == 4:
            curve_ext, _, _ = eval_pool(model, vinp, vtgt, vlm, vvar, EXT_CAP,
                                        chunk=16)
            r["eval_agg_ext"] = curve_stats(curve_ext)
            r["eval_agg_ext"]["curve"] = [curve_ext[t]
                                          for t in range(1, EXT_CAP + 1)]
            r["eval_agg_ext"]["note"] = ("diagnostic-only horizon 2x frozen cap; "
                                         "not part of the launch-fixed readout")
            e = r["eval_agg_ext"]
            print(f"    extended horizon T<={EXT_CAP}: knee={e['knee']}  "
                  f"floor={e['floor']:.4f}  raw_min={e['raw_min']:.4f}"
                  f"@T={e['argmin_T']}  [{time.time()-t0:.0f}s]")

        res["seeds"][sd] = r

    s3 = res["seeds"][3]
    verdict = ("consistent: seed 3 reproduces run A bit-exact; no evidence of "
               "checkpoint/pool/eval inconsistency; seed-4 miss reads as "
               "threshold-adjacent training variance"
               if s3["reproduces_run_A"] else
               "INCONSISTENCY: seed 3 failed to reproduce run A -- investigate "
               "checkpoint/pool/eval before any further use of run-A numbers")
    res["verdict"] = verdict
    print(f"\n  DIAG VERDICT: {verdict}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"saved -> {OUT}  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()

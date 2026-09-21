"""G=48 sufficiency extension (reviewer-gap run A, user-approved 2026-07-17).

POST-FREEZE ADDENDUM. Trains NEW models at G=48 under the EXACT frozen branch-E
protocol and asks one question fixed before launch: does c1_geo = T*/D_geo stay flat when
the grid grows past the frozen range (16/24/32 -> 48)? Frozen anchors (run_E_s34.log +
merged 5-seed): c1_geo = 0.98 / 0.95 / 0.91 at G=16/24/32. Claim 1 (a learned NCA
tracks the D_geo propagation scale) predicts G=48 lands in the same band; a
systematic rise would read as D_geo*log D_geo-type growth and would WEAKEN claim 1.
Both outcomes get reported.

PROTOCOL FIXED BEFORE LAUNCH (all values chosen before any G=48 data existed):
  cap        = 152   -- the FROZEN global anytime cap. Deliberately NOT recomputed from
                        the G=48 pool: recomputing would make the horizon a function of
                        this grid's own distances, which is the horizon confound the
                        branch-E design exists to remove. 152 was fixed by the frozen
                        grids, so for G=48 it is distance-independent by construction.
                        The price is thinner headroom (eligible Dgeo <= 2G = 96, so
                        headroom 1.58x vs 2.1x at G=32) -> MORE cap-censoring in the
                        tortuous tail. Censoring is counted and reported, never imputed.
  theta=0.05, persist=10, win=5, tail_q=0.95, walls=(1,8), pool 512 (gen seed 0),
  eval pool 256 (gen seed 1), anytime T~U[4,cap], iters 20000, batch 12, seg 24,
  lr 1e-3, grad-clip 1.0, seeds 3 4 (the two frozen-checkpoint seeds).
  Eligibility: Dgeo <= 2G (same "clamped" rule as full_population_audit.py).

DUAL-POPULATION REPORTING. The aggregate curve/knee is computed twice: on the FULL
256-scene eval pool (the frozen branch-E aggregate population, so c1_geo_agg is
directly comparable to the 0.98/0.95/0.91 anchors) AND on the unclamped eligible
subset (agg_knee_eligible / c1_geo_agg_eligible), so the aggregate and per-scene
populations reconcile. Convergence and censoring counts are reported
unconditionally; heavy tail censoring is a RESULT, not a nuisance to exclude.

WRITES ONLY: results_E_ext_G48.json (+ .partial), per-seed .pt checkpoints, run log.
Frozen artifacts untouched; new files go to ADDENDUM_MANIFEST.sha256.

Run:  cd experiments && python ext_G48.py
"""
import argparse
import json
import statistics as st
import time

import torch

import branch_F_tortuosity as BF

CAP_FROZEN = 152


def cv(xs):
    return st.pstdev(xs) / st.mean(xs) if xs and st.mean(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--epool", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--cap", type=int, default=CAP_FROZEN)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--theta", type=float, default=0.05)
    ap.add_argument("--tail_q", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--out", default="results_E_ext_G48.json")
    a = ap.parse_args()
    dev = BF.get_device("cuda")
    G = a.grid
    cap = a.cap

    inp, tgt, lm, Dgeo, Dfree, Deuc, tau = BF.build_pool_maze(
        a.pool, G, G, dev, torch.Generator(device=dev).manual_seed(0), walls=(1, 8))
    var = tgt[lm > 0].var().item() + 1e-12
    vinp, vtgt, vlm, vDgeo, vDfree, vDeuc, vtau = BF.build_pool_maze(
        a.epool, G, G, dev, torch.Generator(device=dev).manual_seed(1), walls=(1, 8))
    vvar = vtgt[vlm > 0].var().item() + 1e-12
    mg = vDgeo.mean().item()
    clamped = [i for i in range(a.epool) if float(vDgeo[i]) > 2 * G]
    elig_idx = torch.tensor([i for i in range(a.epool)
                             if i not in clamped and float(vDgeo[i]) > 2],
                            dtype=torch.long)
    mg_el = vDgeo[elig_idx].mean().item() if len(elig_idx) else float("nan")
    Tlist = list(range(1, cap + 1))
    res = {"grid": G, "cap": cap, "cap_provenance": "frozen global cap from run_E_s34 "
           "(fixed before launch; NOT recomputed from this grid)", "theta": a.theta,
           "tail_q": a.tail_q, "iters": a.iters, "seeds": a.seeds,
           "eval_pool_gen_seed": 1, "n_clamped_eval": len(clamped),
           "population_note": "agg_knee/c1_geo_agg = full eval pool (frozen branch-E "
           "population, anchor-comparable); *_eligible = unclamped scenes with Dgeo>2",
           "frozen_c1_geo_anchor": {"16": 0.98, "24": 0.95, "32": 0.91},
           "per_seed": [], "scenes": []}
    print(f"G=48 extension  dev={dev}  cap={cap} (frozen)  theta={a.theta}  "
          f"iters={a.iters}  seeds={a.seeds}")
    print(f"  eval pool: D_geo mean={mg:.1f}  max={vDgeo.max():.0f}  "
          f"clamped(Dgeo>2G={2*G}): {len(clamped)}/{a.epool}  "
          f"tau p50={torch.quantile(vtau, .5):.2f} p90={torch.quantile(vtau, .9):.2f}")

    for sd in a.seeds:
        BF.set_seed(sd)
        model = BF.NCA().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        t0 = time.time()
        npool = inp.shape[0]
        for it in range(a.iters):
            idx = torch.randint(0, npool, (a.batch,), device=dev)
            T = int(torch.randint(4, cap + 1, (1,)).item())
            out = model.run_train(inp[idx], T, seg=a.seg)
            loss = BF.masked_nmse(out, tgt[idx], lm[idx], var)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if (it + 1) % 2000 == 0:
                print(f"    seed={sd} iter {it+1}/{a.iters}  loss={loss.item():.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        torch.save(model.state_dict(), f"{a.out}.G{G}.s{sd}.pt")

        # frozen eval machinery: pool-mean curve + per-scene mean/tail curves
        with torch.no_grad():
            curve = {t: 0.0 for t in Tlist}
            ps = {t: [] for t in Tlist}
            pst = {t: [] for t in Tlist}
            Nv = vinp.shape[0]
            for s in range(0, Nv, 32):
                sl = slice(s, min(Nv, s + 32))
                outs = model.run(vinp[sl], cap, collect=set(Tlist))
                for t in Tlist:
                    num = (((outs[t] - vtgt[sl]) ** 2) * vlm[sl]).sum(dim=(1, 2, 3))
                    den = vlm[sl].sum(dim=(1, 2, 3)).clamp(min=1)
                    e = (num / den / vvar).cpu()
                    ps[t].append(e); curve[t] += e.sum().item()
                    pst[t].append(BF.tail_nmse(outs[t], vtgt[sl], vlm[sl], vvar,
                                               q=a.tail_q).cpu())
            curve = {t: v / Nv for t, v in curve.items()}
            ps = {t: torch.cat(ps[t]) for t in Tlist}
            pst = {t: torch.cat(pst[t]) for t in Tlist}

        knee, floor = BF.abs_knee(curve, theta=a.theta)
        # eligible-only aggregate (per-scene population), alongside the anchor-
        # comparable full-pool aggregate above
        curve_el = {t: ps[t][elig_idx].mean().item() for t in Tlist}
        knee_el, _ = BF.abs_knee(curve_el, theta=a.theta)
        c1_el = (knee_el / mg_el if knee_el is not None and knee_el < cap else None)
        rec = {"seed": sd, "agg_knee": knee, "floor": floor,
               "agg_knee_eligible": knee_el, "mean_Dgeo_eligible": mg_el,
               "c1_geo_agg_eligible": c1_el, "train_s": time.time() - t0}
        if knee is None or knee >= cap:
            why = "never reached theta" if knee is None else "knee at cap (censored)"
            rec["status"] = f"aggregate NON-CONVERGENT: {why}"
            print(f"  G={G} seed={sd}  {rec['status']}  floor={floor:.4f}  "
                  f"[{time.time()-t0:.0f}s]")
            res["per_seed"].append(rec)
            json.dump(res, open(a.out + ".partial", "w"), indent=1)
            continue

        sk, skt, sg, sf, se, tv, pidx = [], [], [], [], [], [], []
        n_cens_mean = n_cens_tail = 0
        for i in range(Nv):
            if i in clamped or float(vDgeo[i]) <= 2:
                continue
            ki, _ = BF.abs_knee({t: ps[t][i].item() for t in Tlist}, theta=a.theta)
            kt, _ = BF.abs_knee({t: pst[t][i].item() for t in Tlist}, theta=a.theta)
            n_cens_mean += ki is None
            n_cens_tail += kt is None
            if ki is None or kt is None:
                continue
            sk.append(float(ki)); skt.append(float(kt))
            sg.append(float(vDgeo[i])); sf.append(max(float(vDfree[i]), 1.0))
            se.append(max(float(vDeuc[i]), 1.0)); tv.append(float(vtau[i]))
            pidx.append(i)
        n_elig = Nv - len(clamped) - sum(1 for i in range(Nv)
                                         if i not in clamped and float(vDgeo[i]) <= 2)
        c1g = [k / d for k, d in zip(sk, sg)]
        c1gt = [k / d for k, d in zip(skt, sg)]
        rec.update({
            "status": "ok", "T_star": knee, "c1_geo_agg": knee / mg,
            "n_eligible": n_elig, "n_converged_both": len(sk),
            "n_censored_meanknee": n_cens_mean, "n_censored_tailknee": n_cens_tail,
            "ceiling_uncond": len(sk) / max(n_elig, 1),
            "c1_geo_scene_mean": st.mean(c1g), "c1_geo_tail_scene_mean": st.mean(c1gt),
            "cv_mean": {"geo": cv(c1g), "free": cv([k / d for k, d in zip(sk, sf)]),
                        "euc": cv([k / d for k, d in zip(sk, se)])},
            "cv_tail": {"geo": cv(c1gt), "free": cv([k / d for k, d in zip(skt, sf)]),
                        "euc": cv([k / d for k, d in zip(skt, se)])},
        })
        res["per_seed"].append(rec)
        res["scenes"].append({"seed": sd, "pool_idx": pidx, "knee": sk, "knee_tail": skt,
                              "dgeo": sg, "dfree": sf, "deuc": se, "tau": tv})
        cm, ct = rec["cv_mean"], rec["cv_tail"]
        print(f"  G={G} seed={sd}  T*={knee:3d}(<cap)  c1_geo={knee/mg:.2f}  "
              f"floor={floor:.4f}  scenes={len(sk)}/{n_elig}  "
              f"CV mean-knee geo/free/euc {100*cm['geo']:.0f}/{100*cm['free']:.0f}/"
              f"{100*cm['euc']:.0f}%  [{time.time()-t0:.0f}s]")
        print(f"       tail knee: c1_geo={st.mean(c1gt):.2f}  CV geo/free/euc "
              f"{100*ct['geo']:.0f}/{100*ct['free']:.0f}/{100*ct['euc']:.0f}%  "
              f"censored mean/tail {n_cens_mean}/{n_cens_tail} of {n_elig}")
        json.dump(res, open(a.out + ".partial", "w"), indent=1)

    ok = [r for r in res["per_seed"] if r.get("status") == "ok"]
    if ok:
        cg = [r["c1_geo_agg"] for r in ok]
        res["c1_geo_G48"] = BF.agg(cg)
        res["sufficiency_readout"] = (
            f"c1_geo(G=48) = {st.mean(cg):.2f} vs frozen 0.98/0.95/0.91 at 16/24/32; "
            "flat-or-below supports claim 1 across a 3x grid span; a rise would weaken it")
        print(f"\n  SUFFICIENCY: c1_geo(G=48) = {st.mean(cg):.2f} "
              f"({[f'{v:.2f}' for v in cg]})  vs frozen anchors 0.98/0.95/0.91")
        cge = [r["c1_geo_agg_eligible"] for r in ok]
        if all(v is not None for v in cge):
            res["c1_geo_G48_eligible"] = BF.agg(cge)
            print(f"  eligible-only aggregate: c1_geo = {st.mean(cge):.2f} "
                  f"({[f'{v:.2f}' for v in cge]})")
        else:
            res["c1_geo_G48_eligible"] = None
            print("  eligible-only aggregate: censored/non-convergent in >=1 seed "
                  "(reported, not imputed)")
        geo_wins = all(r["cv_tail"]["geo"] < min(r["cv_tail"]["free"],
                                                 r["cv_tail"]["euc"]) for r in ok)
        # `ok` holds convergent seeds only; non-convergent seeds have no per-scene CVs
        res["ruler_geo_tightest_tailknee_convergent_seeds_only"] = geo_wins
        res["ruler_n_convergent_seeds"] = len(ok)
        print(f"  RULER at G=48 (tail knee): geo CV tightest in every CONVERGENT seed "
              f"({len(ok)} of {len(res['per_seed'])}): {geo_wins}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

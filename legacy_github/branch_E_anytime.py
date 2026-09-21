import argparse, json, time
import statistics as st
import torch
import branch_F_tortuosity as BF

INF = BF.INF


def cv(xs):
    return st.pstdev(xs) / st.mean(xs) if xs and st.mean(xs) else float("nan")


def tau_bins(tau_vals, knees, dgeo, dfree, deuc, nbin=4):
    if len(tau_vals) < nbin * 2:
        return []
    order = sorted(range(len(tau_vals)), key=lambda i: tau_vals[i])
    out, per = [], len(order) // nbin
    for b in range(nbin):
        idx = order[b * per:(b + 1) * per] if b < nbin - 1 else order[b * per:]
        if not idx:
            continue
        out.append({
            "tau_lo": tau_vals[idx[0]], "tau_hi": tau_vals[idx[-1]], "n": len(idx),
            "tau_mean": st.mean([tau_vals[i] for i in idx]),
            "c1_geo": st.mean([knees[i] / dgeo[i] for i in idx]),
            "c1_free": st.mean([knees[i] / dfree[i] for i in idx]),
            "c1_euc": st.mean([knees[i] / deuc[i] for i in idx]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--cap_factor", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--theta", type=float, default=0.05,
                    help="ABSOLUTE masked-NMSE target defining the knee; one value for every "
                         "grid and seed. A run that never reaches it is non-convergent and is "
                         "reported as such, not handed a knee.")
    ap.add_argument("--walls", type=int, nargs=2, default=[1, 8],
                    help="per-scene wall count is drawn from this range; it is the diversity "
                         "knob that spreads tau, NOT a controlled level")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tail_q", type=float, default=0.95,
                    help="second knee definition, recorded alongside the mean one at no extra "
                         "training cost: the per-scene q-quantile of per-cell error. D_geo is a "
                         "MAX eccentricity, so the mean knee measures a different event and "
                         "lands below D_geo; this is the like-for-like reading.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results_E_maze.json")
    a = ap.parse_args()
    dev = BF.get_device(a.device)
    res = {"connectivity": 8, "anytime": True, "seeds": a.seeds, "scene_family": "randomised maze",
           "walls": a.walls, "knee": f"absolute NMSE<={a.theta}, sustained", "theta": a.theta,
           "iters": a.iters, "per_grid": {}}
    print(f"Branch E (maze family)  dev={dev}  seeds={a.seeds}  theta={a.theta}  iters={a.iters}")

    pools, vpools = {}, {}
    for G in a.grids:
        pools[G] = BF.build_pool_maze(a.pool, G, G, dev, torch.Generator(device=dev).manual_seed(0),
                                      walls=tuple(a.walls))
        vpools[G] = BF.build_pool_maze(256, G, G, dev, torch.Generator(device=dev).manual_seed(1),
                                       walls=tuple(a.walls))
    cap = int(a.cap_factor * max(pools[G][3].max().item() for G in a.grids)) + 8
    res["global_cap"] = cap
    print(f"  GLOBAL anytime cap = {cap} (shared by all grids; distance-independent)")
    for G in a.grids:
        t = vpools[G][6]
        print(f"  G={G:2d}: D_geo mean={vpools[G][3].mean():5.1f}  "
              f"tau p10={torch.quantile(t, .1):.2f} p50={torch.quantile(t, .5):.2f} "
              f"p90={torch.quantile(t, .9):.2f}")

    for G in a.grids:
        inp, tgt, lm, Dgeo, Dfree, Deuc, tau = pools[G]
        var = tgt[lm > 0].var().item() + 1e-12
        vinp, vtgt, vlm, vDgeo, vDfree, vDeuc, vtau = vpools[G]
        vvar = vtgt[vlm > 0].var().item() + 1e-12
        mg, mf, me = vDgeo.mean().item(), vDfree.mean().item(), vDeuc.mean().item()
        Tlist = list(range(1, cap + 1))
        npool = inp.shape[0]

        knees, c1g_a, c1f_a, c1e_a = [], [], [], []
        cvg_a, cvf_a, cve_a, floors, bins_a = [], [], [], [], []
        ok_seeds = []
        scenes_a = []
        cvgt_a, cvft_a, cvet_a, c1gt_a, binst_a = [], [], [], [], []
        for sd in a.seeds:
            BF.set_seed(sd)
            model = BF.NCA().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            t0 = time.time()
            for _ in range(a.iters):
                idx = torch.randint(0, npool, (a.batch,), device=dev)
                T = int(torch.randint(4, cap + 1, (1,)).item())
                out = model.run_train(inp[idx], T, seg=a.seg)
                loss = BF.masked_nmse(out, tgt[idx], lm[idx], var)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


            torch.save(model.state_dict(), f"{a.out}.G{G}.s{sd}.pt")


            with torch.no_grad():
                curve = {t: 0.0 for t in Tlist}
                ps, pst = {t: [] for t in Tlist}, {t: [] for t in Tlist}
                Nv = vinp.shape[0]
                for s in range(0, Nv, 64):
                    sl = slice(s, min(Nv, s + 64))
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
            if knee is None:
                print(f"  G={G:2d} seed={sd}  NON-CONVERGENT: never reached NMSE<={a.theta} "
                      f"within cap (floor={floor:.4f}); excluded  [{time.time()-t0:.0f}s]")
                continue
            if knee >= cap:
                print(f"  G={G:2d} seed={sd}  INVALID: knee at the cap; horizon-confounded; "
                      f"excluded  [{time.time()-t0:.0f}s]")
                continue


            sk, sg, sf, se, tv = [], [], [], [], []
            skt = []
            for i in range(vinp.shape[0]):
                ki, _ = BF.abs_knee({t: ps[t][i].item() for t in Tlist}, theta=a.theta)
                kt, _ = BF.abs_knee({t: pst[t][i].item() for t in Tlist}, theta=a.theta)
                if ki is None or kt is None or vDgeo[i] <= 2:
                    continue

                sk.append(float(ki)); skt.append(float(kt)); sg.append(vDgeo[i].item())
                sf.append(max(vDfree[i].item(), 1.0)); se.append(max(vDeuc[i].item(), 1.0))
                tv.append(vtau[i].item())
            if len(sk) < 32:
                print(f"  G={G:2d} seed={sd}  only {len(sk)} scenes converged; excluded "
                      f"[{time.time()-t0:.0f}s]")
                continue
            c1g = [k / d for k, d in zip(sk, sg)]
            c1f = [k / d for k, d in zip(sk, sf)]
            c1e = [k / d for k, d in zip(sk, se)]

            knees.append(knee); floors.append(floor)
            c1g_a.append(knee / mg); c1f_a.append(knee / mf); c1e_a.append(knee / me)
            cvg_a.append(cv(c1g)); cvf_a.append(cv(c1f)); cve_a.append(cv(c1e))
            bins_a.append(tau_bins(tv, sk, sg, sf, se))
            ok_seeds.append(sd)


            scenes_a.append({"seed": sd, "knee": sk, "knee_tail": skt,
                             "dgeo": sg, "dfree": sf, "deuc": se, "tau": tv})


            c1gt = [k / d for k, d in zip(skt, sg)]
            c1ft = [k / d for k, d in zip(skt, sf)]
            c1et = [k / d for k, d in zip(skt, se)]
            cvgt_a.append(cv(c1gt)); cvft_a.append(cv(c1ft)); cvet_a.append(cv(c1et))
            c1gt_a.append(st.mean(c1gt))
            binst_a.append(tau_bins(tv, skt, sg, sf, se))
            print(f"  G={G:2d} seed={sd}  T*={knee:3d}(<cap)  c1_geo={knee/mg:.2f}  "
                  f"floor={floor:.4f}  scenes={len(c1g)}/{vinp.shape[0]}  "
                  f"per-scene CV: geo {cv(c1g)*100:3.0f}% free {cv(c1f)*100:3.0f}% "
                  f"euc {cv(c1e)*100:3.0f}%  [{time.time()-t0:.0f}s]")
            print(f"       tail knee (p{a.tail_q*100:.0f}), the like-for-like read against a MAX "
                  f"ruler:  c1_geo={st.mean(c1gt):.2f}  "
                  f"CV: geo {cv(c1gt)*100:3.0f}% free {cv(c1ft)*100:3.0f}% "
                  f"euc {cv(c1et)*100:3.0f}%")

        if not knees:
            print(f"  G={G:2d}: ALL seeds non-convergent — no knee reported for this grid")
            continue
        res["per_grid"][str(G)] = {
            "mean_Dgeo": mg, "mean_Dfree": mf, "mean_Deuc": me, "ok_seeds": ok_seeds, "cap": cap,
            "T_knee": BF.agg(knees), "floor_nmse": BF.agg(floors),
            "c1_geo": BF.agg(c1g_a), "c1_free": BF.agg(c1f_a), "c1_euc": BF.agg(c1e_a),
            "c1_geo_cv": BF.agg(cvg_a), "c1_free_cv": BF.agg(cvf_a), "c1_euc_cv": BF.agg(cve_a),
            "tau_bins": [],
            "tail_q": a.tail_q,
            "c1_geo_tail": BF.agg(c1gt_a), "c1_geo_cv_tail": BF.agg(cvgt_a),
            "c1_free_cv_tail": BF.agg(cvft_a), "c1_euc_cv_tail": BF.agg(cvet_a),
            "tau_bins_tail": [],
            "scenes": scenes_a,
        }
        r = res["per_grid"][str(G)]
        print(f"  G={G:2d}  D_geo={mg:5.1f}  T*={r['T_knee']['mean']:.1f}+-{r['T_knee']['std']:.1f}"
              f"  c1_geo={r['c1_geo']['mean']:.2f}+-{r['c1_geo']['std']:.2f}")
        print(f"       per-scene CV:  geo {r['c1_geo_cv']['mean']*100:4.0f}+-{r['c1_geo_cv']['std']*100:.0f}%"
              f"   free {r['c1_free_cv']['mean']*100:4.0f}+-{r['c1_free_cv']['std']*100:.0f}%"
              f"   euc {r['c1_euc_cv']['mean']*100:4.0f}+-{r['c1_euc_cv']['std']*100:.0f}%"
              f"   (geodesic should be TIGHTEST)")


        nb = min(len(b) for b in bins_a)
        if nb:
            r["tau_bins"] = [{
                "tau_mean": st.mean([b[j]["tau_mean"] for b in bins_a]),
                "n": int(st.mean([b[j]["n"] for b in bins_a])),
                "c1_geo": BF.agg([b[j]["c1_geo"] for b in bins_a]),
                "c1_free": BF.agg([b[j]["c1_free"] for b in bins_a]),
                "c1_euc": BF.agg([b[j]["c1_euc"] for b in bins_a]),
            } for j in range(nb)]
            print(f"       tau bin |  tau  |  c1_geo   c1_free   c1_euc   <- c1_geo should stay "
                  f"FLAT across bins; the straight-line rulers should blow up")
            for b in r["tau_bins"]:
                print(f"        n={b['n']:4d}  | {b['tau_mean']:5.2f} | "
                      f"{b['c1_geo']['mean']:6.2f}   {b['c1_free']['mean']:6.2f}   "
                      f"{b['c1_euc']['mean']:6.2f}")


            nbt = min(len(b) for b in binst_a) if all(binst_a) else 0
            if nbt:
                r["tau_bins_tail"] = [{
                    "tau_mean": st.mean([b[j]["tau_mean"] for b in binst_a]),
                    "n": int(st.mean([b[j]["n"] for b in binst_a])),
                    "c1_geo": BF.agg([b[j]["c1_geo"] for b in binst_a]),
                    "c1_free": BF.agg([b[j]["c1_free"] for b in binst_a]),
                    "c1_euc": BF.agg([b[j]["c1_euc"] for b in binst_a]),
                } for j in range(nbt)]
                print(f"       same bins, TAIL knee (p{a.tail_q*100:.0f}) -- the like-for-like "
                      f"reading against a MAX ruler:")
                for b in r["tau_bins_tail"]:
                    print(f"        n={b['n']:4d}  | {b['tau_mean']:5.2f} | "
                          f"{b['c1_geo']['mean']:6.2f}   {b['c1_free']['mean']:6.2f}   "
                          f"{b['c1_euc']['mean']:6.2f}")


        json.dump(res, open(a.out + ".partial", "w"), indent=2)

    Gs = [g for g in a.grids if str(g) in res["per_grid"]]
    if not Gs:
        print("\n  NO VALID GRIDS — refusing to report a law.")
        json.dump(res, open(a.out, "w"), indent=2); return


    common = sorted(set.intersection(*[set(res["per_grid"][str(g)]["ok_seeds"]) for g in Gs]))
    res["common_seeds"] = common
    if common and len(Gs) > 1:
        def val(g, key, sd):
            r = res["per_grid"][str(g)]
            return r[key]["vals"][r["ok_seeds"].index(sd)]
        trend = [[val(g, "c1_geo", sd) for g in Gs] for sd in common]
        res["c1_geo_by_grid"] = {str(g): BF.agg([t[i] for t in trend]) for i, g in enumerate(Gs)}
        res["c1_geo_grid_cv"] = BF.agg([cv(t) for t in trend])
        print(f"\n  SUFFICIENCY (does the quality knee remain proportional to D_geo?)")
        for i, g in enumerate(Gs):
            v = res["c1_geo_by_grid"][str(g)]
            print(f"    G={g:2d}  c1_geo = {v['mean']:.2f}+-{v['std']:.2f}")
        print(f"    within-seed c1 CV across grids = {res['c1_geo_grid_cv']['mean']*100:.0f}"
              f"+-{res['c1_geo_grid_cv']['std']*100:.0f}%   (flat => proportional over the "
              f"tested range; rising => super-linear margin)")

    print(f"\n  RULER: geodesic per-scene CV must be < free/euc CV at every grid.")
    wins = sum(1 for g in Gs
               if res["per_grid"][str(g)]["c1_geo_cv"]["mean"] < res["per_grid"][str(g)]["c1_free_cv"]["mean"]
               and res["per_grid"][str(g)]["c1_geo_cv"]["mean"] < res["per_grid"][str(g)]["c1_euc_cv"]["mean"])
    res["ruler_wins_grids"] = f"{wins}/{len(Gs)}"
    print(f"    geodesic tightest at {wins}/{len(Gs)} grids")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

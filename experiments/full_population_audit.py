"""Full-population policy audit: coverage with NO scene silently dropped.

analyze_policy.py fits and evaluates on the converged rows only; a scene that never
reached theta is absent, so its nominal coverage is conditional. This audit rebuilds the
deterministic evaluation pools (same generator seeds/device as the run), matches every
dumped row back to its pool index, and treats every unmatched (censored) scene as a
SERVED-FAILURE at any budget. It reports:

  1. per grid x seed: total / clamped (Dgeo > 2G) / censored counts;
  2. the V-D policy table recomputed on the full eligible population (unconditional
     coverage; cost averaged over ALL provisioned scenes, censored included);
  3. held-out-scene calibration: constants fitted on even pool indices, coverage
     evaluated on odd pool indices only (addresses in-sample fitting);
  4. one-constant-across-grids and leave-one-seed-out, both unconditional.

All geometric policies execute integer passes, T_i=ceil(c D_i).  Their constants are
stored to two decimal places and fitted on that same discrete policy to match the
unconditional coverage of the flat f-quantile budget.  This keeps fitting, coverage,
and cost accounting identical to deployment.

Run (device must match the training run):
  python3 full_population_audit.py results_E_maze_s01234.json --device cuda
"""
import argparse
import json

import torch

import branch_F_tortuosity as BF
from check_censoring import match_rows
from policy_discrete import eval_policy, fit_c_to_coverage, flat_budget


def build_rows(res, device):
    """One row per (grid, seed, pool scene): knee (None if censored) + true rulers."""
    walls = tuple(res.get("walls", [1, 8]))
    rows = []
    pool_geom = {}
    for g, e in sorted(res["per_grid"].items(), key=lambda kv: int(kv[0])):
        G = int(g)
        _, _, _, Dgeo, Dfree, Deuc, tau = BF.build_pool_maze(
            256, G, G, device, torch.Generator(device=device).manual_seed(1), walls=walls)
        dg = [float(x) for x in Dgeo]
        df = [float(x) for x in Dfree]
        de = [float(x) for x in Deuc]
        tv = [float(x) for x in tau]
        pool_geom[G] = {"dgeo": dg, "clamped": [i for i in range(256) if dg[i] > 2 * G]}
        for blk in e.get("scenes", []):
            kept, misses = match_rows(blk, Dgeo, Dfree, Deuc)
            assert misses == 0, f"G={G} seed={blk['seed']}: {misses} row-match misses"
            # match_rows returns kept pool indices but not the row->index map; rebuild it
            key = {}
            for i in range(256):
                key.setdefault((int(dg[i]), int(df[i]), round(de[i], 3)), []).append(i)
            used = set()
            idx_of_row = []
            for j in range(len(blk["dgeo"])):
                k = (int(blk["dgeo"][j]), int(blk["dfree"][j]), round(blk["deuc"][j], 3))
                i = next(i for i in key.get(k, []) if i not in used)
                used.add(i)
                idx_of_row.append(i)
            knee_tail = {i: blk["knee_tail"][j] for j, i in enumerate(idx_of_row)}
            knee_mean = {i: blk["knee"][j] for j, i in enumerate(idx_of_row)}
            for i in range(256):
                rows.append({
                    "grid": G, "seed": blk["seed"], "pool_idx": i,
                    "dgeo": dg[i], "dfree": df[i], "deuc": de[i], "tau": tv[i],
                    "clamped": dg[i] > 2 * G,
                    "knee_tail": knee_tail.get(i), "knee_mean": knee_mean.get(i),
                })
    return rows, pool_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--knee", choices=["mean", "tail"], default="tail")
    ap.add_argument("--fs", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--out", default="audit_full_population.json")
    ap.add_argument("--reuse-rows", help="CPU-only: reuse rows from an existing audit JSON")
    a = ap.parse_args()

    res = json.load(open(a.json_path))
    knee = "knee_tail" if a.knee == "tail" else "knee_mean"
    if a.reuse_rows:
        prior = json.load(open(a.reuse_rows))
        rows = prior["rows"]
        pool_geom = {G: {"clamped": sorted({r["pool_idx"] for r in rows
                                            if r["grid"] == G and r["clamped"]})}
                     for G in sorted({r["grid"] for r in rows})}
    else:
        rows, pool_geom = build_rows(res, a.device)
    out = {"source": a.json_path, "knee": a.knee, "theta": res.get("theta"),
           "budget_policy": "integer passes: T_i=max(1,ceil(c*D_i)); c fitted on a 0.01 grid",
           "population": "all 256 pool scenes per grid x seed; clamped (Dgeo>2G) excluded "
                         "from the primary population and reported separately",
           "per_unit": [], "policy_full": [], "heldout_scene": [],
           "one_constant": {}, "loso": []}

    print(f"pool clamp counts (Dgeo > 2G): "
          f"{ {G: len(v['clamped']) for G, v in pool_geom.items()} }")
    grids = sorted({r["grid"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    for G in grids:
        for s in seeds:
            u = [r for r in rows if r["grid"] == G and r["seed"] == s]
            el = [r for r in u if not r["clamped"]]
            conv = [r for r in el if r[knee] is not None]
            rec = {"grid": G, "seed": s, "total": len(u),
                   "clamped": sum(r["clamped"] for r in u),
                   "eligible": len(el), "converged": len(conv),
                   "coverage_unconditional": len(conv) / len(el)}
            out["per_unit"].append(rec)
            print(f"G={G:2d} s={s}: eligible {rec['eligible']:3d} "
                  f"converged {rec['converged']:3d} "
                  f"({100*rec['coverage_unconditional']:.1f}%)")

    # primary population: eligible (unclamped) rows, all grids/seeds pooled
    pop = [r for r in rows if not r["clamped"]]
    n_cens = sum(1 for r in pop if r[knee] is None)
    print(f"\nfull population: {len(pop)} rows, {n_cens} censored "
          f"({100*(1-n_cens/len(pop)):.2f}% converged)")

    print(f"\npolicy table, UNCONDITIONAL coverage (censored = unserved), knee={a.knee}:")
    print(f"{'f':>5} | {'flat T':>6} {'cov':>6} | {'geo c':>6} {'cost':>6} {'cov':>6} "
          f"{'save':>5} | {'free cov':>8} | {'euc cov':>8}")
    for f in a.fs:
        Tf = flat_budget(pop, f, knee)
        _, cov_f = eval_policy(pop, "flat", Tf, knee)
        cg = fit_c_to_coverage(pop, "dgeo", cov_f, knee)
        cfr = fit_c_to_coverage(pop, "dfree", cov_f, knee)
        ceu = fit_c_to_coverage(pop, "deuc", cov_f, knee)
        cost_g, cov_g = eval_policy(pop, "dgeo", cg, knee)
        cost_fr, cov_fr = eval_policy(pop, "dfree", cfr, knee)
        cost_eu, cov_eu = eval_policy(pop, "deuc", ceu, knee)
        rec = {"f": f, "flat_T": Tf, "flat_cov": cov_f, "geo_c": cg,
               "geo_cost": cost_g, "geo_cov": cov_g,
               "save_vs_flat": 1 - cost_g / Tf,
               "free_c": cfr, "free_cost": cost_fr, "free_cov": cov_fr,
               "free_save_vs_flat": 1 - cost_fr / Tf,
               "euc_c": ceu, "euc_cost": cost_eu, "euc_cov": cov_eu,
               "euc_save_vs_flat": 1 - cost_eu / Tf}
        out["policy_full"].append(rec)
        print(f"{f:5.2f} | {Tf:6.0f} {100*cov_f:5.1f}% | {cg:6.2f} {cost_g:6.1f} "
              f"{100*cov_g:5.1f}% {100*rec['save_vs_flat']:4.0f}% | "
              f"{cost_fr:6.1f} {100*cov_fr:5.1f}% {100*rec['free_save_vs_flat']:4.0f}% | "
              f"{cost_eu:6.1f} {100*cov_eu:5.1f}% {100*rec['euc_save_vs_flat']:4.0f}%")

    # held-out-scene calibration: fit on even pool indices, evaluate on odd
    print("\nheld-out-scene calibration (fit even pool indices, evaluate odd):")
    fit_pop = [r for r in pop if r["pool_idx"] % 2 == 0]
    hold_pop = [r for r in pop if r["pool_idx"] % 2 == 1]
    for f in a.fs:
        Tf = flat_budget(fit_pop, f, knee)
        _, fit_cov_f = eval_policy(fit_pop, "flat", Tf, knee)
        cg = fit_c_to_coverage(fit_pop, "dgeo", fit_cov_f, knee)
        cost_g, cov_g = eval_policy(hold_pop, "dgeo", cg, knee)
        _, cov_f = eval_policy(hold_pop, "flat", Tf, knee)
        rec = {"f": f, "geo_c": cg, "geo_cov_holdout": cov_g,
               "geo_cost_holdout": cost_g, "flat_T": Tf, "flat_cov_holdout": cov_f}
        out["heldout_scene"].append(rec)
        print(f"  f={f:.2f}: geo c={cg:.2f} held-out cov {100*cov_g:5.1f}% "
              f"cost {cost_g:5.1f} | flat T={Tf:.0f} held-out cov {100*cov_f:5.1f}%")

    # one constant across grids, unconditional per-grid readback
    f0 = 0.95 if 0.95 in a.fs else a.fs[len(a.fs) // 2]
    Tf = flat_budget(pop, f0, knee)
    _, pooled_target = eval_policy(pop, "flat", Tf, knee)
    cg = fit_c_to_coverage(pop, "dgeo", pooled_target, knee)
    out["one_constant"] = {"f": f0, "geo_c": cg, "flat_T": Tf, "per_grid": []}
    print(f"\none constant (f={f0:.2f}): geo c={cg:.2f}, flat T={Tf:.0f}; "
          f"unconditional per-grid:")
    for G in grids:
        sub = [r for r in pop if r["grid"] == G]
        cost_g, cov_g = eval_policy(sub, "dgeo", cg, knee)
        _, cov_f = eval_policy(sub, "flat", Tf, knee)
        rec = {"grid": G, "geo_cov": cov_g, "geo_cost": cost_g, "flat_cov": cov_f}
        out["one_constant"]["per_grid"].append(rec)
        print(f"  G={G:2d}: geo cov {100*cov_g:5.1f}% cost {cost_g:5.1f} | "
              f"flat cov {100*cov_f:5.1f}% cost {Tf:.0f}")

    # leave-one-seed-out, unconditional
    print(f"\nleave-one-seed-out at f={f0:.2f}, unconditional coverage:")
    for s in seeds:
        fit_rows = [r for r in pop if r["seed"] != s]
        hold = [r for r in pop if r["seed"] == s]
        T_fit = flat_budget(fit_rows, f0, knee)
        _, target_fit = eval_policy(fit_rows, "flat", T_fit, knee)
        c = fit_c_to_coverage(fit_rows, "dgeo", target_fit, knee)
        _, cov = eval_policy(hold, "dgeo", c, knee)
        out["loso"].append({"seed": s, "c": c, "coverage": cov})
        print(f"  held-out seed {s}: c={c:.2f} coverage {100*cov:5.1f}%")

    # full row table so figures/reanalysis need no GPU pool rebuild
    out["rows"] = rows

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nsaved -> {a.out} ({len(rows)} rows dumped)")


if __name__ == "__main__":
    main()

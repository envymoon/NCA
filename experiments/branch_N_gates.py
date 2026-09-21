"""Branch N post-training gate adjudication (R, S, P) on the 3-seed screened run.

Reads results_N_screened.json (pilot, seed 0) + results_N_screened_s12.json
(seeds 1 2), merges the per-grid scene dumps, rebuilds the deterministic val-pool
geometry via full_population_audit.build_rows (GPU seconds; geometry is
target-independent), and adjudicates the gates PRE-REGISTERED in
branch_N_diffusion.py (fixed 2026-07-19, before any run):

  R  geodesic per-scene CV < free AND < euc CV at every valid grid.
     Adjudicated on the seed-mean CV per grid (branch-E convention); the per-seed
     table and a scene-bootstrap CV-gap read (2.5th pct > 0) are reported too.
  S  c1(G32)/c1(G16) <= 1.15 AND no seed shows strictly monotone c1 growth.
     c1 values taken from the run JSONs (source of record).
  P  f=0.95, UNCONDITIONAL accounting (censored rows unserved, in denominator),
     integer policy T_i=max(1,ceil(c*D_i)), c fitted on even pool indices,
     frontier evaluated on odd; PASS iff held-out savings vs flat-T > 0 with a
     scene-bootstrap 95% CI excluding 0. Pooled across grids with knee=tail
     (the paper's headline accounting); mean-knee and per-grid fits reported as
     secondary. Bootstrap: B=2000, scenes stratified by grid, a scene's 3
     seed-rows travel together, calibration refitted inside each replicate.

Population: all 256 val scenes per grid x seed; clamped (Dgeo > 2G) excluded from
the primary population and reported (same as the paper audit).

Run:  cd experiments && python3 -u branch_N_gates.py --device cuda
Writes branch_N_gates.json. Read-only w.r.t. every frozen/addendum manifest file.
"""
import argparse
import json
import random
import statistics as st

from full_population_audit import build_rows
from policy_discrete import eval_policy, fit_c_to_coverage, flat_budget

B = 2000
RNG_SEED = 20260719
F_TARGET = 0.95


def cv(xs):
    return st.stdev(xs) / st.mean(xs)


def pct(xs, lo=2.5, hi=97.5):
    ys = sorted(xs)
    n = len(ys)
    return ys[int(lo / 100 * (n - 1))], ys[int(hi / 100 * (n - 1))]


def merged_results(p_pilot, p_s12):
    a = json.load(open(p_pilot))
    b = json.load(open(p_s12))
    assert a["theta"] == b["theta"] and a.get("walls") == b.get("walls")
    for g, e in a["per_grid"].items():
        e["scenes"] = e["scenes"] + b["per_grid"][g]["scenes"]
    a["seeds"] = sorted(a["seeds"] + b["seeds"])
    return a, b


def heldout_policy(pop, knee):
    """Fit flat T and matched geo c on even pool indices, evaluate on odd."""
    fit = [r for r in pop if r["pool_idx"] % 2 == 0]
    hold = [r for r in pop if r["pool_idx"] % 2 == 1]
    T = flat_budget(fit, F_TARGET, knee)
    _, target = eval_policy(fit, "flat", T, knee)
    c = fit_c_to_coverage(fit, "dgeo", target, knee)
    cost_g, cov_g = eval_policy(hold, "dgeo", c, knee)
    _, cov_f = eval_policy(hold, "flat", T, knee)
    return {"flat_T": T, "fit_target_cov": target, "geo_c": c,
            "holdout_geo_cost": cost_g, "holdout_geo_cov": cov_g,
            "holdout_flat_cov": cov_f, "holdout_saving": 1 - cost_g / T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results_N_screened.json")
    ap.add_argument("--s12", default="results_N_screened_s12.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default="branch_N_gates.json")
    a = ap.parse_args()

    res, res_b = merged_results(a.pilot, a.s12)
    rows, pool_geom = build_rows(res, a.device)
    pop = [r for r in rows if not r["clamped"]]
    grids = sorted({r["grid"] for r in pop})
    seeds = sorted({r["seed"] for r in pop})
    out = {"sources": [a.pilot, a.s12], "seeds": seeds, "theta": res["theta"],
           "clamped_per_grid": {G: len(v["clamped"]) for G, v in pool_geom.items()},
           "gate_R": {}, "gate_S": {}, "gate_P": {}}
    n_cens = sum(1 for r in pop if r["knee_tail"] is None)
    print(f"population: {len(pop)} rows ({len(rows)} incl. clamped), "
          f"{n_cens} tail-censored ({100 * (1 - n_cens / len(pop)):.1f}% converged)")

    # ---------------- gate R ----------------
    print("\nGATE R  (geo per-scene CV < free AND < euc at every grid)")
    verdict_R = {"mean": True, "tail": True}
    for label, knee in (("mean", "knee_mean"), ("tail", "knee_tail")):
        rec_g = {}
        for G in grids:
            per_seed = {}
            for s in seeds:
                conv = [r for r in pop
                        if r["grid"] == G and r["seed"] == s and r[knee] is not None]
                per_seed[s] = {rl: cv([r[knee] / max(r[rl], 1.0) for r in conv])
                               for rl in ("dgeo", "dfree", "deuc")}
            agg = {rl: st.mean(per_seed[s][rl] for s in seeds)
                   for rl in ("dgeo", "dfree", "deuc")}
            ok = agg["dgeo"] < agg["dfree"] and agg["dgeo"] < agg["deuc"]
            verdict_R[label] &= ok
            wins = sum(per_seed[s]["dgeo"] < per_seed[s]["dfree"]
                       and per_seed[s]["dgeo"] < per_seed[s]["deuc"] for s in seeds)
            rec_g[G] = {"per_seed": per_seed, "seed_mean": agg,
                        "seed_wins": wins, "pass": ok}
            print(f"  [{label}] G={G:2d} seed-mean CV geo {agg['dgeo']:.4f} "
                  f"free {agg['dfree']:.4f} euc {agg['deuc']:.4f}  "
                  f"per-seed wins {wins}/{len(seeds)}  -> "
                  f"{'PASS' if ok else 'FAIL'}")
            for s in seeds:
                p = per_seed[s]
                print(f"        s{s}: geo {p['dgeo']:.4f} free {p['dfree']:.4f} "
                      f"euc {p['deuc']:.4f}")
        out["gate_R"][label] = rec_g
    out["gate_R"]["verdict"] = {k: bool(v) for k, v in verdict_R.items()}

    # scene-bootstrap CV gaps (tail knee, branch-E ordering-claim read)
    scenes = {}
    for r in pop:
        scenes.setdefault((r["grid"], r["pool_idx"]), []).append(r)
    by_grid = {}
    for (G, _), rs in scenes.items():
        by_grid.setdefault(G, []).append(rs)
    rng = random.Random(RNG_SEED)

    def cv_gaps(rows_of_grid):
        conv = [r for r in rows_of_grid if r["knee_tail"] is not None]
        cvs = {rl: cv([r["knee_tail"] / max(r[rl], 1.0) for r in conv])
               for rl in ("dgeo", "dfree", "deuc")}
        return cvs["dfree"] - cvs["dgeo"], cvs["deuc"] - cvs["dgeo"]

    print("\n  scene-bootstrap CV gaps (tail), 2.5th pct > 0 supports ordering:")
    out["gate_R"]["bootstrap_gaps_tail"] = {}
    for G in grids:
        clusters = by_grid[G]
        point = cv_gaps([r for rs in clusters for r in rs])
        reps = []
        for _ in range(B):
            samp = [r for _ in range(len(clusters))
                    for r in clusters[rng.randrange(len(clusters))]]
            reps.append(cv_gaps(samp))
        (flo, fhi), (elo, ehi) = pct([x[0] for x in reps]), pct([x[1] for x in reps])
        out["gate_R"]["bootstrap_gaps_tail"][G] = {
            "free_gap": point[0], "free_ci": [flo, fhi],
            "euc_gap": point[1], "euc_ci": [elo, ehi]}
        print(f"    G={G:2d} free-geo {point[0]:+.3f} [{flo:+.3f},{fhi:+.3f}]  "
              f"euc-geo {point[1]:+.3f} [{elo:+.3f},{ehi:+.3f}]")

    # ---------------- gate S ----------------
    print("\nGATE S  (c1(G32)/c1(G16) <= 1.15 AND no per-seed monotone growth)")
    c1 = {}
    for src_path in (a.pilot, a.s12):
        src = json.load(open(src_path))
        for g, e in src["per_grid"].items():
            for s, v in zip(src["seeds"], e["c1_geo"]["vals"]):
                c1.setdefault(int(g), {})[s] = v
    mean_c1 = {G: st.mean(c1[G][s] for s in seeds) for G in grids}
    ratio = mean_c1[max(grids)] / mean_c1[min(grids)]
    mono = [s for s in seeds
            if all(c1[b][s] > c1[a_][s]
                   for a_, b in zip(grids, grids[1:]))]
    ok_S = ratio <= 1.15 and not mono
    out["gate_S"] = {"c1_per_seed": {G: c1[G] for G in grids},
                     "c1_seed_mean": mean_c1, "ratio_G32_G16": ratio,
                     "monotone_growth_seeds": mono, "pass": bool(ok_S)}
    for G in grids:
        print(f"  G={G:2d} c1_geo per seed "
              f"{[round(c1[G][s], 3) for s in seeds]}  mean {mean_c1[G]:.3f}")
    print(f"  ratio c1(G{max(grids)})/c1(G{min(grids)}) = {ratio:.3f}  "
          f"monotone-growth seeds: {mono or 'none'}  -> "
          f"{'PASS' if ok_S else 'FAIL'}")

    # ---------------- gate P ----------------
    print(f"\nGATE P  (f={F_TARGET}, unconditional, fit even / evaluate odd, "
          f"bootstrap CI on held-out savings excluding 0)")
    out["gate_P"]["pooled"] = {}
    for label, knee in (("tail", "knee_tail"), ("mean", "knee_mean")):
        point = heldout_policy(pop, knee)
        reps = []
        for _ in range(B):
            samp = []
            for G in grids:
                clusters = by_grid[G]
                samp.extend(r for _ in range(len(clusters))
                            for r in clusters[rng.randrange(len(clusters))])
            try:
                reps.append(heldout_policy(samp, knee)["holdout_saving"])
            except (ValueError, RuntimeError):
                continue
        lo, hi = pct(reps)
        rec = dict(point)
        rec.update({"saving_ci": [lo, hi], "boot_n": len(reps),
                    "pass": bool(point["holdout_saving"] > 0 and lo > 0)})
        out["gate_P"]["pooled"][label] = rec
        print(f"  [{label}] flat T={point['flat_T']} "
              f"(fit cov {100 * point['fit_target_cov']:.1f}%) | geo c={point['geo_c']:.2f} "
              f"held-out cost {point['holdout_geo_cost']:.1f} "
              f"cov {100 * point['holdout_geo_cov']:.1f}% "
              f"(flat {100 * point['holdout_flat_cov']:.1f}%) | "
              f"saving {100 * point['holdout_saving']:.1f}% "
              f"CI [{100 * lo:.1f}%, {100 * hi:.1f}%]  -> "
              f"{'PASS' if rec['pass'] else 'FAIL'}")

    print("\n  per-grid secondary read (tail):")
    out["gate_P"]["per_grid_tail"] = {}
    for G in grids:
        sub = [r for r in pop if r["grid"] == G]
        rec = heldout_policy(sub, "knee_tail")
        out["gate_P"]["per_grid_tail"][G] = rec
        print(f"    G={G:2d} flat T={rec['flat_T']} geo c={rec['geo_c']:.2f} "
              f"cost {rec['holdout_geo_cost']:.1f} cov {100 * rec['holdout_geo_cov']:.1f}% "
              f"(flat {100 * rec['holdout_flat_cov']:.1f}%) "
              f"saving {100 * rec['holdout_saving']:.1f}%")

    verdict = {"R_mean": out["gate_R"]["verdict"]["mean"],
               "R_tail": out["gate_R"]["verdict"]["tail"],
               "S": out["gate_S"]["pass"],
               "P_tail_pooled": out["gate_P"]["pooled"]["tail"]["pass"]}
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print("R+S+P all true -> second-main-task grade; R+S only -> limitations material.")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

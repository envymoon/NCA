import json
import random
import statistics as st

from policy_discrete import eval_policy, fit_c_to_coverage, flat_budget

B = 2000
RNG_SEED = 20260717
FS = (0.90, 0.95, 0.99)


def cv(xs):
    return st.stdev(xs) / st.mean(xs)


def pct(xs, lo=2.5, hi=97.5):
    ys = sorted(xs)
    n = len(ys)
    return ys[int(lo / 100 * (n - 1))], ys[int(hi / 100 * (n - 1))]


def stats_of(pop_by_grid, c_fix, t_fix):
    pop = [r for rows in pop_by_grid.values() for r in rows]
    out = {"ceiling": sum(1 for r in pop if r["knee_tail"] is not None) / len(pop)}
    conv = [r for r in pop if r["knee_tail"] is not None]
    for f in FS:
        Tf = flat_budget(pop, f, "knee_tail")
        _, target = eval_policy(pop, "flat", Tf, "knee_tail")
        cg = fit_c_to_coverage(pop, "dgeo", target, "knee_tail")
        cost_g, _ = eval_policy(pop, "dgeo", cg, "knee_tail")
        out[f"saving_f{f}"] = 1 - cost_g / Tf
    for G, rows in pop_by_grid.items():
        _, out[f"cov_geo_G{G}"] = eval_policy(rows, "dgeo", c_fix, "knee_tail")
        _, out[f"cov_flat_G{G}"] = eval_policy(rows, "flat", t_fix, "knee_tail")
        cg = [r for r in rows if r["knee_tail"] is not None]
        cvs = {rl: cv([r["knee_tail"] / max(r[rl], 1.0) for r in cg])
               for rl in ("dgeo", "dfree", "deuc")}
        out[f"cvgap_free_G{G}"] = cvs["dfree"] - cvs["dgeo"]
        out[f"cvgap_euc_G{G}"] = cvs["deuc"] - cvs["dgeo"]
    return out


def main():
    aud = json.load(open("audit_full_population.json"))
    oc = aud["one_constant"]
    c_fix, t_fix = oc["geo_c"], oc["flat_T"]
    rows = [r for r in aud["rows"] if not r["clamped"]]
    scenes = {}
    for r in rows:
        scenes.setdefault((r["grid"], r["pool_idx"]), []).append(r)
    by_grid = {}
    for (G, _), rs in scenes.items():
        by_grid.setdefault(G, []).append(rs)
    grids = sorted(by_grid)
    print(f"scenes per grid: {[len(by_grid[G]) for G in grids]}  "
          f"rows {len(rows)}  B={B}")

    point = stats_of({G: [r for rs in by_grid[G] for r in rs] for G in grids},
                     c_fix, t_fix)
    rng = random.Random(RNG_SEED)
    reps = []
    for _ in range(B):
        samp = {}
        for G in grids:
            clusters = by_grid[G]
            samp[G] = [r for _ in range(len(clusters))
                       for r in clusters[rng.randrange(len(clusters))]]
        reps.append(stats_of(samp, c_fix, t_fix))

    out = {"B": B, "rng_seed": RNG_SEED, "resampling_unit": "scene (grid, pool_idx), "
           "stratified by grid, 5 seed-rows travel together", "point": point, "ci": {}}
    print(f"\n{'metric':22s} {'point':>8s}   [2.5%, 97.5%]")
    for k in point:
        lo, hi = pct([rep[k] for rep in reps])
        out["ci"][k] = {"lo": lo, "hi": hi}
        print(f"{k:22s} {point[k]:8.3f}   [{lo:.3f}, {hi:.3f}]")
    gaps_ok = all(out["ci"][k]["lo"] > 0 for k in out["ci"] if k.startswith("cvgap"))
    out["ordering_gate_all_cvgaps_positive_at_2.5pct"] = gaps_ok
    print(f"\nordering gate (all six CV gaps > 0 at the 2.5th percentile): {gaps_ok}")

    with open("bootstrap_ci.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved -> bootstrap_ci.json")


if __name__ == "__main__":
    main()

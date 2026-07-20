import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

import torch

import branch_F_tortuosity as BF
from policy_discrete import eval_policy, fit_c_to_coverage, flat_budget

DEV = "cuda"
DATA = Path("../data/TheVGLC")
CKPT = "results_E_maze_s34.json.G{G}.s{s}.pt"
CAP = 152
THETA = 0.05
TAIL_Q = 0.95
F0 = 0.95
RNG_SEED = 20260717
MIN_REACH = 0.20
TRIES = 8
SLICE = 64

LR_OBST = set("Bb")
LR_PASS = set(".-#GME")
ZE_OBST = set("WBPOI-")
ZE_PASS = set("FDSM")


def load_lode_runner():
    out = []
    for f in sorted((DATA / "Lode Runner" / "Processed").glob("*.txt")):
        lines = [l.rstrip("\n") for l in f.open() if l.strip("\n")]
        assert len(lines) == 22 and all(len(l) == 32 for l in lines), f.name
        obs = [[1.0] * 32 for _ in range(5)]
        for l in lines:
            row = []
            for ch in l:
                assert ch in LR_OBST | LR_PASS, (f.name, ch)
                row.append(1.0 if ch in LR_OBST else 0.0)
            obs.append(row)
        obs += [[1.0] * 32 for _ in range(5)]
        out.append((f.stem, obs))
    return out, 32


def load_zelda_rooms():
    seen, out = set(), []
    for f in sorted((DATA / "The Legend of Zelda" / "Processed").glob("tloz*.txt")):
        lines = [l.rstrip("\n") for l in f.open() if l.strip("\n")]
        H, W = len(lines), len(lines[0])
        assert H % 16 == 0 and W % 11 == 0, (f.name, H, W)
        for ry in range(H // 16):
            for rx in range(W // 11):
                tile = [lines[ry * 16 + y][rx * 11:(rx + 1) * 11] for y in range(16)]
                if all(set(t) <= {"-"} for t in tile):
                    continue
                key = "\n".join(tile)
                if key in seen:
                    continue
                seen.add(key)
                obs = []
                for t in tile:
                    row = [1.0, 1.0]
                    for ch in t:
                        assert ch in ZE_OBST | ZE_PASS, (f.name, ch)
                        row.append(1.0 if ch in ZE_OBST else 0.0)
                    row += [1.0, 1.0, 1.0]
                    obs.append(row)
                out.append((f"{f.stem}_r{ry}{rx}", obs))
    return out, 16


def build_scenes(levels, G, k_seeds, rng):
    seeds, obses, names = [], [], []
    for name, obs_l in levels:
        obs_t = torch.tensor(obs_l, device=DEV).view(1, 1, G, G)
        free = (obs_t[0, 0] < 0.5).nonzero(as_tuple=False).tolist()
        if not free:
            continue
        n_free = len(free)
        used = set()
        for _ in range(k_seeds):
            ok = None
            for _ in range(TRIES):
                y, x = free[rng.randrange(n_free)]
                if (y, x) in used:
                    continue
                s = torch.zeros_like(obs_t)
                s[0, 0, y, x] = 1.0
                d = BF.bfs8(s, obs_t, max_iter=8 * (G + G))
                reach_frac = (d < BF.INF / 2).float().sum().item() / n_free
                if reach_frac >= MIN_REACH:
                    ok = (y, x)
                    break
            if ok is None:
                continue
            used.add(ok)
            s = torch.zeros_like(obs_t)
            s[0, 0, ok[0], ok[1]] = 1.0
            seeds.append(s)
            obses.append(obs_t)
            names.append(name)
    seed = torch.cat(seeds)
    obs = torch.cat(obses)
    d = BF.bfs8(seed, obs, max_iter=8 * (G + G))
    reach = (d < BF.INF / 2).float()
    Dnorm = 2.0 * G
    tgt = torch.clamp(torch.where(reach > 0, d, torch.full_like(d, Dnorm)),
                      max=Dnorm) / Dnorm
    Dgeo, Dfree, Deuc = [], [], []
    for b in range(seed.shape[0]):
        rd = d[b, 0]
        far = (rd * reach[b, 0]).argmax()
        fy, fx = int(far // G), int(far % G)
        Dgeo.append(rd[fy, fx].item())
        sy, sx = (seed[b, 0] > 0).nonzero(as_tuple=False)[0]
        Dfree.append(float(max(abs(fy - int(sy)), abs(fx - int(sx)))))
        Deuc.append(float(((fy - int(sy)) ** 2 + (fx - int(sx)) ** 2) ** 0.5))
    inp = torch.cat([seed, obs], 1)
    mask = reach * (obs < 0.5).float()
    return inp, tgt, mask, Dgeo, Dfree, Deuc, names


@torch.no_grad()
def per_scene_knees(model, inp, tgt, mask, var):
    Ts = list(range(1, CAP + 1))
    pst = {t: [] for t in Ts}
    N = inp.shape[0]
    for lo in range(0, N, SLICE):
        sl = slice(lo, min(lo + SLICE, N))
        outs = model.run(inp[sl], CAP, collect=set(Ts))
        for t in Ts:
            pst[t].append(BF.tail_nmse(outs[t], tgt[sl], mask[sl], var, q=TAIL_Q).cpu())
    for t in Ts:
        pst[t] = torch.cat(pst[t])
    knees = []
    for i in range(N):
        k, _ = BF.abs_knee({t: pst[t][i].item() for t in Ts}, theta=THETA)
        knees.append(k)
    return knees


def quant(xs, q):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, math.ceil(q * len(ys)) - 1))]


def cv(xs):
    return st.stdev(xs) / st.mean(xs)


def coverage(rows, budget_fn):
    served = sum(1 for r in rows if r["knee"] is not None and r["knee"] <= budget_fn(r))
    cost = st.mean(budget_fn(r) for r in rows)
    return cost, served / len(rows)


def maze_constants(aud):
    oc = aud["one_constant"]
    c_one, T_one = oc["geo_c"], int(round(oc["flat_T"]))
    pg = {}
    for G in (16, 32):
        fit_rows = [r for r in aud["rows"] if r["grid"] == G and not r["clamped"]]
        T = flat_budget(fit_rows, F0, "knee_tail")
        _, target = eval_policy(fit_rows, "flat", T, "knee_tail")
        pg[G] = {"c": fit_c_to_coverage(fit_rows, "dgeo", target, "knee_tail"),
                 "T": T}
    return c_one, T_one, pg


def recalibrate_rows(rows, c_one, T_one, pg, G):
    cal = {}
    for label, cg, Tf in (("one_constant", c_one, T_one),
                          ("per_grid", pg[G]["c"], pg[G]["T"])):
        gc_cost, gc_cov = coverage(rows, lambda r, c=cg: math.ceil(c * r["dgeo"]))
        fl_cost, fl_cov = coverage(rows, lambda r, T=Tf: T)
        cal[label] = {"geo_c": cg, "flat_T": Tf, "geo_cost": gc_cost,
                      "geo_cov": gc_cov, "flat_cost": fl_cost, "flat_cov": fl_cov}
    return cal


def reanalyze_existing(path):
    aud = json.load(open("audit_full_population.json"))
    c_one, T_one, pg = maze_constants(aud)
    out = json.load(open(path))
    out["protocol"]["budget_policy"] = "integer T=max(1,ceil(cD)); c on 0.01 grid"
    out["maze_constants"] = {"one_constant": {"c": c_one, "T": T_one},
                             "per_grid": {str(g): pg[g] for g in pg}}
    for grec in out["games"].values():
        G = grec["grid"]
        for ckpt in grec["ckpts"].values():
            ckpt["calibration"] = recalibrate_rows(ckpt["rows"], c_one, T_one, pg, G)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"recalibrated integer policies -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reanalyze-existing", action="store_true")
    ap.add_argument("--existing", default="eval_vglc.json")
    args = ap.parse_args()
    if args.reanalyze_existing:
        reanalyze_existing(args.existing)
        return
    assert torch.cuda.is_available()
    aud = json.load(open("audit_full_population.json"))
    c_one, T_one, pg = maze_constants(aud)
    print(f"maze constants: one_constant c={c_one:.4f} T={T_one} | "
          f"per-grid G16 c={pg[16]['c']:.2f} T={pg[16]['T']}  "
          f"G32 c={pg[32]['c']:.2f} T={pg[32]['T']}")

    rng = random.Random(RNG_SEED)
    games = [("lode_runner", *load_lode_runner(), 4),
             ("zelda", *load_zelda_rooms(), 2)]

    out = {"data": "TheVGLC (github.com/TheVGLC/TheVGLC), sparse checkout",
           "protocol": {"theta": THETA, "cap": CAP, "tail_q": TAIL_Q, "f": F0,
                        "min_reach": MIN_REACH, "rng_seed": RNG_SEED},
           "maze_constants": {"one_constant": {"c": c_one, "T": T_one},
                              "per_grid": {str(g): pg[g] for g in pg}},
           "games": {}}

    for game, levels, G, k_seeds in games:
        print(f"\n=== {game}  G={G}  levels/rooms={len(levels)} ===")
        inp, tgt, mask, Dgeo, Dfree, Deuc, names = build_scenes(levels, G, k_seeds, rng)
        var = tgt[mask > 0].var().item() + 1e-12
        n_all = inp.shape[0]
        elig = [i for i in range(n_all) if Dgeo[i] <= 2 * G]
        print(f"scenes={n_all}  eligible(unclamped)={len(elig)}  pool var={var:.5f}  "
              f"Dgeo range {min(Dgeo):.0f}-{max(Dgeo):.0f}")
        grec = {"grid": G, "n_scenes": n_all, "n_eligible": len(elig), "var": var,
                "ckpts": {}}
        for s in (3, 4):
            model = BF.NCA().to(DEV)
            model.load_state_dict(torch.load(CKPT.format(G=G, s=s), map_location=DEV))
            model.eval()
            knees = per_scene_knees(model, inp, tgt, mask, var)
            rows = [{"name": names[i], "dgeo": max(Dgeo[i], 1.0),
                     "dfree": max(Dfree[i], 1.0), "deuc": max(Deuc[i], 1.0),
                     "knee": knees[i]} for i in elig]
            conv = [r for r in rows if r["knee"] is not None]
            cov_ceiling = len(conv) / len(rows)
            cvs = {rl: cv([r["knee"] / r[rl] for r in conv])
                   for rl in ("dgeo", "dfree", "deuc")}
            cal = recalibrate_rows(rows, c_one, T_one, pg, G)
            c_local = quant([r["knee"] / r["dgeo"] for r in conv], F0)
            grec["ckpts"][f"s{s}"] = {
                "coverage_ceiling": cov_ceiling, "n_conv": len(conv),
                "cv": cvs, "calibration": cal,
                "diag_local_c_f95": c_local,
                "rows": rows}
            print(f"[s{s}] (a) coverage: {len(conv)}/{len(rows)} converge "
                  f"= {100*cov_ceiling:.1f}%   (maze ceiling was 94.2%)")
            print(f"[s{s}] (b) ordering: CV geo {100*cvs['dgeo']:.1f}%  "
                  f"free {100*cvs['dfree']:.1f}%  euc {100*cvs['deuc']:.1f}%  "
                  f"-> geo smallest: {cvs['dgeo'] < min(cvs['dfree'], cvs['deuc'])}")
            for label in ("one_constant", "per_grid"):
                a = cal[label]
                print(f"[s{s}] (c) {label:12s} geo c={a['geo_c']:.2f}: "
                      f"cov {100*a['geo_cov']:.1f}% cost {a['geo_cost']:.1f} | "
                      f"flat T={a['flat_T']}: cov {100*a['flat_cov']:.1f}% "
                      f"cost {a['flat_cost']:.1f}")
            print(f"[s{s}] diagnostic only: locally-fitted c(f=.95)={c_local:.2f} "
                  f"vs maze {pg[G]['c']:.2f}")
        out["games"][game] = grec

    with open("eval_vglc.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nsaved -> eval_vglc.json")


if __name__ == "__main__":
    main()

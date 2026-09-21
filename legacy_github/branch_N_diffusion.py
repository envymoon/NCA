import argparse
import json
import sys

import torch
import torch.nn.functional as F

import branch_F_tortuosity as BF
import branch_E_anytime as E

THETA = 0.05
TAIL_Q = 0.95
GAMMA_LADDER = (0.995, 0.998, 0.999)
GATE_O_RFRAC = 0.60
GATE_D_NMSE = 0.20
FRACS = [i / 20 for i in range(21)]

_NB = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]).view(1, 1, 3, 3)
_NB_CACHE = {}


def _nbsum(x):
    k = _NB_CACHE.get(x.device)
    if k is None:
        k = _NB_CACHE[x.device] = _NB.to(x.device)
    return F.conv2d(x, k, padding=1)


def screened_steady(seed, obs, gamma, tol=1e-7, max_iter=40000, check=25):
    free = (obs < 0.5).float()
    s = seed * free
    cnt = _nbsum(free).clamp(min=1.0)
    u = torch.zeros_like(s)
    for it in range(0, max_iter, check):
        u_prev = u
        for _ in range(check):
            u = (s + gamma * _nbsum(u * free) / cnt) * free
        if (u - u_prev).abs().max() < tol:
            return u, it + check
    return u, max_iter


def harmonic_measure(seed, obs, tol=1e-7, max_iter=60000, check=50):
    free = (obs < 0.5).float()
    src = (seed > 0).float() * free
    ones = torch.ones_like(free)
    u = src.clone()
    for it in range(0, max_iter, check):
        u_prev = u
        for _ in range(check):
            u = _nbsum(u * free) / 8.0 * free
            u = torch.where(src > 0, ones, u)
        if (u - u_prev).abs().max() < tol:
            return u, it + check
    return u, max_iter


def make_target(task, seed, obs, gamma):
    if task == "screened":
        u, iters = screened_steady(seed, obs, gamma=gamma)
        peak = u.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        return u / peak, iters
    u, iters = harmonic_measure(seed, obs)
    return u, iters


def install(task, gamma, report):
    orig = BF.build_pool_maze

    def build_pool_diffusion(n, H, W, device, gen, walls=(1, 8), noise=0.05):
        inp, _tgt, lm, Dgeo, Dfree, Deuc, tau = orig(n, H, W, device, gen,
                                                     walls=walls, noise=noise)
        seed, obs = inp[:, :1], inp[:, 1:]
        tgt, iters = make_target(task, seed, obs, gamma)
        report.append({"pool": f"{n}x{H}x{W}", "solver_iters": iters,
                       "tgt_var_masked": tgt[lm > 0].var().item()})
        print(f"    [branch N] target={task} pool {n}x{H}x{W}: solver converged in "
              f"{iters} iters, masked var {tgt[lm > 0].var().item():.5f}")
        return inp, tgt, lm, Dgeo, Dfree, Deuc, tau

    BF.build_pool_maze = build_pool_diffusion


def ring_cv(tgt, lm, dfield):
    vals = []
    m = lm > 0
    d = dfield[m].round().long()
    t = tgt[m]
    for r in d.unique():
        x = t[d == r]
        if len(x) >= 8 and x.mean().abs() > 1e-9:
            vals.append((x.std(unbiased=False) / x.mean().abs()).item())
    vals.sort()
    return vals[len(vals) // 2] if vals else float("nan")


def distance_only_field(tgt, lm, d):
    m = lm > 0
    dv = d[m].round().long().clamp(min=0)
    K = int(dv.max().item()) + 1
    sums = torch.zeros(K, device=tgt.device).index_add_(0, dv, tgt[m])
    cnts = torch.zeros(K, device=tgt.device).index_add_(0, dv, torch.ones_like(tgt[m]))
    g = sums / cnts.clamp(min=1.0)
    idx = d.round().long().clamp(min=0, max=K - 1)
    return g[idx]


def reveal_rstar(tgt, lm, d, Dgeo, var, fill, theta=THETA):
    B = tgt.shape[0]
    rstar = torch.full((B,), 1.0)
    done = torch.zeros(B, dtype=torch.bool)
    rad = Dgeo.view(B, 1, 1, 1).to(tgt.device)
    for f in FRACS:
        pred = torch.where(d <= rad * f, tgt, fill)
        tail = BF.tail_nmse(pred, tgt, lm, var, q=TAIL_Q).cpu()
        newly = (~done) & (tail <= theta)
        rstar[newly] = f
        done |= newly
        if bool(done.all()):
            break
    return rstar


def _q(t, f):
    return torch.quantile(t, f).item()


def dry_targets(task, grids):
    dev = BF.get_device("auto")
    gammas = GAMMA_LADDER if task == "screened" else (None,)
    print(f"Branch N preflight v2  task={task}  dev={dev}  grids={grids}  "
          f"theta={THETA} tail_q={TAIL_Q}")
    print(f"gates: O median r*/Dgeo >= {GATE_O_RFRAC} (ring fill), "
          f"D median dist-only tail NMSE >= {GATE_D_NMSE}, both at every grid")
    passes = {g: True for g in gammas}
    for G in grids:
        gen = torch.Generator(device=dev).manual_seed(1)
        inp, _t0, lm, Dgeo, Dfree, Deuc, tau = BF.build_pool_maze(256, G, G, dev, gen)
        seed, obs = inp[:, :1], inp[:, 1:]
        d = BF.bfs8(seed, obs, max_iter=8 * (G + G))
        print(f"  G={G:2d}  median Dgeo {Dgeo.median().item():.0f}  "
              f"(p90 {_q(Dgeo, 0.9):.0f})")
        for gm in gammas:
            tgt, iters = make_target(task, seed, obs, gm)
            var = tgt[lm > 0].var().item() + 1e-12
            gfield = distance_only_field(tgt, lm, d)
            mfill = ((tgt * lm).sum(dim=(1, 2, 3)) /
                     lm.sum(dim=(1, 2, 3)).clamp(min=1)).view(-1, 1, 1, 1) * torch.ones_like(tgt)
            zero_t = BF.tail_nmse(torch.zeros_like(tgt), tgt, lm, var, q=TAIL_Q).median().item()
            mean_t = BF.tail_nmse(mfill, tgt, lm, var, q=TAIL_Q).median().item()
            dist_t = BF.tail_nmse(gfield, tgt, lm, var, q=TAIL_Q).median().item()
            rcv = ring_cv(tgt, lm, d)
            rs_ring = reveal_rstar(tgt, lm, d, Dgeo, var, gfield)
            rs_mean = reveal_rstar(tgt, lm, d, Dgeo, var, mfill)
            gate_o = _q(rs_ring, 0.5) >= GATE_O_RFRAC
            gate_d = dist_t >= GATE_D_NMSE
            passes[gm] = passes[gm] and gate_o and gate_d
            tag = f"gamma={gm}" if gm is not None else "harmonic"
            print(f"    {tag:12s} solver {iters:6d} it | pool var {var:.5f} | "
                  f"tail NMSE zero {zero_t:.2f} mean {mean_t:.2f} dist-only {dist_t:.2f} | "
                  f"ring-CV {rcv:.3f}")
            print(f"    {'':12s} r*/Dgeo ring-fill med {_q(rs_ring, 0.5):.2f} "
                  f"p90 {_q(rs_ring, 0.9):.2f} | mean-fill med {_q(rs_mean, 0.5):.2f} | "
                  f"gate O {'PASS' if gate_o else 'FAIL'}  gate D {'PASS' if gate_d else 'FAIL'}")
    print("-" * 72)
    if task == "screened":
        chosen = next((g for g in GAMMA_LADDER if passes[g]), None)
        if chosen is None:
            print("VERDICT: no ladder gamma passes gates O+D at every grid -> "
                  "screened variant INVALID as configured, do not train.")
        else:
            print(f"VERDICT: selected gamma = {chosen} (smallest ladder value passing "
                  f"O+D at every grid). Train with --gamma {chosen} after consent.")
    else:
        ok = passes[None]
        print(f"VERDICT: harmonic {'PASSES' if ok else 'FAILS'} gates O+D "
              f"{'-> eligible as scope-boundary probe (after consent)' if ok else '-> do not train'}.")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--task", choices=["screened", "harmonic"], default="screened")
    ap.add_argument("--gamma", type=float, default=None,
                    help="screened retention per sweep; GLOBAL; must be the "
                         "preflight-selected ladder value -- no default")
    ap.add_argument("--dry_targets", action="store_true",
                    help="build targets + gate diagnostics, no training")
    ns, rest = ap.parse_known_args()

    gp = argparse.ArgumentParser(add_help=False)
    gp.add_argument("--grids", type=int, nargs="+", default=[16, 24, 32])
    grids = gp.parse_known_args(rest)[0].grids

    if ns.dry_targets:
        dry_targets(ns.task, grids)
        return

    if ns.task == "screened":
        if ns.gamma is None:
            sys.exit("branch N: --gamma is required for a screened run; use the value "
                     "selected by  --dry_targets --task screened  (pre-registered rule).")
        if ns.gamma not in GAMMA_LADDER:
            sys.exit(f"branch N: --gamma {ns.gamma} is not in the pre-registered "
                     f"ladder {GAMMA_LADDER}; refusing an untracked value.")

    report = []
    install(ns.task, ns.gamma, report)
    print(f"Branch N: non-distance target ({ns.task}"
          + (f", gamma={ns.gamma}" if ns.task == "screened" else "")
          + "), protocol = branch E verbatim, architecture = learned-perception "
            "BF.NCA (paper original)")
    sys.argv = [sys.argv[0]] + rest
    E.main()

    out = "results_E_maze.json"
    if "--out" in rest:
        out = rest[rest.index("--out") + 1]
    try:
        res = json.load(open(out))
    except FileNotFoundError:
        return
    res["task"] = {
        "name": ns.task,
        "target": ("screened-Poisson steady state, gamma=%g global "
                   "(ladder-selected pre-run), per-scene source-normalised" % ns.gamma)
                  if ns.task == "screened"
                  else "harmonic measure (hit source before wall/edge absorption)",
        "pools": report,
        "baseline": "distance-transform target (results_E_maze_s01234.json)",
        "gates": ("target gates O (median r*/Dgeo>=0.60, ring fill) and D (dist-only "
                  "tail NMSE>=0.20) passed at preflight; post-training: R geo CV < "
                  "free/euc at every grid; S c1(G32)/c1(G16)<=1.15 and no per-seed "
                  "monotone growth; P unconditional geo cost < flat at f=0.95 with "
                  "bootstrap CI excluding 0, c fitted on even-indexed val scenes, "
                  "frontier evaluated on odd-indexed; pilot seed 0 committed"),
    }
    json.dump(res, open(out, "w"), indent=2)
    print(f"task stamp written -> {out}")


if __name__ == "__main__":
    main()

"""Branch N — non-distance propagation task (v2, 2026-07-19; NOT YET TRAINED).

WHY THIS EXPERIMENT. Limitations bullet "One task family": every target so far is a
distance field, so a critic can say the budget rule is circular -- "the task IS BFS,
of course the knee tracks D_geo". This branch keeps the ENTIRE frozen protocol
(branch_E_anytime verbatim: same maze pools from the same generator seeds 0/1, hence
the identical global cap; same theta=0.05 tail-NMSE knee; same exclusion rules; same
per-scene row dump; original learned-perception BF.NCA) and changes exactly ONE
thing: the target is no longer a distance transform but an obstacle-aware diffusion
field seeded at the source cell.

v2 CHANGELOG (same day as v1, before any run; kept for the audit trail):
  * v1 claimed gamma=0.97 gives a "~33 cell decay length". WRONG: 1/(1-gamma)=33 is
    the walk's mean LIFETIME (a time). The spatial decay factor per cell is
    lambda = (1 - sqrt(1 - gamma^2))/gamma  (exact 1-D lattice solution), i.e.
    ell = -1/ln(lambda) ~ 4-8 cells at gamma=0.97. Such a target is near-source
    concentrated: far cells are ~0, cost ~0 to predict, and the knee would track the
    constant ell instead of D_geo. Training on it answers nothing. gamma is now
    chosen by the pre-registered ladder + oracle rule below, never by hand.
  * v1's harmonic docstring said "out-of-bounds absorb" but the denominator counted
    only in-bounds neighbours (= reflecting edges; make_maze does NOT guarantee a
    wall border). v2 divides by the full 8-neighbourhood so out-of-bounds and walls
    both truly absorb, matching the text.
  * preflight upgraded from ring-CV-only to the reveal-radius oracle + distance-only
    predictor gates (O and D below): ring-CV certifies the label is not a function
    of distance, but not that far cells matter for reaching theta. The oracle does.

TWO TASK VARIANTS (--task):
  screened  u* solves  u = s + gamma * nbavg_free(u)  on free cells (screened-
            Poisson attenuation: light/sound/signal falloff around walls, a real
            shader effect). Target = u*/max(u*) per scene. gamma is GLOBAL -- one
            value for every grid and seed -- selected by the pre-registered rule:
              ladder {0.995, 0.998, 0.999}  (ell ~ sqrt(gamma/(1-gamma)) ~ 14/22/32
              cells); pick the SMALLEST gamma passing gates O and D at EVERY grid;
              if none passes, screened is declared invalid and is not trained.
            The rule sees only target fields, never training results: not tuning.
  harmonic  u* = P(random walk hits the source before absorption): u=1 at source,
            absorbed (value 0) at walls AND out of bounds, u = full-8-neighbourhood
            average elsewhere. Values depend on corridor widths and wall proximity,
            not on distance alone -- the harder anti-circularity control. NOTE: the
            paper scopes elliptic steady states to multigrid-class solvers, and a
            local relaxation scale can be O(D^2); harmonic is therefore run as a
            SCOPE-BOUNDARY probe / negative control, not as primary support. It
            must pass the same gates O and D before any training.

PRE-REGISTERED GATES (fixed 2026-07-19, before any run):
  Target-validity gates, judged by --dry_targets on the val pools (gen seed 1),
  metric = BF.tail_nmse (p95 per-cell quantile / pool var), theta = 0.05 -- the
  knee metric verbatim:
  O (oracle)     reveal-radius: predictor = true target inside the geodesic ball
                 of radius r = f*D_geo(scene), best distance-only field g(d)
                 outside; r*(scene) = smallest f with tail NMSE <= theta.
                 PASS iff median r*/D_geo >= 0.60 at every grid.
                 (A pure distance task passes trivially; a near-source-concentrated
                 task fails: it certifies the threshold NEEDS far propagation.)
  D (distance)   best distance-only predictor g(d) = pool-level per-ring mean.
                 PASS iff median per-scene tail NMSE of g >= 0.20 (= 4*theta) at
                 every grid: no distance-only predictor can come near the knee, so
                 the label is certifiably not distance-computable. (ring-CV is
                 also reported as the descriptive label!=distance statistic.)
  Post-training gates (unchanged protocol, quantified up front):
  R (ruler)      geodesic per-scene CV < free AND < euc CV at every valid grid.
  S (sufficiency) PASS iff c1(G32)/c1(G16) <= 1.15 AND no seed shows a strictly
                 monotone c1 increase across grids. The VALUE of c1 is free to
                 differ from 1 (a steady state may need a settling margin past
                 first arrival); only systematic growth is disqualifying.
  P (policy)     at target coverage f=0.95 under UNCONDITIONAL accounting (all
                 censored rows counted unserved), the integer geodesic policy's
                 mean provisioned steps < flat-T cost, with a bootstrap 95% CI on
                 the savings excluding 0. Calibration is held out: c is fitted on
                 even-indexed val scenes, the frontier evaluated on odd-indexed.
  Exclusions inherited verbatim (non-convergent / knee-at-cap / <32 scenes reported,
  never imputed). Floors near theta read as capacity, per the Branch M precedent.
  R+S alone reproduce a Branch L/M-class control (limitations material only);
  R+S+P is what would qualify Branch N as a second main task in the paper.

PILOT DISCIPLINE. First run = seed 0 only, grids 16 24 32 (~1-1.2 h), to check the
model floor clears theta with margin. The pilot seed is committed: it counts in the
final statistics whatever it shows, and a bad pilot may only abandon the whole
variant, never swap the seed.

PREFLIGHT (target-only, no training; minutes):
  python3 -u branch_N_diffusion.py --dry_targets --task screened
  python3 -u branch_N_diffusion.py --dry_targets --task harmonic
Prints, per grid (and per ladder gamma for screened): solver iterations, pool var,
trivial/zero/distance-only NMSE, ring-CV, r* quartiles for both fill modes, and the
gate verdicts; for screened, the selected gamma (or "invalid, do not train").

RUN (only after preflight passes AND explicit user consent; -u to avoid buffered
logs; --gamma must be the preflight-selected value, there is no default):
  python3 -u branch_N_diffusion.py --task screened --gamma <selected> \
      --grids 16 24 32 --seeds 0 --out results_N_screened.json          # pilot
  ... then seeds 1 2 appended to the same protocol if the pilot floor clears.
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

import branch_F_tortuosity as BF
import branch_E_anytime as E

THETA = 0.05                      # knee threshold, branch E verbatim
TAIL_Q = 0.95                     # tail quantile, branch E verbatim
GAMMA_LADDER = (0.995, 0.998, 0.999)
GATE_O_RFRAC = 0.60               # median r*/D_geo (ring fill) needed at every grid
GATE_D_NMSE = 0.20                # 4*theta: distance-only predictor must stay above
FRACS = [i / 20 for i in range(21)]

_NB = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]).view(1, 1, 3, 3)
_NB_CACHE = {}


def _nbsum(x):
    k = _NB_CACHE.get(x.device)
    if k is None:
        k = _NB_CACHE[x.device] = _NB.to(x.device)
    return F.conv2d(x, k, padding=1)


def screened_steady(seed, obs, gamma, tol=1e-7, max_iter=40000, check=25):
    """u = s + gamma * (free-neighbour average of u), zero flux through walls.

    Fixed-point contraction ~ gamma per sweep, so iterations ~ ln(tol)/ln(gamma)
    (~16k at gamma=0.999); tol is tested every `check` sweeps to avoid syncs.
    """
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
    """P(random walk hits source before absorption), absorbing walls AND edges.

    Every cell averages over its FULL 8-neighbourhood: walls contribute 0 (they
    absorb), out-of-bounds contributes 0 via zero padding (the grid edge absorbs
    too -- make_maze does not guarantee a wall border, so this is enforced by the
    constant denominator, not assumed). The source is a Dirichlet 1 every sweep.
    """
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
    """Swap the pool builder: identical scenes/rulers/mask (same RNG), new target only."""
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


# ------------------------------------------------------------ preflight machinery

def ring_cv(tgt, lm, dfield):
    """Median over integer-distance rings (pooled across scenes) of within-ring CV.

    A pure distance transform scores 0; positive values certify the label is not a
    function of per-cell distance. Descriptive statistic only -- gates O and D are
    what actually authorise training.
    """
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
    """g(d): pool-level per-ring mean, evaluated everywhere -- the best predictor of
    the form 'a function of the cell's BFS distance alone' (the circularity
    hypothesis made concrete). Pool-level, so no per-scene information leaks in."""
    m = lm > 0
    dv = d[m].round().long().clamp(min=0)
    K = int(dv.max().item()) + 1
    sums = torch.zeros(K, device=tgt.device).index_add_(0, dv, tgt[m])
    cnts = torch.zeros(K, device=tgt.device).index_add_(0, dv, torch.ones_like(tgt[m]))
    g = sums / cnts.clamp(min=1.0)
    idx = d.round().long().clamp(min=0, max=K - 1)
    return g[idx]


def reveal_rstar(tgt, lm, d, Dgeo, var, fill, theta=THETA):
    """Per-scene reveal-radius fraction r*: smallest f in FRACS such that
    [true target inside the geodesic ball d <= f*D_geo, `fill` outside] reaches
    tail NMSE <= theta. f=1.0 reveals every reached cell, so r* always exists."""
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
        gen = torch.Generator(device=dev).manual_seed(1)          # val pool, branch E verbatim
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

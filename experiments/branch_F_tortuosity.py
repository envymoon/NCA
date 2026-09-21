"""
Branch F — turn the loose constant c1 (observed 1.0-1.8 across effects) into a
PREDICTABLE, tortuosity-invariant causal constant.

Key refinement over branch E: a 3x3-kernel NCA's causal cone grows by one CHEBYSHEV
(L-inf, includes diagonals) ring per step, so the matching classical metric is the
8-CONNECTED geodesic, not the 4-connected one. We use 8-connected BFS here.

Knob: a serpentine "comb" maze with `teeth` walls forces a snake path whose geodesic
length grows monotonically with teeth while the straight-line (free-space) distance
stays ~fixed -> tortuosity  tau = D_geo / D_free  spans a wide, monotone range.

Hypotheses (these lift the result from "measured a slope" to "predict the slope"):
  * c1_geo = T* / D_geo  is INVARIANT to tortuosity (a clean causal constant ~1);
  * budgeting by the straight-line free-space distance UNDER-provisions steps by
    exactly the tortuosity factor tau (effect fails to propagate) -> deployment rule:
    budget on the causal geodesic, not the screen-space straight line.

Run: python3 branch_F_tortuosity.py
"""
import argparse, json, time
import statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

INF = 1e9

# A convergence curve must fall by at least this factor for its knee to mean
# anything (see robust_knee). Curves below this are training failures, not
# instant convergence, and are reported as invalid rather than silently used.
MIN_DYN = 2.0


# ---------------------------------------------------------------- audit fixes
# Shared helpers used by every branch (E/F2/F3/G/H2/J) so the camera-ready
# protocol is defined ONCE, not re-implemented per script.

def get_device(pref="auto"):
    """CPU fallback. The old `assert torch.cuda.is_available()` made every branch
    unrunnable on a CPU-only box; the physics under test is device-independent."""
    if pref == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not visible"
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    """Seed the MODEL init + training sampling. Previously only the data
    generators were seeded and `NCA()` was constructed from global RNG state,
    so every keystone number was a single unseeded run."""
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def smooth_median(curve, win=5):
    """Moving median over T -> kills single-point noise without shifting a step edge
    the way a moving mean would."""
    Ts = sorted(curve)
    h = win // 2
    out = {}
    for i, t in enumerate(Ts):
        lo, hi = max(0, i - h), min(len(Ts), i + h + 1)
        out[t] = st.median([curve[u] for u in Ts[lo:hi]])
    return out


def curve_drift(curve, win=5):
    """smoothed[-1] / floor. >1 means the error CREEPS BACK UP at long horizons -- an NCA
    long-horizon stability issue, which is a DIFFERENT phenomenon from not having converged.
    Reported separately so it cannot corrupt the knee (see robust_knee)."""
    s = smooth_median(curve, win)
    Ts = sorted(s)
    floor = min(s.values())
    return (s[Ts[-1]] / floor) if floor > 0 else float("nan")


def robust_knee(curve, tol=0.05, persist=10, win=5):
    """Knee = first arrival at the error floor that HOLDS for `persist` steps.

    The original `next(t for t in Ts if curve[t] <= 1.05*min(curve))` is fragile:
      * the floor is the global MIN of a noisy curve -> biased low -> threshold too
        strict, and one lucky dip sets the bar for every t;
      * it takes the FIRST touch, so a single noise dip that later rises again is
        reported as convergence.

    But requiring the crossing to hold to the END of the horizon (a natural-looking
    fix) is also wrong: with ONE GLOBAL cap the horizon is several times longer than
    small grids need, and an NCA that idles past completion drifts slightly upward.
    That late drift is a long-horizon STABILITY property, not evidence the model had
    not converged -- yet an end-anchored rule reports "never settled" and throws the
    run away. Conflating the two cost us a full 5-seed run of branch E.

    So: smooth (median, kills dips) -> floor := min of the smoothed curve (immune to
    late drift) -> knee := first t whose crossing holds for `persist` consecutive
    steps (bounded persistence: rejects noise dips, ignores far-tail drift). Use
    curve_drift() to inspect the drift separately.

    Returns (knee, floor, dyn); dyn = smoothed[t_min]/floor is the curve's dynamic
    range and is a REQUIRED validity gate: a model that never learned yields a nearly
    flat curve on which every point sits inside the tolerance band, so the rule would
    report knee = t_min and dress a training failure up as instant convergence. A knee
    is meaningful only when the curve actually falls (dyn >> 1); callers must check
    `dyn >= MIN_DYN`. `knee` is None when no crossing persists.
    """
    if not curve:
        return None, float("nan"), float("nan")
    s = smooth_median(curve, win)
    Ts = sorted(s)
    floor = min(s.values())
    dyn = (s[Ts[0]] / floor) if floor > 0 else float("inf")
    thresh = (1.0 + tol) * floor
    ok = [s[t] <= thresh for t in Ts]
    need = min(persist, len(Ts))
    knee = None
    for i in range(len(Ts)):
        if all(ok[i:i + need]):
            knee = Ts[i]
            break
    return knee, floor, dyn


def mean_std(xs):
    """std is NaN (not 0) for n<2: a single run has NO measured spread, and
    reporting 0.0 renders an unreplicated number as perfectly reproducible --
    the exact failure this audit exists to remove."""
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], float("nan")
    return st.mean(xs), st.stdev(xs)


def agg(xs):
    """mean/std/n/spread block for JSON reporting across seeds.

    `n` travels with every number so downstream tables/plots can refuse to draw
    an error bar that was never measured. `single_run` is an explicit banner:
    any value carrying it is NOT reproducibility evidence.
    """
    m, s = mean_std(xs)
    return {"mean": m, "std": s, "n": len(xs), "vals": list(xs),
            "single_run": len(xs) < 2,
            "spread_pct": (100.0 * (max(xs) - min(xs)) / m) if len(xs) > 1 and m else float("nan")}


def neighbor_min8(d):
    """8-connected (Chebyshev) min over neighbors: matches a 3x3 NCA causal ring."""
    p = F.pad(d, (1, 1, 1, 1), value=INF)
    best = p[:, :, 1:-1, 1:-1].clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            best = torch.minimum(best, p[:, :, 1 + dy:1 + dy + d.shape[-2], 1 + dx:1 + dx + d.shape[-1]])
    return best

def bfs8(seed, obstacle, max_iter):
    d = torch.where(seed > 0, torch.zeros_like(seed), torch.full_like(seed, INF))
    d = torch.where(obstacle > 0, torch.full_like(seed, INF), d)
    for _ in range(max_iter):
        nb = neighbor_min8(d) + 1
        d_new = torch.minimum(d, nb)
        d_new = torch.where(obstacle > 0, torch.full_like(d, INF), d_new)
        if torch.equal(d_new, d):
            break
        d = d_new
    return d

def make_comb(n, H, W, device, teeth, noise, gen):
    """Serpentine comb: `teeth` full-width walls with gaps alternating end-to-end,
    forcing a snake path. Light per-scene jitter so the NCA learns a family, not one."""
    obs = (torch.rand(n, 1, H, W, generator=gen, device=device) < noise).float()
    gw = max(2, W // 5)
    for i in range(teeth):
        base = int((i + 1) * H / (teeth + 1))
        jit = torch.randint(-1, 2, (n,), generator=gen, device=device)
        gjit = torch.randint(-2, 3, (n,), generator=gen, device=device)
        for b in range(n):
            r = min(H - 2, max(1, base + int(jit[b])))
            obs[b, 0, r, :] = 1.0
            if i % 2 == 0:                          # gap at left end
                lo = max(0, 0 + int(gjit[b])); obs[b, 0, r, lo:lo + gw] = 0.0
            else:                                   # gap at right end
                hi = min(W, W + int(gjit[b])); obs[b, 0, r, hi - gw:hi] = 0.0
    seed = torch.zeros(n, 1, H, W, device=device)
    # seed in the TOP strip, on the end opposite the first gap, to maximise the snake
    for b in range(n):
        col = W - 1 - int(torch.randint(0, gw, (1,), generator=gen, device=device))
        row = int(torch.randint(0, 2, (1,), generator=gen, device=device))
        if obs[b, 0, row, col] > 0:
            col = max(0, col - gw)
        seed[b, 0, row, col] = 1.0
        obs[b, 0, row, col] = 0.0
    return seed, obs

def make_maze(n, H, W, device, gen, walls=(1, 8), noise=0.05, min_reach=0.20, tries=8):
    """Randomised wall-gap maze -- the project's canonical scene family.

    WHY THIS REPLACES make_comb. make_comb put its walls at deterministic rows
    (i+1)*H/(teeth+1) with +-1px jitter, alternated the gaps left/right deterministically with
    +-2px jitter, and always seeded in the top strip at the right end. The whole dataset
    therefore held ONE topology per `teeth` value -- about six in total. A net with 5.7k
    parameters can memorise six distance fields, and a memorised field is not propagation,
    which is the only thing a step-budget law is about. Measured directly: comb at G=64 hits
    NMSE floor 0.011 in 1500 iters while this generator, same net, same budget, sits at 0.354.

    Every scene here draws its own wall count, per-wall orientation, position, gap centre and
    gap width, plus its own seed cell, so tortuosity arrives as a natural per-scene spread
    rather than as six discrete settings. tau is measured per scene (see build_pool_maze) and
    binned post hoc, instead of being dialled in.

    The seed is resampled for up to `tries` batched rounds to seek at least `min_reach` of
    the scene's free cells. A scene that still misses the threshold after the final round is
    retained rather than silently discarded.
    """
    wlo, whi = walls
    obs = (torch.rand(n, 1, H, W, generator=gen, device=device) < noise).float()
    nw = torch.randint(wlo, whi + 1, (n,), generator=gen, device=device)
    for b in range(n):
        for _ in range(int(nw[b])):
            horiz = bool(torch.rand(1, generator=gen, device=device) < 0.5)
            p = int(torch.randint(3, (H if horiz else W) - 3, (1,), generator=gen, device=device))
            gc = int(torch.randint(0, W if horiz else H, (1,), generator=gen, device=device))
            gw = int(torch.randint(max(2, W // 10), max(3, W // 4), (1,), generator=gen, device=device))
            lo, hi = max(0, gc - gw // 2), min(W if horiz else H, gc + gw // 2)
            if horiz:
                obs[b, 0, p, :] = 1.0; obs[b, 0, p, lo:hi] = 0.0
            else:
                obs[b, 0, :, p] = 1.0; obs[b, 0, lo:hi, p] = 0.0

    free = (obs < 0.5).float()
    seed = torch.zeros(n, 1, H, W, device=device)
    need = torch.ones(n, dtype=torch.bool, device=device)
    for _ in range(tries):
        if not need.any():
            break
        for b in need.nonzero(as_tuple=False).flatten().tolist():
            fc = (free[b, 0] > 0).nonzero(as_tuple=False)
            pick = fc[torch.randint(0, len(fc), (1,), generator=gen, device=device)][0]
            seed[b, 0] = 0.0
            seed[b, 0, pick[0], pick[1]] = 1.0
        d = bfs8(seed, obs, max_iter=8 * (H + W))
        frac = ((d < INF / 2).float() * free).sum(dim=(1, 2, 3)) / free.sum(dim=(1, 2, 3)).clamp(min=1)
        need = frac < min_reach
    return seed, obs


def build_pool_maze(n, H, W, device, gen, walls=(1, 8), noise=0.05):
    """Pool on the randomised maze family. Returns per-scene rulers AND per-scene tau.

    All three rulers are measured to THE SAME cell -- the geodesically farthest reachable one
    -- because they are competing predictors of the same event: when that cell finally
    receives its information. (The old BF.build_pool took D_euc as the max Euclidean distance
    over reachable cells, i.e. to a DIFFERENT cell than D_geo referred to, so its Euclidean
    ruler was not comparable with branch E's. Fixed here by construction.)
    """
    seed, obs = make_maze(n, H, W, device, gen, walls=walls, noise=noise)
    d = bfs8(seed, obs, max_iter=8 * (H + W))
    reach = (d < INF / 2).float()
    Dnorm = 2.0 * max(H, W)
    tgt = torch.clamp(torch.where(reach > 0, d, torch.full_like(d, Dnorm)), max=Dnorm) / Dnorm
    Dgeo, Dfree, Deuc = [], [], []
    for b in range(n):
        rd = d[b, 0]
        far = (rd * reach[b, 0]).argmax()
        fy, fx = int(far // W), int(far % W)
        Dgeo.append(rd[fy, fx].item())
        sy, sx = (seed[b, 0] > 0).nonzero(as_tuple=False)[0]
        Dfree.append(float(max(abs(fy - int(sy)), abs(fx - int(sx)))))   # Chebyshev, same cell
        Deuc.append(float(((fy - int(sy)) ** 2 + (fx - int(sx)) ** 2) ** 0.5))  # L2, same cell
    Dgeo = torch.tensor(Dgeo, dtype=torch.float)
    Dfree = torch.tensor(Dfree, dtype=torch.float)
    Deuc = torch.tensor(Deuc, dtype=torch.float)
    tau = Dgeo / Dfree.clamp(min=1.0)                 # tortuosity, OBSERVED not dialled in
    inp = torch.cat([seed, obs], 1)
    loss_mask = reach * (obs < 0.5).float()
    return inp, tgt, loss_mask, Dgeo, Dfree, Deuc, tau


def build_pool(n, H, W, device, teeth, noise, gen):
    """DEPRECATED scene family (near-degenerate: ~6 topologies). Use build_pool_maze."""
    seed, obs = make_comb(n, H, W, device, teeth, noise, gen)
    d = bfs8(seed, obs, max_iter=8 * (H + W))
    reach = (d < INF / 2).float()
    Dnorm = 2.0 * max(H, W)
    tgt = torch.clamp(torch.where(reach > 0, d, torch.full_like(d, Dnorm)), max=Dnorm) / Dnorm
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    Dgeo, Deuc, Dfree = [], [], []
    for b in range(n):
        rd = d[b, 0]
        far = (rd * reach[b, 0]).argmax()            # farthest reachable cell (geodesic)
        fy, fx = far // W, far % W
        Dgeo.append(rd[fy, fx].item())
        sy, sx = (seed[b, 0] > 0).nonzero(as_tuple=False)[0]
        # Chebyshev free-space distance to that same cell (wall-blind comparator)
        Dfree.append(max(abs(int(fy) - int(sy)), abs(int(fx) - int(sx))))
        eu = torch.sqrt((yy - sy).float() ** 2 + (xx - sx).float() ** 2)
        Deuc.append((eu * reach[b, 0]).max().item())
    inp = torch.cat([seed, obs], 1)
    # AUDIT FIX (mask unification): score REACHABLE free cells only, matching
    # branch E. Unreachable free cells carry a constant sentinel target (Dnorm),
    # so including them let the model bank easy loss on a constant region --
    # diluting the NMSE and biasing the knee early. `reach` is returned so
    # callers can build eval masks from the same definition.
    loss_mask = reach * (obs < 0.5).float()
    return (inp, tgt, loss_mask, torch.tensor(Dgeo, dtype=torch.float),
            torch.tensor(Deuc, dtype=torch.float), torch.tensor(Dfree, dtype=torch.float))

class NCA(nn.Module):
    def __init__(self, C=12, hidden=96, in_ch=2):
        """in_ch = number of static input planes concatenated to the state each step
        (default 2 = [seed, obstacle]). Branch L passes 3 to add a medium field."""
        super().__init__()
        self.C = C
        self.in_ch = in_ch
        self.perceive = nn.Conv2d(C + in_ch, 3 * (C + in_ch), 3, padding=1,
                                  groups=C + in_ch, bias=False)
        self.w1 = nn.Conv2d(3 * (C + in_ch), hidden, 1)
        self.w2 = nn.Conv2d(hidden, C, 1)
        nn.init.zeros_(self.w2.weight); nn.init.zeros_(self.w2.bias)
        self.readout = nn.Conv2d(C, 1, 1)

    def _step(self, state, inp):
        return state + self.w2(F.relu(self.w1(self.perceive(torch.cat([state, inp], 1)))))

    def run(self, inp, T, collect=None):
        B, _, H, W = inp.shape
        state = torch.zeros(B, self.C, H, W, device=inp.device)
        outs = {}
        for t in range(1, T + 1):
            state = self._step(state, inp)
            if collect and t in collect:
                outs[t] = self.readout(state)
        return self.readout(state) if collect is None else outs

    def run_train(self, inp, T, seg=24):
        """Checkpointed unroll -> activation memory ~O(seg) not O(T); enables deep T."""
        B, _, H, W = inp.shape
        state = torch.zeros(B, self.C, H, W, device=inp.device)
        def seg_fn(s, n):
            for _ in range(int(n.item())):
                s = self._step(s, inp)
            return s
        t = 0
        while t < T:
            n = min(seg, T - t)
            state = checkpoint(seg_fn, state, torch.tensor(float(n)), use_reentrant=False)
            t += n
        return self.readout(state)

def abs_knee(curve, theta=0.05, persist=10, win=5):
    """First T where the smoothed error reaches an ABSOLUTE target and stays there.

    Returns (knee, floor). knee is None if the run never reaches `theta` -- that is a
    non-convergent run and must be reported as one, not handed a knee.

    WHY NOT A RELATIVE BAND. robust_knee accepts `err <= 1.05 * floor`, i.e. it measures "when
    does this run stop improving", each run judged against its own floor. A run that fits
    badly has a high floor and so gets a loose absolute bar and an early crossing. Across
    branch E's 14 seeded runs this produced r(floor, c1_geo) = -0.71: the worse the fit, the
    earlier the reported knee, which is exactly the wrong direction and drove the apparent
    c1 decline across grids. A budget law is about "when is the answer right", which is an
    absolute question, and every run must be asked the same one.

    Pick `theta` once, globally, and hold it across grids/levels/seeds. `persist` guards
    against a lucky dip, and smoothing plus the bounded window keep late-horizon drift from
    invalidating a run that genuinely converged (see robust_knee's history).

    Persistence is STRICT: a crossing in the last `persist`-1 steps of the curve returns None,
    because there is no room left to observe that it holds. robust_knee instead let the
    ok-window truncate at the end of the curve, so a lucky dip on the final step counted as a
    sustained crossing. Do not claim persistence you could not have measured; the cap is
    2x D_geo, so a genuine convergence has room to prove itself.
    """
    s = smooth_median(curve, win)
    Ts = sorted(s)
    floor = min(s.values())
    ok = [s[t] <= theta for t in Ts]
    need = min(persist, len(Ts))
    knee, run = None, 0
    for i in range(len(Ts) - 1, -1, -1):          # O(T): suffix run-lengths of consecutive ok
        run = run + 1 if ok[i] else 0
        if run >= need:
            knee = Ts[i]                          # scanning backwards, the last write is the earliest
    return knee, floor


def masked_nmse(pred, tgt, m, var):
    return (((pred - tgt) ** 2) * m).sum() / (m.sum() + 1e-9) / var

def tail_nmse(pred, tgt, m, var, q=0.95):
    """Per-scene q-quantile of per-cell squared error over masked cells, normalised by var.

    Why not the mean. The rulers under test (D_geo, D_euc) are MAX eccentricities: they predict
    when the FARTHEST cell receives its information. Mean error saturates once MOST cells are
    right, which is a strictly earlier and different event -- a skewed distance distribution
    makes the gap large. Scoring the knee on the mean therefore measures a different cell
    population than the ruler describes, and reads back a knee well below the ruler.

    A high quantile asks "is any cell still badly wrong?", which is what a SUFFICIENT budget
    has to guarantee, and unlike the max it is not decided by a single outlier cell. Cells are
    selected by ERROR, never by distance, so no candidate ruler is favoured by the choice.

    Returns one value per scene, shape (B,) -- same contract as the per-scene mean it replaces.
    """
    B = pred.shape[0]
    e = ((pred - tgt) ** 2).reshape(B, -1)
    keep = m.reshape(B, -1) > 0
    e = torch.where(keep, e, torch.full_like(e, -1.0))   # squared errors are >=0, so unmasked
    es, _ = torch.sort(e, dim=1)                          # cells sort to the front and are skipped
    M = keep.sum(1)                                       # masked cells per scene
    N = e.shape[1]
    idx = ((N - M) + (q * (M - 1).clamp(min=0)).long()).clamp(max=N - 1)
    return es.gather(1, idx.unsqueeze(1)).squeeze(1) / var

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--teeth", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--iters", type=int, default=450)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results_F.json")
    a = ap.parse_args()
    dev = get_device(a.device)
    H = W = a.grid
    res = {"grid": a.grid, "noise": a.noise, "connectivity": 8, "seeds": a.seeds,
           "mask": "reachable_free", "knee": "sustained-crossing", "levels": []}
    print(f"Branch F: tortuosity sweep (8-conn/Chebyshev)  grid={a.grid}  dev={dev}  seeds={a.seeds}")
    for nt in a.teeth:
        # data pool is fixed across seeds -> the spread we report is pure
        # model-init/training variance, which is what the audit flagged.
        gen = torch.Generator(device=dev).manual_seed(100 + nt)
        inp, tgt, lm, Dgeo, Deuc, Dfree = build_pool(a.pool, H, W, dev, nt, a.noise, gen)
        mean_geo = Dgeo.mean().item()
        T_train = int(1.6 * mean_geo) + 10
        var = tgt[lm > 0].var().item() + 1e-12
        gen2 = torch.Generator(device=dev).manual_seed(900 + nt)
        vinp, vtgt, vlm, vDgeo, vDeuc, vDfree = build_pool(256, H, W, dev, nt, a.noise, gen2)
        vvar = vtgt[vlm > 0].var().item() + 1e-12
        vgeo, veuc, vfree = vDgeo.mean().item(), vDeuc.mean().item(), vDfree.mean().item()
        tau = (vDgeo / vDfree.clamp(min=1)).mean().item()
        Tlist = list(range(1, T_train + 1))

        knees, c1g, c1f, c1e, floors = [], [], [], [], []
        for sd in a.seeds:
            set_seed(sd)                                   # AUDIT FIX: seeded init
            model = NCA().to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            npool = inp.shape[0]; t0 = time.time()
            for _ in range(a.iters):
                idx = torch.randint(0, npool, (a.batch,), device=dev)
                out = model.run_train(inp[idx], T_train, seg=a.seg)
                loss = masked_nmse(out, tgt[idx], lm[idx], var)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            with torch.no_grad():
                outs = model.run(vinp, T_train, collect=set(Tlist))
                curve = {t: masked_nmse(outs[t], vtgt, vlm, vvar).item() for t in Tlist}
            knee, floor, dyn = robust_knee(curve)
            if knee is None or dyn < MIN_DYN:
                why = ("never settled below threshold within T_train" if knee is None
                       else f"curve too flat to have a knee (dyn={dyn:.2f} < {MIN_DYN})")
                print(f"  teeth={nt} seed={sd}  INVALID: {why}; excluded  [{time.time()-t0:.0f}s]")
                continue
            knees.append(knee); floors.append(floor)
            c1g.append(knee / vgeo); c1f.append(knee / vfree); c1e.append(knee / veuc)
            print(f"  teeth={nt} seed={sd}  tau={tau:.2f}  T*={knee:3d}"
                  f"  c1_geo={knee/vgeo:.2f}  floor={floor:.3f}  dyn={dyn:.1f}  [{time.time()-t0:.0f}s]")
        if not knees:
            print(f"  teeth={nt}: ALL seeds invalid — skipping level")
            continue

        L = {"teeth": nt, "tau": tau, "mean_Dgeo": vgeo, "mean_Deuc": veuc, "mean_Dfree": vfree,
             "T_knee": agg(knees), "c1_geo": agg(c1g), "c1_free": agg(c1f), "c1_euc": agg(c1e),
             "floor": agg(floors)}
        res["levels"].append(L)
        print(f"  teeth={nt}  tau={tau:.2f}  D_geo={vgeo:5.1f} D_free={vfree:5.1f}"
              f"  T*={L['T_knee']['mean']:.1f}+-{L['T_knee']['std']:.1f}"
              f"  c1_geo={L['c1_geo']['mean']:.2f}+-{L['c1_geo']['std']:.2f}"
              f"  c1_free={L['c1_free']['mean']:.2f}+-{L['c1_free']['std']:.2f}")

    # Across-tortuosity invariance, now computed on per-seed MEANS.
    cg = [L["c1_geo"]["mean"] for L in res["levels"]]
    cf = [L["c1_free"]["mean"] for L in res["levels"]]
    taus = [L["tau"] for L in res["levels"]]
    res["c1_geo_mean"] = st.mean(cg); res["c1_geo_cv"] = st.pstdev(cg) / st.mean(cg)
    res["c1_free_mean"] = st.mean(cf); res["c1_free_cv"] = st.pstdev(cf) / st.mean(cf)
    # within-level seed noise: the floor below which any cross-level CV is meaningless
    res["seed_cv_within_level"] = st.mean(
        [L["c1_geo"]["std"] / L["c1_geo"]["mean"] for L in res["levels"] if L["c1_geo"]["mean"]])
    pred_cf = [res["c1_geo_mean"] * t for t in taus]
    res["free_pred_resid_pct"] = st.mean([abs(p - c) / c * 100 for p, c in zip(pred_cf, cf)])
    print(f"\nc1_geo = {res['c1_geo_mean']:.2f}  (cross-tortuosity CV={res['c1_geo_cv']*100:.0f}%  <- want SMALL)")
    print(f"c1_free= {res['c1_free_mean']:.2f}  (cross-tortuosity CV={res['c1_free_cv']*100:.0f}%  <- want LARGE)")
    print(f"within-level SEED CV = {res['seed_cv_within_level']*100:.0f}%  <- noise floor; "
          f"cross-tortuosity CV must clear this to mean anything")
    print(f"straight-line constant predicted as c1_geo*tau: residual {res['free_pred_resid_pct']:.0f}%")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()

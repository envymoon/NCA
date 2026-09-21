import argparse, json, time
import statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

INF = 1e9


MIN_DYN = 2.0


def get_device(pref="auto"):
    if pref == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not visible"
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def smooth_median(curve, win=5):
    Ts = sorted(curve)
    h = win // 2
    out = {}
    for i, t in enumerate(Ts):
        lo, hi = max(0, i - h), min(len(Ts), i + h + 1)
        out[t] = st.median([curve[u] for u in Ts[lo:hi]])
    return out


def curve_drift(curve, win=5):
    s = smooth_median(curve, win)
    Ts = sorted(s)
    floor = min(s.values())
    return (s[Ts[-1]] / floor) if floor > 0 else float("nan")


def robust_knee(curve, tol=0.05, persist=10, win=5):
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
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], float("nan")
    return st.mean(xs), st.stdev(xs)


def agg(xs):
    m, s = mean_std(xs)
    return {"mean": m, "std": s, "n": len(xs), "vals": list(xs),
            "single_run": len(xs) < 2,
            "spread_pct": (100.0 * (max(xs) - min(xs)) / m) if len(xs) > 1 and m else float("nan")}


def neighbor_min8(d):
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
    obs = (torch.rand(n, 1, H, W, generator=gen, device=device) < noise).float()
    gw = max(2, W // 5)
    for i in range(teeth):
        base = int((i + 1) * H / (teeth + 1))
        jit = torch.randint(-1, 2, (n,), generator=gen, device=device)
        gjit = torch.randint(-2, 3, (n,), generator=gen, device=device)
        for b in range(n):
            r = min(H - 2, max(1, base + int(jit[b])))
            obs[b, 0, r, :] = 1.0
            if i % 2 == 0:
                lo = max(0, 0 + int(gjit[b])); obs[b, 0, r, lo:lo + gw] = 0.0
            else:
                hi = min(W, W + int(gjit[b])); obs[b, 0, r, hi - gw:hi] = 0.0
    seed = torch.zeros(n, 1, H, W, device=device)

    for b in range(n):
        col = W - 1 - int(torch.randint(0, gw, (1,), generator=gen, device=device))
        row = int(torch.randint(0, 2, (1,), generator=gen, device=device))
        if obs[b, 0, row, col] > 0:
            col = max(0, col - gw)
        seed[b, 0, row, col] = 1.0
        obs[b, 0, row, col] = 0.0
    return seed, obs

def make_maze(n, H, W, device, gen, walls=(1, 8), noise=0.05, min_reach=0.20, tries=8):
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
        Dfree.append(float(max(abs(fy - int(sy)), abs(fx - int(sx)))))
        Deuc.append(float(((fy - int(sy)) ** 2 + (fx - int(sx)) ** 2) ** 0.5))
    Dgeo = torch.tensor(Dgeo, dtype=torch.float)
    Dfree = torch.tensor(Dfree, dtype=torch.float)
    Deuc = torch.tensor(Deuc, dtype=torch.float)
    tau = Dgeo / Dfree.clamp(min=1.0)
    inp = torch.cat([seed, obs], 1)
    loss_mask = reach * (obs < 0.5).float()
    return inp, tgt, loss_mask, Dgeo, Dfree, Deuc, tau


def build_pool(n, H, W, device, teeth, noise, gen):
    seed, obs = make_comb(n, H, W, device, teeth, noise, gen)
    d = bfs8(seed, obs, max_iter=8 * (H + W))
    reach = (d < INF / 2).float()
    Dnorm = 2.0 * max(H, W)
    tgt = torch.clamp(torch.where(reach > 0, d, torch.full_like(d, Dnorm)), max=Dnorm) / Dnorm
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    Dgeo, Deuc, Dfree = [], [], []
    for b in range(n):
        rd = d[b, 0]
        far = (rd * reach[b, 0]).argmax()
        fy, fx = far // W, far % W
        Dgeo.append(rd[fy, fx].item())
        sy, sx = (seed[b, 0] > 0).nonzero(as_tuple=False)[0]

        Dfree.append(max(abs(int(fy) - int(sy)), abs(int(fx) - int(sx))))
        eu = torch.sqrt((yy - sy).float() ** 2 + (xx - sx).float() ** 2)
        Deuc.append((eu * reach[b, 0]).max().item())
    inp = torch.cat([seed, obs], 1)


    loss_mask = reach * (obs < 0.5).float()
    return (inp, tgt, loss_mask, torch.tensor(Dgeo, dtype=torch.float),
            torch.tensor(Deuc, dtype=torch.float), torch.tensor(Dfree, dtype=torch.float))

class NCA(nn.Module):
    def __init__(self, C=12, hidden=96, in_ch=2):
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
    s = smooth_median(curve, win)
    Ts = sorted(s)
    floor = min(s.values())
    ok = [s[t] <= theta for t in Ts]
    need = min(persist, len(Ts))
    knee, run = None, 0
    for i in range(len(Ts) - 1, -1, -1):
        run = run + 1 if ok[i] else 0
        if run >= need:
            knee = Ts[i]
    return knee, floor


def masked_nmse(pred, tgt, m, var):
    return (((pred - tgt) ** 2) * m).sum() / (m.sum() + 1e-9) / var

def tail_nmse(pred, tgt, m, var, q=0.95):
    B = pred.shape[0]
    e = ((pred - tgt) ** 2).reshape(B, -1)
    keep = m.reshape(B, -1) > 0
    e = torch.where(keep, e, torch.full_like(e, -1.0))
    es, _ = torch.sort(e, dim=1)
    M = keep.sum(1)
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
            set_seed(sd)
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


    cg = [L["c1_geo"]["mean"] for L in res["levels"]]
    cf = [L["c1_free"]["mean"] for L in res["levels"]]
    taus = [L["tau"] for L in res["levels"]]
    res["c1_geo_mean"] = st.mean(cg); res["c1_geo_cv"] = st.pstdev(cg) / st.mean(cg)
    res["c1_free_mean"] = st.mean(cf); res["c1_free_cv"] = st.pstdev(cf) / st.mean(cf)

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

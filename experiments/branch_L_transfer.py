"""
Branch L — does the budgeting law TRANSFER off the distance task? (breaking circularity)

THE PROBLEM THIS EXISTS TO FIX. Every branch so far trains the NCA to compute the geodesic
distance field, and the a-priori budget predictor is a BFS over the obstacle mask. But that
same BFS *already computes the answer*. On the distance task the "budget calculator" is the
solver, so the tool is circular: nobody would train a 200-step NCA to reproduce what the
predictor handed them for free. A reviewer will notice immediately.

The law only escapes circularity in the regime where the propagation GEOMETRY is cheap but
the VALUE is expensive. So we need a task where:
  * the free-space graph (hence D_geo) still comes from one cheap mask-only BFS, and
  * the target value is NOT recoverable from that BFS.

TASK: weighted-geodesic potential in a random medium. Each free cell carries a traversal
cost w(x) ~ U[1-het, 1+het]; V(c) = min over paths seed->c of the summed cost. Ground truth
is exact min-plus Bellman-Ford (no learned supervision), consistent with the rest of the
project. The mask-only BFS sees only WHICH cells connect, never w, so it cannot produce V.

THE KNOB. `het` interpolates continuously between the two regimes:
    het = 0    -> w == 1 -> V IS the hop-count distance  -> BFS solves it   (circular limit)
    het > 0    -> V depends on the medium                -> BFS cannot      (the useful regime)
This makes "geometry cheap, value expensive" a dial we turn, not an excuse we write.

WHAT IS AT RISK. The mask-only budget D_geo counts HOPS ignoring w. But the optimal weighted
path may DETOUR through cheap cells and so take MORE hops than the unweighted geodesic. Define
H* = max over reachable cells of the hop-length of that cell's optimal weighted path. The
causal floor is H*, not D_geo, and H* >= D_geo by construction. If H*/D_geo grows with het,
then the cheap mask-only budget UNDER-provisions in a heterogeneous medium and the law needs
an amendment -- exactly the kind of failure the Euclidean ruler was retracted for. We measure
H*/D_geo rather than assume it is 1.

Reported per het level:
  * T*  : measured NCA knee (anytime-trained, ONE GLOBAL cap across het levels)
  * D_geo : mask-only BFS eccentricity        <- the cheap a-priori budget (the tool)
  * H*    : hop-length of optimal weighted paths <- the true causal floor
  * sweeps_BF : Bellman-Ford sweeps to fixed point <- method-independent reference budget
  * ruler discrimination: per-scene CV of T*/D_geo vs T*/D_free (Chebyshev) vs T*/D_euc (L2)

Falsifiable outcomes:
  (a) T* <= D_geo at every het  -> the cheap mask-only budget stays SUFFICIENT off the
      distance task -> the tool is real and non-circular.
  (b) T* > D_geo once het > 0   -> mask-only budgeting under-provisions in a medium; the
      law is distance-task-specific and must be narrowed. Report it.

Run: python3 branch_L_transfer.py --hets 0.0 0.3 0.6 0.9
"""
import argparse, json, time
import statistics as st
import torch, torch.nn.functional as F
import branch_F_tortuosity as BF

INF = BF.INF


def shift8(x):
    """Stack the 8 Chebyshev-neighbour shifts of x -> (8,B,1,H,W), padded with INF."""
    H, W = x.shape[-2], x.shape[-1]
    p = F.pad(x, (1, 1, 1, 1), value=INF)
    outs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            outs.append(p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W])
    return torch.stack(outs)


def bf_weighted8(seed, obs, w, max_iter):
    """Exact min-plus Bellman-Ford on the 8-connected free-space graph.

    Entering cell x costs w(x); V(seed)=0. Also tracks, for each cell, the HOP LENGTH of the
    optimal path reaching it (taken from the arg-min predecessor), and the number of sweeps
    to reach the fixed point. Returns (V, hop, sweeps).
    """
    V = torch.where(seed > 0, torch.zeros_like(seed), torch.full_like(seed, INF))
    V = torch.where(obs > 0, torch.full_like(seed, INF), V)
    hop = torch.where(seed > 0, torch.zeros_like(seed), torch.full_like(seed, INF))
    sweeps = 0
    for it in range(max_iter):
        cand_v = shift8(V)                       # (8,B,1,H,W)
        cand_h = shift8(hop)
        best, idx = cand_v.min(dim=0)            # best predecessor by VALUE
        best_hop = cand_h.gather(0, idx.unsqueeze(0)).squeeze(0)
        nv = best + w                            # cost of entering this cell
        nh = best_hop + 1
        upd = nv < V
        Vn = torch.where(upd, nv, V)
        Hn = torch.where(upd, nh, hop)
        Vn = torch.where(obs > 0, torch.full_like(V, INF), Vn)
        Hn = torch.where(obs > 0, torch.full_like(hop, INF), Hn)
        if torch.equal(Vn, V):
            break
        V, hop = Vn, Hn
        sweeps = it + 1
    return V, hop, sweeps


def build_pool_medium(n, H, W, dev, het, gen, walls=(1, 8), noise=0.05):
    """Randomised maze + random medium. Target = weighted-geodesic potential (NOT the hop
    distance unless het==0). D_geo comes from a mask-only BFS that never sees w.

    WHY make_maze AND NOT make_comb. The comb family holds about six obstacle topologies in
    total (walls at deterministic rows, gaps alternating deterministically, seed always in the
    same corner). A 5.7k-parameter net can memorise six fields, and a memorised field is not
    propagation -- which is the only thing a step-budget law is about. Measured: same net, same
    budget, G=64 -- comb reaches NMSE 0.011 while the randomised family sits at 0.354. Every
    comb-based number in this project measures memorisation. See BF.make_maze.

    RULERS ARE MEASURED TO THE SAME CELL. D_geo, D_free and D_euc are competing predictors of
    ONE event, so they must refer to one cell: the mask-only geodesically farthest reachable
    one. (The old code took D_euc as the max Euclidean distance over reachable cells -- a
    DIFFERENT cell than D_geo referred to -- which made it incomparable with branch E's.)

    H* is deliberately NOT tied to that cell: it is the causal floor, the max over ALL cells of
    the hop length of the optimal WEIGHTED path, which the medium can push onto another cell.
    """
    seed, obs = BF.make_maze(n, H, W, dev, gen, walls=walls, noise=noise)
    # medium: w ~ U[1-het, 1+het] on free cells (obstacles are impassable, w irrelevant)
    w = 1.0 + het * (2.0 * torch.rand(n, 1, H, W, generator=gen, device=dev) - 1.0)
    V, hop, sweeps = bf_weighted8(seed, obs, w, max_iter=8 * (H + W))
    d_hop = BF.bfs8(seed, obs, max_iter=8 * (H + W))      # MASK-ONLY: the cheap predictor
    reach = (V < INF / 2).float()

    Vnorm = 2.0 * max(H, W) * (1.0 + het)                 # analytic normaliser (no test-stat leak)
    tgt = torch.clamp(torch.where(reach > 0, V, torch.full_like(V, Vnorm)), max=Vnorm) / Vnorm
    loss_mask = reach * (obs < 0.5).float()

    Dgeo, Dfree, Deuc, Hstar = [], [], [], []
    for b in range(n):
        rm = reach[b, 0]
        dh = d_hop[b, 0]
        far = (dh * rm).argmax()
        fy, fx = int(far // W), int(far % W)
        Dgeo.append(dh[fy, fx].item())                    # mask-only eccentricity (the budget)
        Hstar.append((hop[b, 0] * rm).max().item())       # true causal floor (optimal-path hops)
        sy, sx = (seed[b, 0] > 0).nonzero(as_tuple=False)[0]
        Dfree.append(float(max(abs(fy - int(sy)), abs(fx - int(sx)))))          # Chebyshev, same cell
        Deuc.append(float(((fy - int(sy)) ** 2 + (fx - int(sx)) ** 2) ** 0.5))  # L2, same cell
    # medium plane is CENTRED so that het=0 hands the net an all-zero (information-free) plane
    inp = torch.cat([seed, obs, w - 1.0], 1)
    Dgeo = torch.tensor(Dgeo, dtype=torch.float)
    Dfree = torch.tensor(Dfree, dtype=torch.float)
    tau = Dgeo / Dfree.clamp(min=1.0)                     # tortuosity, OBSERVED not dialled in
    return (inp, tgt, loss_mask, Dgeo, Dfree, torch.tensor(Deuc, dtype=torch.float),
            torch.tensor(Hstar, dtype=torch.float), tau, sweeps)


def cv(xs):
    return st.pstdev(xs) / st.mean(xs) if xs and st.mean(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--hets", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--test_per", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--walls", type=int, nargs=2, default=[1, 8],
                    help="per-scene wall count is drawn uniformly from this range; the "
                         "diversity knob. Tortuosity is OBSERVED per scene, never dialled in.")
    ap.add_argument("--theta", type=float, default=0.05,
                    help="ABSOLUTE masked-NMSE target defining the knee; one value for every "
                         "het level and seed. A band relative to each run's own floor gives "
                         "harder runs a looser bar and reads back an earlier knee -- across "
                         "branch E's 14 runs that artifact produced r(floor,c1)=-0.71.")
    ap.add_argument("--tail_q", type=float, default=0.95,
                    help="knee is measured on this per-scene quantile of per-cell error; the "
                         "rulers are MAX eccentricities, so the mean scores the wrong cells")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results_L.json")
    a = ap.parse_args()
    dev = BF.get_device(a.device)
    H = W = a.grid
    res = {"grid": a.grid, "hets": a.hets, "seeds": a.seeds, "task": "weighted-geodesic potential",
           "gt": "exact min-plus Bellman-Ford", "scenes": "randomised maze (BF.make_maze)",
           "walls": a.walls, "theta": a.theta,
           "knee": f"first T where p{a.tail_q*100:.0f} per-cell error reaches an ABSOLUTE "
                   f"NMSE<={a.theta} and stays there",
           "tail_q": a.tail_q, "levels": []}
    print(f"Branch L (transfer off the distance task)  grid={a.grid}  dev={dev}  seeds={a.seeds}"
          f"  theta={a.theta}  iters={a.iters}")

    pools = {het: build_pool_medium(a.pool, H, W, dev, het,
                                    torch.Generator(device=dev).manual_seed(20 + int(het * 100)),
                                    walls=tuple(a.walls))
             for het in a.hets}
    vpools = {het: build_pool_medium(a.test_per, H, W, dev, het,
                                     torch.Generator(device=dev).manual_seed(700 + int(het * 100)),
                                     walls=tuple(a.walls))
              for het in a.hets}
    # ONE GLOBAL CAP across het levels (a per-level cap would scale with the level)
    cap = int(2.0 * max(pools[h][3].max().item() for h in a.hets)) + 12
    res["global_cap"] = cap
    print(f"  GLOBAL anytime cap = {cap} (shared by all het levels)")

    for het in a.hets:
        inp, tgt, lm, Dgeo, Dfree, Deuc, Hstar, tau, sweeps = pools[het]
        var = tgt[lm > 0].var().item() + 1e-12
        vinp, vtgt, vlm, vDgeo, vDfree, vDeuc, vHstar, vtau, vsweeps = vpools[het]
        vvar = vtgt[vlm > 0].var().item() + 1e-12
        mg, mf, me = vDgeo.mean().item(), vDfree.mean().item(), vDeuc.mean().item()
        mh = vHstar.mean().item()
        detour = (vHstar / vDgeo.clamp(min=1)).mean().item()
        Tlist = list(range(1, cap + 1))
        npool = inp.shape[0]
        tq = torch.tensor([0.1, 0.5, 0.9])
        print(f"  het={het:.1f}: D_geo mean={mg:5.1f}  tau p10/p50/p90="
              f"{'/'.join(f'{v:.2f}' for v in torch.quantile(vtau, tq).tolist())}")
        # scenes whose target saturates at the Vnorm clamp: their far cells share the
        # unreachable-background value, so "when does the info arrive" is unmeasurable there.
        # Count them so the analysis can exclude/footnote them (branch E: 0.4-1.6% of scenes).
        vclamp = int(((((vtgt >= 1.0 - 1e-6).float() * vlm).sum(dim=(1, 2, 3))) > 0).sum())
        if vclamp:
            print(f"           target-clamp: {vclamp}/{vinp.shape[0]} val scenes saturate "
                  f"(exclude or footnote in analysis)")

        knees, c1g, c1f, c1e, cvg, cvf, cve = [], [], [], [], [], [], []
        for sd in a.seeds:
            BF.set_seed(sd)
            model = BF.NCA(in_ch=3).to(dev)            # [seed, obs, medium]
            opt = torch.optim.Adam(model.parameters(), 1e-3)
            t0 = time.time()
            for _ in range(a.iters):
                idx = torch.randint(0, npool, (a.batch,), device=dev)
                T = int(torch.randint(4, cap + 1, (1,)).item())      # ANYTIME
                out = model.run_train(inp[idx], T, seg=a.seg)
                loss = BF.masked_nmse(out, tgt[idx], lm[idx], var)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            # keep the trained weights (~30KB): any later metric-definition question
            # (different theta, different tail quantile) is then one forward pass, not a retrain
            torch.save(model.state_dict(), f"{a.out}.het{het}.s{sd}.pt")

            with torch.no_grad():
                curve = {t: 0.0 for t in Tlist}
                ps = {t: [] for t in Tlist}
                Nv = vinp.shape[0]
                for s in range(0, Nv, 32):
                    sl = slice(s, min(Nv, s + 32))
                    outs = model.run(vinp[sl], cap, collect=set(Tlist))
                    for t in Tlist:
                        # KNEE MEASURED ON THE TAIL, NOT THE MEAN. D_geo is a MAX eccentricity,
                        # so the event it predicts is "the farthest cell is finally right", not
                        # "most cells are right". The mean-NMSE knee scores the wrong cell
                        # population and reads back a knee below the ruler by construction --
                        # branch F2 already saw this. The training loss stays mean-NMSE; only
                        # the measurement changes, so the model is identical to the mean-knee
                        # run and the difference is attributable to the knee definition alone.
                        e = BF.tail_nmse(outs[t], vtgt[sl], vlm[sl], vvar, q=a.tail_q)
                        ps[t].append(e.cpu())
                        curve[t] += e.sum().item()
                curve = {t: curve[t] / Nv for t in Tlist}
                ps = {t: torch.cat(ps[t]) for t in Tlist}
            knee, floor = BF.abs_knee(curve, theta=a.theta)
            if knee is None:
                print(f"  het={het:.1f} seed={sd}  NON-CONVERGENT: never reached NMSE<={a.theta} "
                      f"within cap (floor={floor:.4f}); excluded  [{time.time()-t0:.0f}s]")
                continue
            if knee >= cap:
                print(f"  het={het:.1f} seed={sd}  knee at the cap ({knee}); horizon-confounded, "
                      f"excluded  [{time.time()-t0:.0f}s]")
                continue
            knees.append(knee); c1g.append(knee / mg); c1f.append(knee / mf); c1e.append(knee / me)
            g_, f_, e_ = [], [], []
            for i in range(vinp.shape[0]):
                ki, _ = BF.abs_knee({t: ps[t][i].item() for t in Tlist}, theta=a.theta)
                if ki is None or vDgeo[i] <= 2:
                    continue                    # scene never converged; not a knee
                g_.append(ki / vDgeo[i].item())
                f_.append(ki / max(vDfree[i].item(), 1.0))
                e_.append(ki / max(vDeuc[i].item(), 1.0))
            if len(g_) < 32:
                print(f"  het={het:.1f} seed={sd}  only {len(g_)} scenes converged; excluded "
                      f"[{time.time()-t0:.0f}s]")
                knees.pop(); c1g.pop(); c1f.pop(); c1e.pop()
                continue
            cvg.append(cv(g_)); cvf.append(cv(f_)); cve.append(cv(e_))
            print(f"  het={het:.1f} seed={sd}  T*={knee:3d}  D_geo={mg:.0f}  T*/D_geo={knee/mg:.2f}"
                  f"  per-scene CV: geo {cv(g_)*100:3.0f}% free {cv(f_)*100:3.0f}% "
                  f"euc {cv(e_)*100:3.0f}%  ({len(g_)} scenes)  [{time.time()-t0:.0f}s]")
        if not knees:
            print(f"  het={het:.1f}: ALL seeds invalid — skipping level")
            continue
        L = {"het": het, "mean_Dgeo": mg, "mean_Dfree": mf, "mean_Deuc": me, "mean_Hstar": mh,
             "mean_tau": vtau.mean().item(), "clamped_scenes": vclamp,
             "detour_ratio_Hstar_over_Dgeo": detour, "bf_sweeps": vsweeps,
             "T_knee": BF.agg(knees), "c1_geo": BF.agg(c1g), "c1_free": BF.agg(c1f),
             "c1_euc": BF.agg(c1e),
             "c1_geo_cv": BF.agg(cvg) if cvg else None,
             "c1_free_cv": BF.agg(cvf) if cvf else None,
             "c1_euc_cv": BF.agg(cve) if cve else None,
             "sufficient": bool(BF.agg(knees)["mean"] <= mg)}
        res["levels"].append(L)
        print(f"  het={het:.1f}  D_geo={mg:5.1f}  H*={mh:5.1f} (detour {detour:.2f}x)"
              f"  BF sweeps={vsweeps}  T*={L['T_knee']['mean']:.1f}+-{L['T_knee']['std']:.1f}"
              f"  T*/D_geo={L['c1_geo']['mean']:.2f}+-{L['c1_geo']['std']:.2f}"
              f"  SUFFICIENT={L['sufficient']}")

    if not res["levels"]:
        print("\n  NO VALID LEVELS — refusing to report.")
        json.dump(res, open(a.out, "w"), indent=2); return

    print(f"\n  ==== does the cheap mask-only budget survive off the distance task? ====")
    print(f"  {'het':>5s}  {'detour H*/D_geo':>15s}  {'T*/D_geo':>16s}  {'sufficient':>10s}"
          f"  {'CV geo':>8s}  {'CV free':>8s}  {'CV euc':>8s}")
    for L in res["levels"]:
        cg = f"{L['c1_geo_cv']['mean']*100:.0f}%" if L["c1_geo_cv"] else "n/a"
        cf = f"{L['c1_free_cv']['mean']*100:.0f}%" if L["c1_free_cv"] else "n/a"
        ce = f"{L['c1_euc_cv']['mean']*100:.0f}%" if L["c1_euc_cv"] else "n/a"
        print(f"  {L['het']:5.1f}  {L['detour_ratio_Hstar_over_Dgeo']:15.2f}"
              f"  {L['c1_geo']['mean']:8.2f}+-{L['c1_geo']['std']:.2f}"
              f"  {str(L['sufficient']):>10s}  {cg:>8s}  {cf:>8s}  {ce:>8s}")
    res["all_sufficient"] = all(L["sufficient"] for L in res["levels"])
    res["circularity_broken"] = any(L["het"] > 0 and L["sufficient"] for L in res["levels"])
    print(f"\n  het=0 is the CIRCULAR control (V == hop distance, BFS solves it).")
    print(f"  het>0 is the real test: BFS gives the graph but CANNOT give V.")
    print(f"  mask-only budget sufficient at every het? {res['all_sufficient']}")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

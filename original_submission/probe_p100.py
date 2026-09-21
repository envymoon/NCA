"""p100 / farthest-cell knee probe on the FROZEN eval pool (code-audit finding 3).

POST-FREEZE ADDENDUM (2026-07-17). Eval-only on frozen E s3/s4 checkpoints; writes ONLY
probe_p100.json / .log. Answers the audit's open question: the frozen p95 tail knee can
in principle exclude the <5% most eccentric cells -- the very cells that define D_geo.
With checkpoints on disk this is one forward pass per unit, no retraining.

POOL. The frozen eval pool itself (generator seed 1), unclamped rows only -- so the p95
numbers here are directly comparable to the frozen per-scene knees.

THREE KNEE DEFINITIONS on the same forward pass, same abs_knee machinery
(theta=0.05, persist=10, win=5, cap=152, pool-variance normalisation):
  p95      BF.tail_nmse q=0.95   (the frozen definition; reference)
  p100     BF.tail_nmse q=1.0    (max masked cell; outlier-sensitive by construction)
  farcell  normalised squared error at THE cell that defines D_geo (argmax of the
           BFS field over reachable cells) -- the ruler's own cell, no quantile.

READOUT FIXED BEFORE FIRST RUN, per grid x checkpoint:
  1. coverage under each definition (stricter definitions may censor more; report);
  2. median knee shift p95 -> p100 -> farcell on jointly-converged scenes;
  3. c1 = median knee/Dgeo under each definition -- does the budget constant move,
     and does it stay near the paper's affine story (T* ~ 0.87 D_geo + offset)?
  4. ruler ordering under p100 and farcell: is CV(T*/Dgeo) still < CV(T*/Dfree),
     CV(T*/Deuc) on converged rows? Gate: ordering unchanged in every cell.
The paper's V-B/V-C wording needs updating only if (3) moves c1 materially or (4) flips.

Run (cuda):
  cd experiments && python probe_p100.py
"""
import json
import math
import statistics as st

import torch

import branch_F_tortuosity as BF

DEV = "cuda"
CKPT = "results_E_maze_s34.json.G{G}.s{s}.pt"
CAP = 152
THETA = 0.05
SLICE = 64


def cv(xs):
    return st.stdev(xs) / st.mean(xs)


@torch.no_grad()
def knee_three_ways(model, inp, tgt, mask, var, far_idx):
    """far_idx: (N,) flat index of the D_geo-defining cell per scene."""
    Ts = list(range(1, CAP + 1))
    p95 = {t: [] for t in Ts}
    p100 = {t: [] for t in Ts}
    farc = {t: [] for t in Ts}
    N = inp.shape[0]
    for lo in range(0, N, SLICE):
        sl = slice(lo, min(lo + SLICE, N))
        outs = model.run(inp[sl], CAP, collect=set(Ts))
        fidx = far_idx[sl]
        for t in Ts:
            p95[t].append(BF.tail_nmse(outs[t], tgt[sl], mask[sl], var, q=0.95).cpu())
            p100[t].append(BF.tail_nmse(outs[t], tgt[sl], mask[sl], var, q=1.0).cpu())
            e = ((outs[t] - tgt[sl]) ** 2).flatten(1)
            farc[t].append((e.gather(1, fidx.unsqueeze(1)).squeeze(1) / var).cpu())
    for d in (p95, p100, farc):
        for t in Ts:
            d[t] = torch.cat(d[t])
    out = {}
    for name, d in (("p95", p95), ("p100", p100), ("farcell", farc)):
        out[name] = [BF.abs_knee({t: d[t][i].item() for t in Ts}, theta=THETA)[0]
                     for i in range(N)]
    return out


def main():
    assert torch.cuda.is_available()
    out = {"protocol": {"theta": THETA, "cap": CAP, "pool_gen_seed": 1}, "per_grid": {}}
    for G in (16, 24, 32):
        inp, tgt, mask, Dgeo, Dfree, Deuc, _ = BF.build_pool_maze(
            256, G, G, DEV, torch.Generator(device=DEV).manual_seed(1), walls=(1, 8))
        dg = [float(x) for x in Dgeo]
        keep = [i for i in range(256) if dg[i] <= 2 * G]
        var = tgt[mask > 0].var().item() + 1e-12
        # the D_geo-defining cell: argmax of the BFS field over reachable cells
        d = BF.bfs8(inp[:, 0:1], inp[:, 1:2], max_iter=8 * (G + G))
        reach = (d < BF.INF / 2).float()
        far_idx = (d[:, 0] * reach[:, 0]).flatten(1).argmax(1)
        print(f"\n=== frozen pool G={G}: eligible {len(keep)}/256 ===")
        grec = {"n_eligible": len(keep), "ckpts": {}}
        for s in (3, 4):
            model = BF.NCA().to(DEV)
            model.load_state_dict(torch.load(CKPT.format(G=G, s=s), map_location=DEV))
            model.eval()
            kn = knee_three_ways(model, inp, tgt, mask, var, far_idx)
            rec = {}
            joint = [i for i in keep if all(kn[n][i] is not None
                                            for n in ("p95", "p100", "farcell"))]
            for name in ("p95", "p100", "farcell"):
                ks = kn[name]
                conv = [i for i in keep if ks[i] is not None]
                rows = [(ks[i], max(dg[i], 1.0), max(float(Dfree[i]), 1.0),
                         max(float(Deuc[i]), 1.0)) for i in conv]
                cvs = {rl: cv([row[0] / row[1 + j] for row in rows])
                       for j, rl in enumerate(("dgeo", "dfree", "deuc"))}
                rec[name] = {
                    "coverage": len(conv) / len(keep),
                    "median_knee": st.median(ks[i] for i in conv),
                    "median_knee_joint": (st.median(ks[i] for i in joint)
                                          if joint else None),
                    "c1_median": st.median(ks[i] / max(dg[i], 1.0) for i in conv),
                    "cv": cvs,
                    "geo_smallest": cvs["dgeo"] < min(cvs["dfree"], cvs["deuc"])}
                r = rec[name]
                print(f"[s{s}] {name:8s} cov {100*r['coverage']:5.1f}%  "
                      f"median knee {r['median_knee']:5.1f} "
                      f"(joint {r['median_knee_joint']})  c1_med {r['c1_median']:.2f}  "
                      f"CV geo/free/euc {100*cvs['dgeo']:.0f}/{100*cvs['dfree']:.0f}/"
                      f"{100*cvs['deuc']:.0f}%  geo smallest: {r['geo_smallest']}")
            grec["ckpts"][f"s{s}"] = rec
        out["per_grid"][str(G)] = grec

    with open("probe_p100.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nsaved -> probe_p100.json")


if __name__ == "__main__":
    main()

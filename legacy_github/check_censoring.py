import argparse
import json
import statistics as st

import torch

import branch_F_tortuosity as BF


def match_rows(blk, Dgeo, Dfree, Deuc):
    key = {}
    for i in range(Dgeo.shape[0]):
        key.setdefault((int(Dgeo[i]), int(Dfree[i]), round(float(Deuc[i]), 3)), []).append(i)
    kept = set()
    misses = 0
    for j in range(len(blk["dgeo"])):
        k = (int(blk["dgeo"][j]), int(blk["dfree"][j]), round(blk["deuc"][j], 3))
        cands = key.get(k, [])
        free = [i for i in cands if i not in kept]
        if free:
            kept.add(free[0])
        else:
            misses += 1
    return kept, misses


def quartile_report(vals, kept, name):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = []
    for q in range(4):
        idx = order[q * len(order) // 4:(q + 1) * len(order) // 4]
        frac = sum(1 for i in idx if i in kept) / max(len(idx), 1)
        lo, hi = vals[idx[0]], vals[idx[-1]]
        out.append(f"{name} {lo:5.1f}-{hi:5.1f}: {100 * frac:5.1f}%")
    return "  ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="MUST be the device the training run used (generator streams differ)")
    a = ap.parse_args()

    res = json.load(open(a.json_path))
    walls = tuple(res.get("walls", [1, 8]))
    dev = a.device
    worst = 1.0
    for g, e in sorted(res["per_grid"].items(), key=lambda kv: int(kv[0])):
        G = int(g)
        _, _, _, Dgeo, Dfree, Deuc, tau = BF.build_pool_maze(
            256, G, G, dev, torch.Generator(device=dev).manual_seed(1), walls=walls)
        dg = [float(x) for x in Dgeo]
        tv = [float(x) for x in tau]
        for blk in e.get("scenes", []):
            kept, misses = match_rows(blk, Dgeo, Dfree, Deuc)
            n = Dgeo.shape[0]
            dropped = [i for i in range(n) if i not in kept]
            print(f"G={G:2d} seed={blk['seed']}  kept {len(kept)}/{n}"
                  f"  (match misses: {misses}, should be 0)")
            print(f"   kept fraction by tau quartile:   {quartile_report(tv, kept, 'tau')}")
            print(f"   kept fraction by D_geo quartile: {quartile_report(dg, kept, 'D')}")
            if dropped:
                print(f"   dropped scenes: mean D_geo {st.mean(dg[i] for i in dropped):5.1f} "
                      f"vs kept {st.mean(dg[i] for i in kept):5.1f};  "
                      f"mean tau {st.mean(tv[i] for i in dropped):4.2f} "
                      f"vs kept {st.mean(tv[i] for i in kept):4.2f}")
            frac_worst_q = sum(1 for i in sorted(range(n), key=lambda i: tv[i])[3 * n // 4:]
                               if i in kept) / (n - 3 * n // 4)
            worst = min(worst, frac_worst_q)
    print(f"\nVERDICT: worst kept-fraction in any top tau quartile = {100 * worst:.1f}%")
    print("  >90%: censoring is negligible; quote coverage freely.")
    print("  70-90%: report convergence fraction per grid next to c1 in the paper.")
    print("  <70%: c1 flatness is suspect -- the hard scenes are missing from the average.")


if __name__ == "__main__":
    main()

import argparse, ctypes, json, os, platform, random, statistics as st, subprocess, sys, tempfile, time

import torch
import triton
import triton.language as tl

import branch_F_tortuosity as BF

AUDIT = "audit_full_population.json"
CKPT = "results_E_maze_s34.json.G{G}.s{S}.pt"
THETA = 0.05
CAP = 152
F_PRIMARY = 0.95
WARMUP, TIMING_GROUPS, BLOCK_REPS, HOST_REPS = 25, 25, 50, 100


@triton.jit
def k0_reset(X, INP, NPIX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NPIX
    cols = tl.arange(0, 16)
    isinp = (cols[None, :] >= 12) & (cols[None, :] < 14)
    inp01 = tl.load(INP + idx[:, None] * 2 + (cols[None, :] - 12),
                    mask=m[:, None] & isinp, other=0.0)
    row = tl.where(isinp, inp01, 0.0)
    tl.store(X + idx[:, None] * 16 + cols[None, :], row, mask=m[:, None])


@triton.jit
def k1_perceive(X, P, WP, H: tl.constexpr, W: tl.constexpr,
                NPIX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NPIX
    hw = idx % (H * W)
    b0 = idx - hw
    y = hw // W
    x = hw % W
    for c in tl.static_range(14):
        a0 = tl.zeros((BLOCK,), dtype=tl.float32)
        a1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(9):
            ky = k // 3 - 1
            kx = k % 3 - 1
            ny = y + ky
            nx = x + kx
            nm = m & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nidx = b0 + ny * W + nx
            v = tl.load(X + nidx * 16 + c, mask=nm, other=0.0)
            a0 += tl.load(WP + c * 27 + 0 * 9 + k) * v
            a1 += tl.load(WP + c * 27 + 1 * 9 + k) * v
            a2 += tl.load(WP + c * 27 + 2 * 9 + k) * v
        tl.store(P + idx * 64 + 3 * c + 0, a0, mask=m)
        tl.store(P + idx * 64 + 3 * c + 1, a1, mask=m)
        tl.store(P + idx * 64 + 3 * c + 2, a2, mask=m)


@triton.jit
def k2_update(X, P, W1, B1, W2, B2, NPIX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NPIX
    c64 = tl.arange(0, 64)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)
    p = tl.load(P + idx[:, None] * 64 + c64[None, :], mask=m[:, None], other=0.0)
    out = tl.zeros((BLOCK, 16), dtype=tl.float32)
    for hc in tl.static_range(3):
        w1 = tl.load(W1 + c64[:, None] * 96 + (hc * 32 + c32)[None, :])
        h = tl.dot(p, w1, input_precision="ieee")
        h += tl.load(B1 + hc * 32 + c32)[None, :]
        h = tl.maximum(h, 0.0)
        w2 = tl.load(W2 + (hc * 32 + c32)[:, None] * 16 + c16[None, :])
        out += tl.dot(h, w2, input_precision="ieee")
    out += tl.load(B2 + c16)[None, :]
    xrow = tl.load(X + idx[:, None] * 16 + c16[None, :], mask=m[:, None], other=0.0)
    xrow += tl.where(c16[None, :] < 12, out, 0.0)
    tl.store(X + idx[:, None] * 16 + c16[None, :], xrow, mask=m[:, None])


@triton.jit
def k3_readout(X, RO, R, NPIX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NPIX
    c16 = tl.arange(0, 16)
    xrow = tl.load(X + idx[:, None] * 16 + c16[None, :], mask=m[:, None], other=0.0)
    w = tl.load(RO + c16)
    r = tl.sum(xrow * w[None, :], axis=1) + tl.load(RO + 16)
    tl.store(R + idx, r, mask=m)


def pack_weights(sd, dev):
    wp = sd["perceive.weight"].reshape(14, 3, 3, 3).contiguous()
    w1 = torch.zeros(64, 96, device=dev)
    w1[:42] = sd["w1.weight"].reshape(96, 42).t()
    b1 = sd["w1.bias"].clone()
    w2 = torch.zeros(96, 16, device=dev)
    w2[:, :12] = sd["w2.weight"].reshape(12, 96).t()
    b2 = torch.zeros(16, device=dev)
    b2[:12] = sd["w2.bias"]
    ro = torch.zeros(17, device=dev)
    ro[:12] = sd["readout.weight"].reshape(12)
    ro[16] = sd["readout.bias"][0]
    return (wp.to(dev).contiguous().view(-1), w1.contiguous().view(-1), b1.to(dev),
            w2.contiguous().view(-1), b2, ro)


class Shader:

    def __init__(self, sd, H, W, B, dev):
        self.H, self.W, self.B, self.dev = H, W, B, dev
        self.npix = B * H * W
        self.wp, self.w1, self.b1, self.w2, self.b2, self.ro = pack_weights(sd, dev)
        self.X = torch.zeros(self.npix, 16, device=dev)
        self.P = torch.zeros(self.npix, 64, device=dev)
        self.R = torch.zeros(self.npix, device=dev)
        self.INP = torch.zeros(self.npix, 2, device=dev)
        self.block = 64
        self.grid = (triton.cdiv(self.npix, self.block),)

    def set_scenes(self, inp):
        self.INP.copy_(inp.permute(0, 2, 3, 1).reshape(self.npix, 2))

    def frame_ops(self, T, readout=True):
        g, blk, n = self.grid, self.block, self.npix
        k0_reset[g](self.X, self.INP, NPIX=n, BLOCK=blk)
        for _ in range(T):
            k1_perceive[g](self.X, self.P, self.wp, H=self.H, W=self.W, NPIX=n, BLOCK=blk)
            k2_update[g](self.X, self.P, self.w1, self.b1, self.w2, self.b2, NPIX=n, BLOCK=blk)
        if readout:
            k3_readout[g](self.X, self.ro, self.R, NPIX=n, BLOCK=blk)

    def readout_img(self):
        return self.R.view(self.B, 1, self.H, self.W)

    def mem_bytes(self):
        bufs = [self.X, self.P, self.R, self.INP, self.wp, self.w1, self.b1,
                self.w2, self.b2, self.ro]
        return sum(t.numel() * t.element_size() for t in bufs)


BFS_C = r"""
#include <stdlib.h>
#include <string.h>
int bfs8(const unsigned char *obs, int H, int W, int sy, int sx, int *dist) {
    int n = H * W, head = 0, tail = 0, maxd = 0;
    static int q[1 << 16];
    for (int i = 0; i < n; i++) dist[i] = -1;
    if (obs[sy * W + sx]) return 0;
    dist[sy * W + sx] = 0; q[tail++] = sy * W + sx;
    while (head < tail) {
        int cur = q[head++], cy = cur / W, cx = cur % W, d = dist[cur];
        if (d > maxd) maxd = d;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            if (!dy && !dx) continue;
            int ny = cy + dy, nx = cx + dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            int ni = ny * W + nx;
            if (obs[ni] || dist[ni] >= 0) continue;
            dist[ni] = d + 1; q[tail++] = ni;
        }
    }
    return maxd;
}
"""


def build_bfs(tmpdir):
    src = os.path.join(tmpdir, "bfs8.c")
    so = os.path.join(tmpdir, "bfs8.so")
    open(src, "w").write(BFS_C)
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", src, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.bfs8.restype = ctypes.c_int
    return lib


def bfs_native(lib, obs_np, sy, sx):
    H, W = obs_np.shape
    dist = (ctypes.c_int * (H * W))()
    obs_c = (ctypes.c_ubyte * (H * W)).from_buffer_copy(obs_np.tobytes())
    return lib.bfs8(obs_c, H, W, sy, sx, dist)


def summarize(samples):
    vals = sorted(float(x) for x in samples)
    if not vals:
        return {"n": 0, "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "mean": float("nan"), "samples": []}

    def quantile(q):
        pos = q * (len(vals) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
        w = pos - lo
        return vals[lo] * (1 - w) + vals[hi] * w

    return {"n": len(vals), "median": st.median(vals), "p10": quantile(0.10),
            "p90": quantile(0.90), "mean": st.mean(vals), "samples": vals}


def time_graph(graph, warmup=WARMUP, groups=TIMING_GROUPS,
               block=BLOCK_REPS, host_reps=HOST_REPS):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    gpu_blocked, host_blocked = [], []
    for _ in range(groups):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        e0.record()
        for _ in range(block):
            graph.replay()
        e1.record()
        torch.cuda.synchronize()
        host_blocked.append((time.perf_counter_ns() - t0) / 1e3 / block)
        gpu_blocked.append(e0.elapsed_time(e1) * 1e3 / block)

    host_frame = []
    torch.cuda.synchronize()
    for _ in range(host_reps):
        t0 = time.perf_counter_ns()
        graph.replay()
        torch.cuda.synchronize()
        host_frame.append((time.perf_counter_ns() - t0) / 1e3)
    return {"gpu_blocked_us": summarize(gpu_blocked),
            "host_blocked_us": summarize(host_blocked),
            "host_frame_us": summarize(host_frame)}


def safe_cmd(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True,
                                       timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def capture_frame(shader, T):
    shader.frame_ops(T)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        shader.frame_ops(T)
    return g


def parity_phase(model, shader, vinp, vtgt, vlm, vvar, cap, dev):
    B = vinp.shape[0]
    shader.set_scenes(vinp)
    Tlist = list(range(1, cap + 1))
    with torch.no_grad():

        ps_t = {t: [] for t in Tlist}
        for s in range(0, B, 64):
            sl = slice(s, min(B, s + 64))
            outs = model.run(vinp[sl], cap, collect=set(Tlist))
            for t in Tlist:
                ps_t[t].append(BF.tail_nmse(outs[t], vtgt[sl], vlm[sl], vvar).cpu())
        ps_t = {t: torch.cat(v) for t, v in ps_t.items()}
        ref_final = model.run(vinp, cap)

    ps_k = {}
    maxdev = 0.0
    shader.frame_ops(0)
    for t in Tlist:
        g, blk, n = shader.grid, shader.block, shader.npix
        k1_perceive[g](shader.X, shader.P, shader.wp, H=shader.H, W=shader.W, NPIX=n, BLOCK=blk)
        k2_update[g](shader.X, shader.P, shader.w1, shader.b1, shader.w2, shader.b2, NPIX=n, BLOCK=blk)
        k3_readout[g](shader.X, shader.ro, shader.R, NPIX=n, BLOCK=blk)
        pred = shader.readout_img()
        ps_k[t] = BF.tail_nmse(pred, vtgt, vlm, vvar).cpu()
    maxdev = (shader.readout_img() - ref_final).abs().max().item()
    match, mismatched = 0, []
    knees_t, knees_k = [], []
    for i in range(B):
        kt, _ = BF.abs_knee({t: ps_t[t][i].item() for t in Tlist}, theta=THETA)
        kk, _ = BF.abs_knee({t: ps_k[t][i].item() for t in Tlist}, theta=THETA)
        knees_t.append(kt); knees_k.append(kk)
        if kt == kk:
            match += 1
        else:
            mismatched.append({"scene": i, "torch": kt, "kernel": kk})
    max_curve_dev = max((ps_t[t] - ps_k[t]).abs().max().item() for t in Tlist)
    return {"n_scenes": B, "knee_match": match, "knee_mismatch": mismatched[:10],
            "max_readout_dev": maxdev, "max_tailnmse_dev": max_curve_dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--ckpt_seed", type=int, default=3)
    ap.add_argument("--parity_seeds", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--out", default="branch_O_shader_v2.json")
    ap.add_argument("--timing_seed", type=int, default=20260719)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    audit = json.load(open(AUDIT))
    policies = {p["f"]: p for p in audit["policy_full"]}
    res = {"protocol": "see module docstring; constants frozen from " + AUDIT,
           "theta": THETA, "cap": CAP, "f_primary": F_PRIMARY,
           "policies": {str(f): {"flat_T": p["flat_T"], "geo_c": p["geo_c"]}
                        for f, p in policies.items()},
           "timing_protocol_version": 2, "warmup": WARMUP,
           "timing_groups": TIMING_GROUPS, "block_reps": BLOCK_REPS,
           "host_frame_reps": HOST_REPS, "timing_seed": a.timing_seed,
           "primary_timing": "host_frame_us.median (graph replay + synchronize)",
           "measurement_scope": {
               "included": "one CUDA-graph submission and completion wait; reset, 2T update kernels, readout",
               "excluded": "graph capture/setup and executable metadata; host-to-device input transfer; policy selection; renderer integration",
               "input_residency": "static input tensor already resident on GPU",
               "dynamic_geometry_accounting": "compute-only diagnostic: subtracts native CPU BFS but not changed-mask transfer or graph selection",
           },
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "triton": triton.__version__,
           "environment": {"platform": platform.platform(), "python": sys.version,
                           "cuda_runtime": torch.version.cuda,
                           "nvidia_smi": safe_cmd(["nvidia-smi", "--query-gpu=driver_version,power.limit,clocks.current.graphics", "--format=csv,noheader"]),
                           "gcc": safe_cmd(["gcc", "--version"]).splitlines()[0]},
           "per_grid": {}}

    if a.smoke:
        a.grids, a.parity_seeds = [16], [a.ckpt_seed]

    tmpdir = tempfile.mkdtemp(prefix="branch_O_")
    lib = build_bfs(tmpdir)

    for G in a.grids:
        print(f"=== G={G} ===", flush=True)
        vinp, vtgt, vlm, vDgeo, vDfree, vDeuc, vtau = BF.build_pool_maze(
            256, G, G, dev, torch.Generator(device=dev).manual_seed(1))
        vvar = vtgt[vlm > 0].var().item() + 1e-12
        gres = {"parity": {}, "bfs": {}, "timing": {}}


        for sdd in a.parity_seeds:
            sd = torch.load(CKPT.format(G=G, S=sdd), map_location=dev)
            model = BF.NCA().to(dev); model.load_state_dict(sd); model.eval()
            shader = Shader(sd, G, G, 256, dev)
            p = parity_phase(model, shader, vinp, vtgt, vlm, vvar, CAP, dev)
            gres["parity"][f"s{sdd}"] = p
            print(f"  parity s{sdd}: knees {p['knee_match']}/{p['n_scenes']} match, "
                  f"max|Δreadout|={p['max_readout_dev']:.2e}, "
                  f"max|Δtail|={p['max_tailnmse_dev']:.2e}", flush=True)
        if a.smoke:
            break


        obs = (vinp[:, 1] > 0.5).to(torch.uint8).cpu().numpy()
        seeds_yx = [(int(y), int(x)) for y, x in
                    ((vinp[i, 0] > 0.5).nonzero(as_tuple=False)[0] for i in range(256))]
        mism = 0
        for i in range(256):
            d = bfs_native(lib, obs[i], *seeds_yx[i])
            if d != int(vDgeo[i].item()):
                mism += 1
        bfs_groups = []
        REP_B = 20
        for _ in range(REP_B):
            t0 = time.perf_counter_ns()
            for i in range(256):
                bfs_native(lib, obs[i], *seeds_yx[i])
            bfs_groups.append((time.perf_counter_ns() - t0) / 1e3 / 256)
        bfs_summary = summarize(bfs_groups)
        t_bfs_us = bfs_summary["median"]
        gres["bfs"] = {"dgeo_mismatch": mism, "per_scene_us": t_bfs_us,
                       "per_scene_us_summary": bfs_summary,
                       "impl": "C -O2 array-queue via ctypes (per-call marshal incl.)"}
        print(f"  BFS native: {t_bfs_us:.1f} us/scene, Dgeo mismatches {mism}/256", flush=True)


        sd = torch.load(CKPT.format(G=G, S=a.ckpt_seed), map_location=dev)
        model = BF.NCA().to(dev); model.load_state_dict(sd); model.eval()
        sh1 = Shader(sd, G, G, 1, dev)
        sh1.set_scenes(vinp[:1])
        eligible = [i for i in range(256) if vDgeo[i].item() <= 2 * G]
        arms = {}
        for f, pol in policies.items():
            geo_T = [max(1, -int(-(pol["geo_c"] * max(vDgeo[i].item(), 1.0)) // 1))
                     for i in eligible]
            arms[f] = {"flat_T": pol["flat_T"], "geo_T": geo_T}
        uniqT = sorted({t for f in arms for t in arms[f]["geo_T"]}
                       | {arms[f]["flat_T"] for f in arms})
        print(f"  eligible {len(eligible)}/256, unique budgets {len(uniqT)} "
              f"(T {uniqT[0]}..{uniqT[-1]})", flush=True)


        graphs = {T: capture_frame(sh1, T) for T in uniqT}
        timing_order = uniqT[:]
        random.Random(a.timing_seed + G).shuffle(timing_order)
        tT = {}
        for j, T in enumerate(timing_order, 1):
            tT[T] = time_graph(graphs[T])
            print(f"  timing {j:02d}/{len(timing_order)} T={T}: "
                  f"host {tT[T]['host_frame_us']['median']:.1f} us, "
                  f"device {tT[T]['gpu_blocked_us']['median']:.1f} us", flush=True)
        del graphs
        xs = uniqT
        ys = [tT[T]["host_frame_us"]["median"] for T in xs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        icept = my - slope * mx
        device_ys = [tT[T]["gpu_blocked_us"]["median"] for T in xs]
        dmy = sum(device_ys) / n
        device_slope = sum((x - mx) * (y - dmy) for x, y in zip(xs, device_ys)) / sum((x - mx) ** 2 for x in xs)
        device_icept = dmy - device_slope * mx

        Tf = arms[F_PRIMARY]["flat_T"]
        with torch.no_grad():
            for _ in range(WARMUP):
                model.run(vinp[:1], Tf)
            torch.cuda.synchronize()
            eag = []
            for _ in range(50):
                t0 = time.perf_counter_ns()
                model.run(vinp[:1], Tf)
                torch.cuda.synchronize()
                eag.append((time.perf_counter_ns() - t0) / 1e3)
        gres["timing"] = {
            "per_T_us": {str(T): tT[T] for T in uniqT},
            "timing_order": timing_order,
            "host_t_step_us": slope, "host_intercept_us": icept,
            "device_t_step_us": device_slope, "device_intercept_us": device_icept,
            "eager_flatT_host_us": summarize(eag),
            "mem_bytes": sh1.mem_bytes(),
            "graph_nodes_per_frame": "2T+2 kernels; one graph submission",
            "memory_scope": "resident tensor buffers only; excludes CUDA graph executable metadata",
        }

        for f, arm in arms.items():
            bfsu = t_bfs_us
            ares = {
                "flat_T": arm["flat_T"], "geo_T_mean": st.mean(arm["geo_T"]),
                "n_eligible": len(eligible),
            }
            for metric in ("host_frame_us", "gpu_blocked_us"):
                tf = tT[arm["flat_T"]][metric]["median"]
                tg = st.mean([tT[T][metric]["median"] for T in arm["geo_T"]])
                dg = tf - tg
                ares[metric] = {
                    "t_flat_us": tf, "t_geo_us": tg, "saving_us": dg,
                    "saving_pct": 100 * dg / tf,
                    "breakeven_frames_static_compute_only": bfsu / dg if dg > 0 else float("inf"),
                    "net_saving_us_dynamic_compute_only": dg - bfsu,
                }
            gres.setdefault("arms", {})[str(f)] = ares
            if f == F_PRIMARY:
                pr = ares["host_frame_us"]
                print(f"  f={f}: flat {pr['t_flat_us']:.1f}us vs "
                      f"geo {pr['t_geo_us']:.1f}us  "
                      f"saving {pr['saving_pct']:+.1f}%, BFS {bfsu:.1f}us; "
                      f"compute-only dynamic net "
                      f"{pr['net_saving_us_dynamic_compute_only']:+.1f} us/frame", flush=True)
        res["per_grid"][str(G)] = gres
        json.dump(res, open(a.out + ".partial", "w"), indent=1)

    if not a.smoke:
        res["pooled_arms"] = {}
        for f in policies:
            rows = [(g, res["per_grid"][str(g)]["arms"][str(f)]) for g in a.grids]
            total = sum(ares["n_eligible"] for _, ares in rows)
            pooled = {"n_eligible": total}
            for metric in ("host_frame_us", "gpu_blocked_us"):
                tf = sum(ares["n_eligible"] * ares[metric]["t_flat_us"] for _, ares in rows) / total
                tg = sum(ares["n_eligible"] * ares[metric]["t_geo_us"] for _, ares in rows) / total
                bfsu = sum(ares["n_eligible"] * res["per_grid"][str(g)]["bfs"]["per_scene_us"]
                           for g, ares in rows) / total
                dg = tf - tg
                pooled[metric] = {"t_flat_us": tf, "t_geo_us": tg,
                                  "saving_us": dg, "saving_pct": 100 * dg / tf,
                                  "bfs_us": bfsu,
                                  "breakeven_frames_static_compute_only": bfsu / dg if dg > 0 else float("inf"),
                                  "net_saving_us_dynamic_compute_only": dg - bfsu}
            res["pooled_arms"][str(f)] = pooled
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"saved -> {a.out}")
    else:
        print("smoke: parity only, nothing saved")


if __name__ == "__main__":
    main()

import math
import statistics as st


C_RESOLUTION = 0.01


def quantile(xs, q):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, math.ceil(q * len(ys)) - 1))]


def geometric_budget(c, distance):
    return max(1, math.ceil(c * max(float(distance), 1.0)))


def flat_budget(rows, f, knee):
    conv = [r[knee] for r in rows if r[knee] is not None]
    return int(math.ceil(quantile(conv, f)))


def eval_policy(rows, ruler, constant, knee):
    if ruler == "flat":
        budgets = [int(constant)] * len(rows)
    else:
        budgets = [geometric_budget(constant, r[ruler]) for r in rows]
    served = sum(r[knee] is not None and r[knee] <= b
                 for r, b in zip(rows, budgets))
    return st.mean(budgets), served / len(rows)


def fit_c_to_coverage(rows, ruler, target_coverage, knee,
                      resolution=C_RESOLUTION):
    conv = [r for r in rows if r[knee] is not None]
    if not conv:
        raise ValueError("cannot calibrate a policy without converged rows")
    required_ticks = []
    for r in conv:
        d = max(float(r[ruler]), 1.0)
        k = int(r[knee])
        tick = max(1, int(math.floor((k - 1) / (d * resolution))) + 1)
        while geometric_budget(round(tick * resolution, 10), d) < k:
            tick += 1
        while tick > 1 and geometric_budget(round((tick - 1) * resolution, 10), d) >= k:
            tick -= 1
        required_ticks.append(tick)
    n_required = int(math.ceil(target_coverage * len(rows) - 1e-12))
    if n_required > len(required_ticks):
        raise ValueError("target coverage exceeds the convergence ceiling")
    required_ticks.sort()
    c = round(required_ticks[max(0, n_required - 1)] * resolution, 10)
    _, coverage = eval_policy(rows, ruler, c, knee)
    if coverage < target_coverage:
        raise RuntimeError("discrete calibration failed to reach target coverage")
    return c


def fit_matched_policy(rows, ruler, f, knee, resolution=C_RESOLUTION):
    T = flat_budget(rows, f, knee)
    _, target = eval_policy(rows, "flat", T, knee)
    c = fit_c_to_coverage(rows, ruler, target, knee, resolution=resolution)
    return c, T, target

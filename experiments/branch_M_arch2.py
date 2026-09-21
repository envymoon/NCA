"""Branch M — second-architecture control (POST-FREEZE ADDENDUM, 2026-07-18).

The paper's stated limitation is "single architecture, 5.7k parameters". This branch
retrains under the branch_E_anytime protocol VERBATIM -- same pool RNG (train seed 0,
val seed 1), same global-cap rule, same theta=0.05 absolute knee, same tail-quantile
definition, same per-scene row dump -- with exactly ONE changed variable: the perception
stage. The learned depthwise 3x3 (378 trainable weights) is frozen to the classic
identity + Sobel-x + Sobel-y filter bank of Mordvintsev et al.'s Growing NCA, the
canonical perception in the NCA literature. Nothing else moves, so any change in the
measured constants is attributable to the architecture alone.

What would strengthen the paper: c1_geo stays ~1 (no super-linear training margin is an
architecture-robust fact, not a quirk of our perception parameterisation) and the
geodesic ruler keeps the smallest per-scene CV. What would falsify the generalisation:
c1_geo shifts far from the learned-perception value, or the CV ordering flips.

Run: python3 branch_M_arch2.py --grids 32 --seeds 0 1 2 --out results_M_sobel.json
"""
import json
import sys

import torch

import branch_F_tortuosity as BF
import branch_E_anytime as E


class SobelNCA(BF.NCA):
    """BF.NCA with the depthwise perception conv FROZEN to identity+Sobel-x+Sobel-y.

    Same module names and tensor shapes as BF.NCA (state_dicts stay interchangeable);
    the perception weights are copied in at construction and excluded from training.
    Grouped-conv channel order is [id, sx, sy] per input plane, matching the
    (3k, 1, 3, 3) weight layout of the learned original.
    """

    def __init__(self, C=12, hidden=96, in_ch=2):
        super().__init__(C=C, hidden=hidden, in_ch=in_ch)
        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        filt = torch.stack([ident, sx, sx.t()])                    # (3, 3, 3)
        with torch.no_grad():
            self.perceive.weight.copy_(filt.repeat(C + in_ch, 1, 1).unsqueeze(1))
        self.perceive.weight.requires_grad_(False)


def main():
    probe = SobelNCA()
    total = sum(p.numel() for p in probe.parameters())
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"Branch M: fixed-Sobel perception control  "
          f"params total={total} trainable={trainable} (frozen perception={total - trainable})")

    BF.NCA = SobelNCA        # the one changed variable; protocol is branch E's, verbatim
    E.main()

    # E.main wrote the result file without knowing the class was swapped -- stamp the
    # architecture identity into the JSON so the file is self-describing.
    out = "results_E_maze.json"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    try:
        res = json.load(open(out))
    except FileNotFoundError:
        return
    res["architecture"] = {
        "name": "sobel_fixed_perception",
        "perception": "frozen identity+Sobel-x+Sobel-y depthwise (Mordvintsev-style)",
        "params_total": total, "params_trainable": trainable,
        "baseline": "BF.NCA learned depthwise 3x3 (results_E_maze_s01234.json)",
    }
    json.dump(res, open(out, "w"), indent=2)
    print(f"architecture stamp written -> {out}")


if __name__ == "__main__":
    main()

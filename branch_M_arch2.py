import json
import sys

import torch

import branch_F_tortuosity as BF
import branch_E_anytime as E


class SobelNCA(BF.NCA):

    def __init__(self, C=12, hidden=96, in_ch=2):
        super().__init__(C=C, hidden=hidden, in_ch=in_ch)
        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        filt = torch.stack([ident, sx, sx.t()])
        with torch.no_grad():
            self.perceive.weight.copy_(filt.repeat(C + in_ch, 1, 1).unsqueeze(1))
        self.perceive.weight.requires_grad_(False)


def main():
    probe = SobelNCA()
    total = sum(p.numel() for p in probe.parameters())
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"Branch M: fixed-Sobel perception control  "
          f"params total={total} trainable={trainable} (frozen perception={total - trainable})")

    BF.NCA = SobelNCA
    E.main()


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

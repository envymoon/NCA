"""Diagnostic (NOT submission data): is the het=0.9 quality-floor failure just undertraining?

Trains het=0.9 seed 0 for 40k iters, twice the locked protocol's 20k, with the identical
RNG stream (compatible_nca3 seeding + same batch/horizon draws), records the training loss
curve, and evaluates at 20k and 40k. The 20k evaluation must reproduce the submission run's
het=0.9 seed 0 unit; the 20k->40k delta answers the undertraining question.
"""
import json
import sys
import time

import torch

sys.argv = [sys.argv[0]]
import branch_L_clean as L
import branch_F_tortuosity as BF

HET = 0.9
SEED = 0
TOTAL_ITERS = 40000
EVAL_AT = (20000, 40000)
LOSS_EVERY = 100
OUT = "diag_L_het09_s0_40k.json"


def main():
    args = L.parse_args()
    L.validate_args(args)
    device = BF.get_device("cuda")
    L.assert_control_compatibility(device)
    pools, vpools = L.make_paired_pools(args, device)

    inp, tgt, lm, dgeo, dfree, deuc, hstar, tau, sweeps = pools[HET]
    vinp, vtgt, vlm, vdgeo, vdfree, vdeuc, vhstar, vtau, vsweeps = vpools[HET]
    var = tgt[lm > 0].var().item() + 1e-12
    vvar = vtgt[vlm > 0].var().item() + 1e-12
    clamped = ((((vtgt >= 1.0 - 1e-6).float() * vlm).sum(dim=(1, 2, 3))) > 0).cpu()
    eligible = ((vdgeo > 2).cpu() & ~clamped)

    model = L.compatible_nca3(SEED, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    bucket = []
    evals = {}
    t0 = time.time()
    for it in range(1, TOTAL_ITERS + 1):
        idx = torch.randint(0, len(inp), (args.batch,), device=device)
        horizon = int(torch.randint(4, args.cap + 1, (1,), device=device).item())
        pred = model.run_train(inp[idx], horizon, seg=args.seg)
        loss = BF.masked_nmse(pred, tgt[idx], lm[idx], var)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bucket.append(loss.item())
        if it % LOSS_EVERY == 0:
            losses.append(sum(bucket) / len(bucket))
            bucket = []
        if it in EVAL_AT:
            torch.save({"diag": True, "iters": it, "het": HET, "seed": SEED,
                        "state_dict": model.state_dict()}, f"{OUT}.it{it}.pt")
            ev = L.evaluate(model, vinp, vtgt, vlm, vvar, args, eligible)
            summary = L.summarize_seed(ev, vdgeo, vdfree, vdeuc, eligible)
            evals[str(it)] = {**summary, **ev}
            cov = summary["tail_unconditional_coverage"]
            print(f"iters={it}: mean knee={ev['aggregate_mean_knee']} "
                  f"tail knee={ev['aggregate_tail_knee']} "
                  f"mean floor={ev['aggregate_mean_floor']:.4f} "
                  f"tail floor={ev['aggregate_tail_floor']:.4f} "
                  f"coverage={cov:.3f} "
                  f"geo CV={summary['tail_ratio_cv_conditional']['geo']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    L.write_json(OUT, {
        "diag": "het09_undertraining_probe",
        "protocol_ref": "branch_L_clean locked protocol, het=0.9 seed=0, iters extended to 40k",
        "loss_every": LOSS_EVERY,
        "train_loss": losses,
        "evals": evals,
    })
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

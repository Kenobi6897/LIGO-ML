"""Stage 2, Step 4 — THE TRANSFER MEASUREMENT. The stage's headline number.

Take the Stage 1 checkpoint — trained purely on simulated Gaussian noise at design
sensitivity — and score it, untouched, on the Stage 2 test split: the same signal
population, injected into real O3 noise. Whatever changes is the price of pretending
detector noise is Gaussian.

Stage 1 pre-paid the prediction (stage1_gw150914.py): on real O1 noise its background
segments scored median logit +17.5 where simulated test negatives score around -3. If
that generalises, the damage should land almost entirely in the FALSE-ALARM FLOOR — the
noise moves toward "signal", not the signals toward "noise".

WHAT IS COMPARED, AND WHY IT IS APPLES TO APPLES
  - Same model weights, eval mode, no adaptation.
  - Same injection recipe, mass range, SNR definition-in-band; only the noise (and its
    PSD, measured per block) differs.
  - Thresholds are Stage 1's own: set on STAGE 1 VAL negatives, exactly as
    stage1_eval.py set them. The question "how many false alarms per hour does the
    deployed Gaussian-trained detector produce on real data?" only makes sense at the
    thresholds that detector actually shipped with.

THE CHECKS THAT CAN FAIL (predictions, not vibes):
  1. AUC drops on real noise. If it does NOT, per the plan note: suspect the "real"
     noise isn't real (wrong file, cached Gaussian, wrong split) before celebrating.
  2. The false-alarm floor rises at fixed threshold. This is the mechanism claim.
  3. The sim-side numbers reproduce Stage 1's eval (AUC 0.9877 +- 0.002) — proving the
     harness itself is measuring the same thing stage1_eval.py measured.

Run (from WSL2, after stage2_dataset.py; needs ~/ligo-data/stage1.h5 + stage1_cnn.pt):
    python stage2_transfer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_data import Stage1Dataset
from stage1_model import load_checkpoint
from stage1_train import DEFAULT_CKPT as STAGE1_CKPT

from stage2_data import Stage2Dataset

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-1, 1e-2, 1e-3)


def score(model, ds, device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=1024, num_workers=2)
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 4 — the transfer measurement")
    ap.add_argument("--ckpt", type=Path, default=STAGE1_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt)
    model.to(device)
    print(f"Stage 1 checkpoint: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f} "
          f"— trained on simulated Gaussian noise, never shown a byte of real data\n")

    # The two worlds, same model.
    sim_val = Stage1Dataset("val")
    sim_test = Stage1Dataset("test")
    real_test = Stage2Dataset("test")
    print(f"sim  : {sim_test}")
    print(f"real : {real_test}\n")

    sim_val_scores = score(model, sim_val, device)
    sim_scores = score(model, sim_test, device)
    real_scores = score(model, real_test, device)

    sim_neg = sim_scores[sim_test.y == 0]
    sim_pos = sim_scores[sim_test.y == 1]
    real_neg = real_scores[real_test.y == 0]
    real_pos = real_scores[real_test.y == 1]

    # --- AUC: the one-number version ------------------------------------------------
    auc_sim = float(roc_auc_score(sim_test.y, sim_scores))
    auc_real = float(roc_auc_score(real_test.y, real_scores))
    print(f"AUC, same model, same signal population:")
    print(f"  simulated Gaussian noise : {auc_sim:.4f}   (Stage 1 reported 0.9877)")
    print(f"  real O3 noise            : {auc_real:.4f}   (drop: {auc_sim - auc_real:+.4f})")

    # --- the mechanism: where did the scores move? ------------------------------------
    print(f"\nlogit medians (the +17.5 clue, at scale):")
    print(f"  {'':>12} {'negatives':>12} {'positives':>12}")
    print(f"  {'sim':>12} {np.median(sim_neg):>12.2f} {np.median(sim_pos):>12.2f}")
    print(f"  {'real':>12} {np.median(real_neg):>12.2f} {np.median(real_pos):>12.2f}")
    neg_shift = float(np.median(real_neg) - np.median(sim_neg))
    pos_shift = float(np.median(real_pos) - np.median(sim_pos))
    print(f"  negative floor moved {neg_shift:+.1f} logits; positives moved {pos_shift:+.1f}")

    # --- FA/h at Stage 1's OWN thresholds ----------------------------------------------
    thresholds = {fap: float(np.quantile(sim_val_scores[sim_val.y == 0], 1 - fap))
                  for fap in FAPS}
    sim_h = len(sim_neg) / 3600.0
    real_h = len(real_neg) / 3600.0
    print(f"\nfalse alarms at Stage 1's deployed thresholds "
          f"(set on sim val, as stage1_eval.py set them):")
    print(f"  {'target FAP':>12} {'sim FA/h':>10} {'real FA/h':>11} {'ratio':>8}")
    fa_ratio = {}
    for fap in FAPS:
        s = (sim_neg > thresholds[fap]).mean() * 3600
        r = (real_neg > thresholds[fap]).mean() * 3600
        fa_ratio[fap] = r / s if s > 0 else np.inf
        print(f"  {fap:>12.0e} {s:>10.1f} {r:>11.1f} {fa_ratio[fap]:>8.1f}x")
    print(f"  (background: {sim_h:.2f} h sim, {real_h:.2f} h real held-out noise)")

    # --- the checks that can fail --------------------------------------------------------
    checks = [
        (auc_real < auc_sim - 0.005,
         f"the drop exists: real AUC {auc_real:.4f} < sim AUC {auc_sim:.4f} - 0.005"),
        (fa_ratio[1e-2] > 1.5,
         f"the false-alarm floor rises at fixed threshold ({fa_ratio[1e-2]:.1f}x at FAP 1e-2)"),
        (abs(auc_sim - 0.9877) < 0.002,
         f"harness sanity: sim AUC {auc_sim:.4f} reproduces Stage 1's 0.9877"),
    ]
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")
    if auc_real >= auc_sim - 0.005:
        print("  SUSPICIOUS: no degradation on 'real' noise — verify the noise is real"
              " (wrong file / cached Gaussian / wrong split) before believing this.")

    # --- figure ---------------------------------------------------------------------------
    fig, (axh, axr) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    bins = np.linspace(min(sim_neg.min(), real_neg.min()) - 1,
                       max(sim_pos.max(), real_pos.max()) + 1, 80)
    axh.hist(sim_neg, bins=bins, density=True, alpha=0.45, color="tab:blue",
             label=f"sim negatives (med {np.median(sim_neg):+.1f})")
    axh.hist(real_neg, bins=bins, density=True, alpha=0.45, color="tab:red",
             label=f"REAL negatives (med {np.median(real_neg):+.1f})")
    axh.hist(real_pos, bins=bins, density=True, histtype="step", lw=1.6, color="darkred",
             label=f"real positives (med {np.median(real_pos):+.1f})")
    axh.hist(sim_pos, bins=bins, density=True, histtype="step", lw=1.2, color="tab:blue",
             ls="--", label=f"sim positives (med {np.median(sim_pos):+.1f})")
    for fap, ls in zip(FAPS, (":", "--", "-.")):
        axh.axvline(thresholds[fap], color="k", lw=1.2, ls=ls, alpha=0.6,
                    label=f"Stage 1 threshold @ FAP {fap:.0e}")
    axh.set_xlabel("CNN score (logit)")
    axh.set_ylabel("density")
    axh.set_title("The mechanism: real noise moves toward 'signal'\n"
                  "(Gaussian-trained model, same signals, only the noise changed)")
    axh.legend(fontsize=7, loc="upper left")
    axh.grid(alpha=0.3)

    for scores, ds, c, lbl, auc in (
        (sim_scores, sim_test, "tab:blue", "sim test", auc_sim),
        (real_scores, real_test, "tab:red", "REAL O3 test", auc_real),
    ):
        fpr, tpr, _ = roc_curve(ds.y, scores)
        axr.semilogx(fpr, tpr, color=c, lw=2, label=f"{lbl} — AUC {auc:.4f}")
    axr.semilogx([1e-4, 1], [1e-4, 1], "k--", lw=1, alpha=0.5, label="chance")
    axr.set_xlim(1e-4, 1)
    axr.set_ylim(0, 1.02)
    axr.set_xlabel("false positive rate")
    axr.set_ylabel("true positive rate")
    axr.set_title("Same model, two worlds — the price of assuming Gaussian\n"
                  "(the gap is Stage 2's result; recovering it is Step 5's job)")
    axr.legend(loc="lower right", fontsize=9)
    axr.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "4_transfer.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

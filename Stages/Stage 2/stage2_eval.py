"""Stage 2, Step 6 — evaluate on real noise. Three arms, one figure.

  ARM A  Stage-1 model on SIMULATED noise   — the reference (Stage 1's own result)
  ARM B  Stage-1 model on REAL noise        — the drop (Step 4's transfer, per-SNR)
  ARM C  Stage-2 model on REAL noise        — the recovery (what retraining buys back)

The A->B gap is the price of assuming Gaussian; the B->C gap is how much of that price
is learnable; the A->C gap is the tax reality still charges after you've learned it.

DISCIPLINE, UNCHANGED FROM STAGE 1:
  - Not accuracy. ROC/AUC + efficiency vs injected SNR at fixed FAP + FA per hour.
  - Thresholds set on VAL negatives, measured on TEST negatives. On real data val and
    test are different stretches of time, so the threshold-transfer check now also
    measures non-stationarity across weeks — expect it looser than Stage 1's 2x, and
    say so rather than hide it (gate at 3x, report exactly).
  - Arms B and C use thresholds from their own model's scores on STAGE 2 val — each
    detector is deployed with thresholds set on its own recent past.

THE GAUSSIAN CEILING IS NOW A DIAGNOSTIC, NOT A LAW. TPR = Phi(rho - z) was a theorem
in Stage 1's Gaussian noise. In real noise Neyman-Pearson still holds but the optimal
statistic is no longer this Gaussian formula — the curve is plotted as the *Gaussian*
ceiling, and each arm's distance below it measures the non-Gaussianity tax. Arm C
sitting ABOVE it would still be the leak alarm (real noise is strictly harder).

THE CHECKS THAT CAN FAIL:
  1. arm C AUC > arm B AUC (retraining on real noise must beat transferring from sim);
  2. arm C efficiency at SNR 18-20 @ FAP 1e-2 >= 0.90 (finds loud signals in real noise);
  3. arm C efficiency at SNR 4-6   @ FAP 1e-2 <= 0.80 (still fails at low SNR — the
     99%-AUC fraud alarm, unchanged);
  4. arm C thresholds transfer val->test within 3x at FAP 1e-1, 1e-2.

Run (from WSL2, after stage2_train.py):
    python stage2_eval.py
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
from scipy.stats import norm as normal
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_data import Stage1Dataset
from stage1_eval import wilson
from stage1_model import load_checkpoint
from stage1_train import DEFAULT_CKPT as STAGE1_CKPT

from stage2_data import Stage2Dataset
from stage2_train import DEFAULT_CKPT as STAGE2_CKPT

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-1, 1e-2, 1e-3)
FAP_COLORS = ("tab:blue", "tab:orange", "tab:green")
SNR_EDGES = np.arange(4.0, 22.0, 2.0)
CENTRES = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])


def score(model, ds, device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=1024, num_workers=2)
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).ravel()


class Arm:
    """One (model, dataset) pairing: thresholds from its val, everything else from test."""

    def __init__(self, name: str, model, val_ds, test_ds, device):
        self.name = name
        val_scores = score(model, val_ds, device)
        self.test_scores = score(model, test_ds, device)
        self.y = test_ds.y
        self.snr = test_ds.snr
        self.thresholds = {
            fap: float(np.quantile(val_scores[val_ds.y == 0], 1 - fap)) for fap in FAPS
        }
        self.neg = self.test_scores[self.y == 0]
        self.pos = self.test_scores[self.y == 1]
        self.pos_snr = self.snr[self.y == 1]
        self.auc = float(roc_auc_score(self.y, self.test_scores))
        self.hours = len(self.neg) / 3600.0
        self.measured_fap = {
            fap: float((self.neg > t).mean()) for fap, t in self.thresholds.items()
        }

    def efficiency(self, fap: float):
        t = self.thresholds[fap]
        tpr, lo, hi = [], [], []
        for a, b in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            in_bin = (self.pos_snr >= a) & (self.pos_snr < b)
            k, n = int((self.pos[in_bin] > t).sum()), int(in_bin.sum())
            tpr.append(k / n if n else np.nan)
            w = wilson(k, n)
            lo.append(w[0]), hi.append(w[1])
        return np.array(tpr), np.array(lo), np.array(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 6 — evaluate on real noise")
    ap.add_argument("--ckpt1", type=Path, default=STAGE1_CKPT)
    ap.add_argument("--ckpt2", type=Path, default=STAGE2_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1, c1 = load_checkpoint(args.ckpt1)
    m2, c2 = load_checkpoint(args.ckpt2)
    m1.to(device), m2.to(device)
    print(f"Stage 1 model: epoch {c1['epoch']}, val AUC {c1['val_auc']:.4f} (Gaussian-trained)")
    print(f"Stage 2 model: epoch {c2['epoch']}, val AUC {c2['val_auc']:.4f} (real-noise-trained)\n")

    arms = {
        "A": Arm("S1 model / sim noise", m1, Stage1Dataset("val"), Stage1Dataset("test"), device),
        "B": Arm("S1 model / REAL noise", m1, Stage2Dataset("val"), Stage2Dataset("test"), device),
        "C": Arm("S2 model / REAL noise", m2, Stage2Dataset("val"), Stage2Dataset("test"), device),
    }

    print(f"{'arm':>4} {'model/noise':>24} {'AUC':>8} " +
          " ".join(f"FA/h@{fap:.0e}".rjust(11) for fap in FAPS))
    for k, arm in arms.items():
        fas = " ".join(f"{arm.measured_fap[fap]*3600:11.1f}" for fap in FAPS)
        print(f"{k:>4} {arm.name:>24} {arm.auc:8.4f} {fas}")
    print(f"\n  A->B (the price of assuming Gaussian): {arms['A'].auc - arms['B'].auc:+.4f} AUC")
    print(f"  B->C (recovered by retraining)        : {arms['C'].auc - arms['B'].auc:+.4f} AUC")
    print(f"  A->C (reality's remaining tax)        : {arms['A'].auc - arms['C'].auc:+.4f} AUC")
    print(f"  background: {arms['C'].hours:.2f} h of held-out real noise "
          f"(FA/h supportable down to ~{1/arms['C'].hours:.1f})")

    for k in ("B", "C"):
        arm = arms[k]
        print(f"\nefficiency (TPR) per SNR bin — arm {k} ({arm.name}):")
        print(f"{'SNR bin':>10} " + " ".join(f"FAP {fap:.0e}".rjust(11) for fap in FAPS))
        effs = {fap: arm.efficiency(fap) for fap in FAPS}
        for i, (a, b) in enumerate(zip(SNR_EDGES[:-1], SNR_EDGES[1:])):
            row = " ".join(f"{effs[fap][0][i]:11.3f}" for fap in FAPS)
            print(f"{a:4.0f}-{b:<5.0f} {row}")

    # --- the checks that can fail ------------------------------------------------------
    effC = arms["C"].efficiency(1e-2)[0]
    transfer_ok = all(
        arms["C"].measured_fap[fap] <= 3 * fap and arms["C"].measured_fap[fap] >= fap / 3
        for fap in (1e-1, 1e-2)
    )
    checks = [
        (arms["C"].auc > arms["B"].auc,
         f"retraining beats transfer on real noise ({arms['C'].auc:.4f} > {arms['B'].auc:.4f})"),
        (effC[-1] >= 0.90,
         f"arm C efficiency {effC[-1]:.3f} at SNR 18-20 @ FAP 1e-2 >= 0.90 (finds loud signals)"),
        (effC[0] <= 0.80,
         f"arm C efficiency {effC[0]:.3f} at SNR 4-6 @ FAP 1e-2 <= 0.80 (fails at low SNR, as it must)"),
        (transfer_ok,
         "arm C thresholds transfer val->test within 3x at FAP 1e-1, 1e-2 "
         f"(measured {arms['C'].measured_fap[1e-1]:.2e}, {arms['C'].measured_fap[1e-2]:.2e})"),
    ]
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")
    if arms["C"].auc > 0.999:
        print("  SUSPICIOUS: arm C AUC > 0.999 on real noise — assume leakage before genius.")

    # --- the figure -----------------------------------------------------------------------
    fig, (axm, axg, axr) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # (1) THE money plot: three arms at FAP 1e-2, Gaussian ceiling for scale
    rho = np.linspace(SNR_EDGES[0], SNR_EDGES[-1], 200)
    for k, c, ls in (("A", "tab:gray", "--"), ("B", "tab:red", "-"), ("C", "tab:blue", "-")):
        arm = arms[k]
        t, lo, hi = arm.efficiency(1e-2)
        yerr = [np.maximum(t - lo, 0), np.maximum(hi - t, 0)]
        axm.errorbar(CENTRES, t, yerr=yerr, fmt="o" + ls, color=c, lw=2, capsize=3,
                     label=f"{k}: {arm.name}")
    axm.plot(rho, normal.cdf(rho - normal.ppf(1 - 1e-2)), "k:", lw=1.4,
             label="Gaussian ceiling (no longer a theorem here)")
    axm.set_xlabel("injected optimal SNR (in-band, vs the block's measured PSD)")
    axm.set_ylabel("detection efficiency (TPR)")
    axm.set_ylim(0, 1.02)
    axm.set_title("The Stage 2 money plot — FAP 1e-2\nA->B is the price of assuming Gaussian; B->C is what learning buys back")
    axm.legend(loc="lower right", fontsize=8)
    axm.grid(alpha=0.3)

    # (2) arm C at all three FAPs — the Gabbard figure, real-noise edition
    for fap, c in zip(FAPS, FAP_COLORS):
        t, lo, hi = arms["C"].efficiency(fap)
        yerr = [np.maximum(t - lo, 0), np.maximum(hi - t, 0)]
        axg.errorbar(CENTRES, t, yerr=yerr, fmt="o-", color=c, lw=2, capsize=3,
                     label=f"CNN @ FAP {fap:.0e}")
        axg.plot(rho, normal.cdf(rho - normal.ppf(1 - fap)), ls="--", color=c, lw=1.1, alpha=0.6)
    axg.plot([], [], "k--", lw=1.1, alpha=0.6, label="Gaussian ceiling")
    axg.set_xlabel("injected optimal SNR")
    axg.set_ylabel("detection efficiency (TPR)")
    axg.set_ylim(0, 1.02)
    axg.set_title("Stage-2 model on real O3 noise\n(distance below dashed = the non-Gaussianity tax)")
    axg.legend(loc="lower right", fontsize=8)
    axg.grid(alpha=0.3)

    # (3) ROC on real noise: transfer vs retrained
    for k, c in (("B", "tab:red"), ("C", "tab:blue")):
        arm = arms[k]
        fpr, tpr, _ = roc_curve(arm.y, arm.test_scores)
        axr.semilogx(fpr, tpr, color=c, lw=2, label=f"{k}: {arm.name} — AUC {arm.auc:.4f}")
    fprA, tprA, _ = roc_curve(arms["A"].y, arms["A"].test_scores)
    axr.semilogx(fprA, tprA, color="tab:gray", lw=1.4, ls="--",
                 label=f"A: {arms['A'].name} — AUC {arms['A'].auc:.4f}")
    axr.semilogx([1e-4, 1], [1e-4, 1], "k--", lw=1, alpha=0.4, label="chance")
    axr.set_xlim(1e-4, 1)
    axr.set_ylim(0, 1.02)
    axr.set_xlabel("false positive rate")
    axr.set_ylabel("true positive rate")
    axr.set_title("ROC — what retraining recovers on real noise")
    axr.legend(loc="lower right", fontsize=8)
    axr.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "6_eval.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

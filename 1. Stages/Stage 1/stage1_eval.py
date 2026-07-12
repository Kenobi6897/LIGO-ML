"""Stage 1, Step 6 — evaluate. NOT on accuracy.

Accuracy is meaningless here: a model that says "noise" to everything scores 50% and has
learned nothing, and a model evaluated at SNR 20 looks perfect while being useless at the
SNR where detection is actually hard. So:

  - **ROC / AUC** on the held-out test split (10k segments the training never saw).
  - **The money plot: detection efficiency (TPR) vs injected SNR at fixed false-alarm
    probability.** This is the Gabbard figure, and the whole point of Stage 1.
  - **FAR quoted as false positives per hour of held-out noise** — not per year. Our
    background is ~1.4 h of test negatives; a per-year claim needs ~10^4x more and faking
    one is worse than not making one.

THRESHOLDS ARE SET ON VAL, MEASURED ON TEST. Choosing the threshold on the same negatives
you then report FAP on makes the reported FAP true by construction — a tautology wearing
a measurement's clothes. The threshold for each target FAP is the quantile of the *val*
negatives' scores; the test split then reports the FAP that threshold actually achieves.

THE MATCHED-FILTER LINE IS A BOUND, NOT A BASELINE. For a template with known parameters,
known arrival time and known phase, the optimal statistic is Gaussian: TPR = Phi(rho - z)
at threshold z. That is the Neyman-Pearson ceiling for this problem — no search, CNN or
otherwise, beats it, because a real search doesn't know the time/phase/masses and pays
trials factors for scanning them. The CNN should sit BELOW it and degrade at low SNR like
it does. Sitting ABOVE it is not a triumph, it is a leak ([[Stage 1#Footguns]]). The real
apples-to-apples matched-filter baseline (template bank, unknown time) is Stage 4's job.

THE CHECKS THAT CAN FAIL:
  1. test AUC >= 0.90 (and SUSPICIOUS if > 0.999);
  2. efficiency at SNR 18-20 @ FAP 1e-2 >= 0.95      — it finds loud signals;
  3. efficiency at SNR 4-6   @ FAP 1e-2 <= 0.80      — it FAILS at low SNR like it must.
     A model acing the lowest bin has not out-thought Neyman-Pearson; it is reading a
     fingerprint we left on the data. This check failing is the 99%-AUC fraud alarm.
  4. thresholds transfer: measured test FAP within 2x of nominal (1e-1, 1e-2 — the
     1e-3 threshold rides on ~5 val negatives and is reported without being gated).

Run (from WSL2, after stage1_train.py):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_eval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm as normal
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from stage1_data import Stage1Dataset
from stage1_model import load_checkpoint
from stage1_train import DEFAULT_CKPT

OUT = Path(__file__).parent / "outputs"

FAPS = (1e-1, 1e-2, 1e-3)  # per-segment false-alarm probabilities, per Gabbard
FAP_COLORS = ("tab:blue", "tab:orange", "tab:green")
SNR_EDGES = np.arange(4.0, 22.0, 2.0)  # 8 bins over the injected range


def score_split(model, split: str, device) -> tuple[np.ndarray, Stage1Dataset]:
    """Logits for every segment of a split, in split order."""
    ds = Stage1Dataset(split)
    loader = DataLoader(ds, batch_size=1024, num_workers=2)
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).ravel(), ds


def wilson(k: int, n: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson score interval — sane error bars even at 0/n and n/n, where the naive
    binomial sd collapses to zero exactly where the plot needs honesty most."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 Step 6 — evaluate the CNN")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt)
    model.to(device)
    print(f"checkpoint: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f} ({args.ckpt})\n")

    val_scores, val = score_split(model, "val", device)
    test_scores, test = score_split(model, "test", device)

    # --- thresholds: set on val negatives, never on what we report -----------------
    val_neg = val_scores[val.y == 0]
    thresholds = {fap: float(np.quantile(val_neg, 1.0 - fap)) for fap in FAPS}

    test_pos = test.y == 1
    test_neg_scores = test_scores[~test_pos]
    neg_hours = len(test_neg_scores) / 3600.0  # each negative is exactly 1 s of noise

    # --- ROC / AUC ------------------------------------------------------------------
    auc = float(roc_auc_score(test.y, test_scores))
    fpr, tpr, _ = roc_curve(test.y, test_scores)
    print(f"test AUC: {auc:.4f}   ({test_pos.sum():,} pos / {(~test_pos).sum():,} neg)")

    # --- FAR, honestly: measured on test, quoted per hour of held-out noise ----------
    print(f"\nbackground: {len(test_neg_scores):,} s of held-out noise = {neg_hours:.2f} h")
    print(f"{'target FAP':>12} {'measured FAP':>14} {'false alarms':>14} {'FAR / hour':>12}")
    measured_fap = {}
    for fap in FAPS:
        n_fa = int((test_neg_scores > thresholds[fap]).sum())
        measured_fap[fap] = n_fa / len(test_neg_scores)
        print(f"{fap:>12.0e} {measured_fap[fap]:>14.2e} {n_fa:>14,} {measured_fap[fap]*3600:>12.1f}")
    print("  (per HOUR because that is what 1.4 h of background can support — a per-year"
          "\n   figure would extrapolate 4 orders of magnitude past the data. Stage 4's"
          "\n   problem, with a real background set.)")

    # --- the money plot: efficiency vs injected SNR at fixed FAP ---------------------
    snr = test.snr[test_pos]
    pos_scores = test_scores[test_pos]
    centres = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])
    eff = {}  # fap -> (tpr per bin, lo, hi)
    for fap in FAPS:
        t, lo, hi = [], [], []
        for a, b in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            in_bin = (snr >= a) & (snr < b)
            k, n = int((pos_scores[in_bin] > thresholds[fap]).sum()), int(in_bin.sum())
            t.append(k / n if n else np.nan)
            w = wilson(k, n)
            lo.append(w[0]), hi.append(w[1])
        eff[fap] = (np.array(t), np.array(lo), np.array(hi))

    print(f"\nefficiency (TPR) per SNR bin:")
    print(f"{'SNR bin':>10} " + " ".join(f"FAP {fap:.0e}".rjust(11) for fap in FAPS))
    for i, (a, b) in enumerate(zip(SNR_EDGES[:-1], SNR_EDGES[1:])):
        row = " ".join(f"{eff[fap][0][i]:11.3f}" for fap in FAPS)
        print(f"{a:4.0f}-{b:<5.0f} {row}")

    # --- the checks that can fail -----------------------------------------------------
    eff_hi = eff[1e-2][0][-1]  # SNR 18-20 @ FAP 1e-2
    eff_lo = eff[1e-2][0][0]   # SNR 4-6   @ FAP 1e-2
    transfer_ok = all(
        measured_fap[fap] <= 2 * fap and measured_fap[fap] >= fap / 2 for fap in (1e-1, 1e-2)
    )
    checks = [
        (auc >= 0.90, f"test AUC {auc:.4f} >= 0.90"),
        (eff_hi >= 0.95, f"efficiency {eff_hi:.3f} at SNR 18-20 @ FAP 1e-2 >= 0.95 (finds loud signals)"),
        (eff_lo <= 0.80, f"efficiency {eff_lo:.3f} at SNR 4-6 @ FAP 1e-2 <= 0.80 (FAILS at low SNR, as physics demands)"),
        (transfer_ok, "val-set thresholds transfer to test within 2x at FAP 1e-1, 1e-2"),
    ]
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")
    if auc > 0.999:
        print("  SUSPICIOUS: AUC > 0.999 — assume leakage before assuming genius.")

    # --- figure -----------------------------------------------------------------------
    fig, (axr, axm, axs) = plt.subplots(1, 3, figsize=(17, 5.2))

    axr.semilogx(fpr, tpr, color="tab:red", lw=2)
    axr.semilogx([1e-4, 1], [1e-4, 1], "k--", lw=1, alpha=0.5, label="chance")
    axr.set_xlim(1e-4, 1)
    axr.set_ylim(0, 1.02)
    axr.set_xlabel("false positive rate")
    axr.set_ylabel("true positive rate")
    axr.set_title(f"ROC — test split, AUC = {auc:.4f}\n(positives span SNR 4-20; AUC averages over all of it)")
    axr.legend(loc="lower right")
    axr.grid(alpha=0.3, which="both")

    rho = np.linspace(SNR_EDGES[0], SNR_EDGES[-1], 200)
    for fap, c in zip(FAPS, FAP_COLORS):
        t, lo, hi = eff[fap]
        axm.errorbar(centres, t, yerr=[t - lo, hi - t], fmt="o-", color=c, lw=2,
                     capsize=3, label=f"CNN @ FAP {fap:.0e}")
        axm.plot(rho, normal.cdf(rho - normal.ppf(1 - fap)), ls="--", color=c, lw=1.2,
                 alpha=0.7)
    axm.plot([], [], "k--", lw=1.2, alpha=0.7, label="MF bound (known signal)")
    axm.set_xlabel("injected optimal SNR (30-350 Hz, in-crop)")
    axm.set_ylabel("detection efficiency (TPR)")
    axm.set_ylim(0, 1.02)
    axm.set_title("The money plot — efficiency vs SNR at fixed FAP\n(dashed = Neyman-Pearson ceiling; sitting above it = leak)")
    axm.legend(loc="lower right", fontsize=9)
    axm.grid(alpha=0.3)

    axs.scatter(snr, pos_scores, s=4, alpha=0.15, color="tab:red", label="positives")
    axs.scatter(np.random.default_rng(0).uniform(2.0, 3.4, len(test_neg_scores)),
                test_neg_scores, s=4, alpha=0.15, color="tab:gray", label="negatives (at x<3.5)")
    for fap, c in zip(FAPS, FAP_COLORS):
        axs.axhline(thresholds[fap], color=c, lw=1.5, ls=":",
                    label=f"threshold @ FAP {fap:.0e} ({measured_fap[fap]*3600:.0f} FA/h)")
    axs.set_xlabel("injected optimal SNR (negatives drawn at x < 3.5)")
    axs.set_ylabel("CNN score (logit)")
    axs.set_title("Scores vs SNR — where the thresholds actually sit")
    axs.legend(loc="lower right", fontsize=8)
    axs.grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "6_eval.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

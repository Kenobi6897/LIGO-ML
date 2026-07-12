"""Stage 3, Step 5 — the glitch benchmark. Which morphologies fool a chirp detector?

Two arms, two test sets:
    arm C  Stage-2 model (glitch-naive — saw glitches only as unlabelled ~28/h traffic)
    arm D  Stage-3 model (glitch-trained — saw ~2.4k labelled specimens)
scored on:
    the Stage 2 test split   -> efficiency vs SNR (this must NOT degrade: the price of
                                glitch rejection must not be paid in detection)
    the Stage 3 glitch test  -> per-class false-alarm fraction

THRESHOLDS: per arm, set at FAP 1e-2 / 1e-3 on the arm's own scores over STAGE 2 VAL
negatives — ordinary real noise. The money table then reads: "at the threshold this
detector actually runs at on ordinary noise, what fraction of each glitch class fires?"

THE CHECKS THAT CAN FAIL:
  1. arm D's Stage-2-test efficiency at SNR 18-20 @ FAP 1e-2 >= 0.90, and its AUC within
     0.01 of arm C's — glitch training must not cost detection;
  2. arm D's OVERALL glitch FA fraction < arm C's at FAP 1e-2 — the specimens taught it
     something;
  3. arm D still fails at low SNR (4-6 @ 1e-2 <= 0.80) — the fraud alarm, unchanged.

Run (from WSL2, after stage3_train.py):
    python stage3_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_eval import wilson
from stage1_model import load_checkpoint

from stage2_data import Stage2Dataset
from stage2_train import DEFAULT_CKPT as STAGE2_CKPT

from stage3_data import Stage3Dataset
from stage3_train import DEFAULT_CKPT as STAGE3_CKPT

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-2, 1e-3)
SNR_EDGES = np.arange(4.0, 22.0, 2.0)


def score(model, ds, device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=1024, num_workers=2):
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Step 5 — the glitch benchmark")
    ap.add_argument("--ckpt2", type=Path, default=STAGE2_CKPT)
    ap.add_argument("--ckpt3", type=Path, default=STAGE3_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mC, cC = load_checkpoint(args.ckpt2)
    mD, cD = load_checkpoint(args.ckpt3)
    mC.to(device), mD.to(device)
    print(f"arm C: Stage-2 model (epoch {cC['epoch']}, val AUC {cC['val_auc']:.4f}) — glitch-naive")
    print(f"arm D: Stage-3 model (epoch {cD['epoch']}, val AUC {cD['val_auc']:.4f}) — glitch-trained\n")

    s2val, s2test = Stage2Dataset("val"), Stage2Dataset("test")
    g_test = Stage3Dataset("test")
    names = g_test.glitch_names
    gmask = g_test.is_glitch == 1
    print(f"stage2 test: {s2test}\nglitch test: {g_test}\n")

    arms = {}
    for key, model in (("C", mC), ("D", mD)):
        val_scores = score(model, s2val, device)
        thr = {fap: float(np.quantile(val_scores[s2val.y == 0], 1 - fap)) for fap in FAPS}
        t2 = score(model, s2test, device)
        t3 = score(model, g_test, device)
        arms[key] = dict(thr=thr, t2=t2, t3=t3,
                         auc2=float(roc_auc_score(s2test.y, t2)))

    # --- detection must not degrade -----------------------------------------------------
    print(f"stage2-test AUC:  C {arms['C']['auc2']:.4f}   D {arms['D']['auc2']:.4f}")
    eff = {}
    for key in ("C", "D"):
        a = arms[key]
        pos = s2test.y == 1
        e = []
        for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            m = pos & (s2test.snr >= lo) & (s2test.snr < hi)
            e.append(float((a["t2"][m] > a["thr"][1e-2]).mean()))
        eff[key] = np.array(e)
    print("\nefficiency @ FAP 1e-2 per SNR bin (stage2 test):")
    print(f"  {'bin':>8} " + "".join(f"{k:>8}" for k in ("C", "D")))
    for i, (lo, hi) in enumerate(zip(SNR_EDGES[:-1], SNR_EDGES[1:])):
        print(f"  {lo:3.0f}-{hi:<4.0f} {eff['C'][i]:8.3f}{eff['D'][i]:8.3f}")

    # --- the money table: per-class glitch FA fraction ------------------------------------
    print("\nglitch false-alarm fraction at thresholds set on ordinary noise:")
    print(f"  {'class':22s} {'n':>5} " +
          " ".join(f"C@{f:.0e} D@{f:.0e}".rjust(19) for f in FAPS))
    per_class = {}
    for li, nm in enumerate(names):
        m = gmask & (g_test.glitch_label == li)
        ng = int(m.sum())
        if ng == 0:
            continue
        row = []
        for fap in FAPS:
            fC = float((arms["C"]["t3"][m] > arms["C"]["thr"][fap]).mean())
            fD = float((arms["D"]["t3"][m] > arms["D"]["thr"][fap]).mean())
            row.extend([fC, fD])
        per_class[nm] = (ng, row)
        print(f"  {nm:22s} {ng:>5} "
              f"{row[0]:8.3f} {row[1]:8.3f}   {row[2]:8.3f} {row[3]:8.3f}")
    allg = gmask
    tot = {}
    for key in ("C", "D"):
        tot[key] = [float((arms[key]["t3"][allg] > arms[key]["thr"][fap]).mean()) for fap in FAPS]
    print(f"  {'ALL GLITCHES':22s} {int(allg.sum()):>5} "
          f"{tot['C'][0]:8.3f} {tot['D'][0]:8.3f}   {tot['C'][1]:8.3f} {tot['D'][1]:8.3f}")

    # --- checks ------------------------------------------------------------------------------
    checks = [
        (eff["D"][-1] >= 0.90 and arms["D"]["auc2"] >= arms["C"]["auc2"] - 0.01,
         f"glitch training costs no detection (D eff {eff['D'][-1]:.3f} at 18-20, "
         f"AUC {arms['D']['auc2']:.4f} vs C {arms['C']['auc2']:.4f})"),
        (tot["D"][0] < tot["C"][0],
         f"labelled specimens help: overall glitch FA @1e-2 D {tot['D'][0]:.3f} < C {tot['C'][0]:.3f}"),
        (eff["D"][0] <= 0.80,
         f"D still fails at low SNR ({eff['D'][0]:.3f} at 4-6 @ 1e-2) — no fraud"),
    ]
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")

    # --- figure --------------------------------------------------------------------------------
    fig, (axb, axe) = plt.subplots(1, 2, figsize=(15, 5.5))
    cls = [nm for nm in names if nm in per_class]
    xs = np.arange(len(cls))
    fC = [max(per_class[nm][1][2], 2e-4) for nm in cls]  # @1e-3, floored for log axis
    fD = [max(per_class[nm][1][3], 2e-4) for nm in cls]
    axb.bar(xs - 0.2, fC, 0.4, color="tab:red", label="C: glitch-naive")
    axb.bar(xs + 0.2, fD, 0.4, color="tab:blue", label="D: glitch-trained")
    axb.axhline(1e-3, color="k", lw=1, ls=":", label="the FAP the threshold promises")
    axb.set_yscale("log")
    axb.set_xticks(xs, cls, rotation=45, ha="right", fontsize=7)
    axb.set_ylabel("fraction of class above threshold @ FAP 1e-3")
    axb.set_title("Which glitches fool a chirp detector\n(bars at floor = zero measured)")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3, axis="y")

    centres = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])
    for key, c in (("C", "tab:red"), ("D", "tab:blue")):
        axe.plot(centres, eff[key], "o-", color=c, lw=2,
                 label=f"{key} — AUC {arms[key]['auc2']:.4f}")
    axe.set_xlabel("injected optimal SNR")
    axe.set_ylabel("efficiency @ FAP 1e-2 (stage2 test)")
    axe.set_ylim(0, 1.02)
    axe.set_title("The price check: glitch training must not cost detection")
    axe.legend(loc="lower right", fontsize=8)
    axe.grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "6_eval.png", dpi=130)
    print(f"\n  plot -> {OUT / '6_eval.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

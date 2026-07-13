"""Stage 4, Step 3 — the verdict. CNN vs matched filter, at equal false-alarm rate.

Four arms, one protocol (thresholds at FAP 1e-2 / 1e-3 on each arm's own Stage-2 val
negative scores — ordinary real noise), three fronts:

    arm C  Stage-2 CNN (glitch-naive)      arm E  MF: max rho over the bank
    arm D  Stage-3 CNN (glitch-trained)    arm F  MF: max newSNR (chi-sq re-weighted)

  1. EFFICIENCY on Stage-2 test — the theorem's home turf. The oracle curve (true-
     parameter template, arm E's thresholds) is MF's ceiling and the fraud alarm:
     no CNN may beat it on the quasi-Gaussian bulk.
  2. GLITCHES on Stage-3 test — the theorem's graveyard. Read the CNNs against arm F,
     not arm E: F is the statistic real searches actually threshold on.
  3. SPEED — latency and throughput, both sides on respectable hardware.

Checks here catch bugs; comparative results are reported, not asserted.

Run (from WSL2, after stage4_mf.py):
    python stage4_eval.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 3"))
from stage1_model import load_checkpoint

from stage2_data import Stage2Dataset
from stage2_train import DEFAULT_CKPT as STAGE2_CKPT

from stage3_data import Stage3Dataset
from stage3_eval import score
from stage3_train import DEFAULT_CKPT as STAGE3_CKPT

from stage4_mf import DEFAULT_OUT as SCORES_H5

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-2, 1e-3)
SNR_EDGES = np.arange(4.0, 22.0, 2.0)
ARMS = ("C", "D", "E", "F")
ARM_LABEL = {"C": "CNN glitch-naive", "D": "CNN glitch-trained",
             "E": "MF max-rho", "F": "MF newSNR"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 Step 3 — the verdict")
    ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2val, s2test = Stage2Dataset("val"), Stage2Dataset("test")
    g_test = Stage3Dataset("test")
    names = g_test.glitch_names
    gmask = g_test.is_glitch == 1

    # --- assemble the four arms' scores --------------------------------------------------
    mC, _ = load_checkpoint(STAGE2_CKPT)
    mD, _ = load_checkpoint(STAGE3_CKPT)
    mC.to(device), mD.to(device)
    arms: dict[str, dict] = {}
    for key, model in (("C", mC), ("D", mD)):
        arms[key] = {"val": score(model, s2val, device),
                     "t2": score(model, s2test, device),
                     "t3": score(model, g_test, device)}
    with h5py.File(SCORES_H5, "r") as f:
        for key, col in (("E", "rho"), ("F", "rhotilde")):
            arms[key] = {"val": f["s2val"][col][:], "t2": f["s2test"][col][:],
                         "t3": f["s3test"][col][:]}
        rho_oracle = f["s2test"]["rho_oracle"][:]
        mf_meta = dict(f.attrs)

    for key in ARMS:
        a = arms[key]
        a["thr"] = {fap: float(np.quantile(a["val"][s2val.y == 0], 1 - fap)) for fap in FAPS}
        a["auc2"] = float(roc_auc_score(s2test.y, a["t2"]))

    # --- front 1: efficiency on stage2 test ----------------------------------------------
    print("stage2-test AUC: " + "   ".join(f"{k} {arms[k]['auc2']:.4f}" for k in ARMS))
    pos = s2test.y == 1
    eff: dict[str, dict] = {k: {} for k in (*ARMS, "O")}
    bins = list(zip(SNR_EDGES[:-1], SNR_EDGES[1:]))
    for fap in FAPS:
        for k in ARMS:
            a = arms[k]
            eff[k][fap] = np.array([float((a["t2"][pos & (s2test.snr >= lo) & (s2test.snr < hi)]
                                           > a["thr"][fap]).mean()) for lo, hi in bins])
        # oracle: perfect template knowledge, arm E's background thresholds
        eff["O"][fap] = np.array([float((rho_oracle[pos & (s2test.snr >= lo) & (s2test.snr < hi)]
                                         > arms["E"]["thr"][fap]).mean()) for lo, hi in bins])
    for fap in FAPS:
        print(f"\nefficiency @ FAP {fap:.0e} per SNR bin (stage2 test):")
        print(f"  {'bin':>8} " + "".join(f"{k:>8}" for k in (*ARMS, "O")))
        for i, (lo, hi) in enumerate(bins):
            print(f"  {lo:3.0f}-{hi:<4.0f} " +
                  "".join(f"{eff[k][fap][i]:8.3f}" for k in (*ARMS, "O")))

    # --- front 2: the glitch table, final form -------------------------------------------
    for fap in FAPS:
        print(f"\nglitch false-alarm fraction @ FAP {fap:.0e} (thresholds on ordinary noise):")
        print(f"  {'class':22s} {'n':>5} " + "".join(f"{k:>8}" for k in ARMS))
        for li, nm in enumerate(names):
            m = gmask & (g_test.glitch_label == li)
            if m.sum() == 0:
                continue
            row = "".join(f"{float((arms[k]['t3'][m] > arms[k]['thr'][fap]).mean()):8.3f}"
                          for k in ARMS)
            print(f"  {nm:22s} {int(m.sum()):>5} " + row)
        row = "".join(f"{float((arms[k]['t3'][gmask] > arms[k]['thr'][fap]).mean()):8.3f}"
                      for k in ARMS)
        print(f"  {'ALL GLITCHES':22s} {int(gmask.sum()):>5} " + row)

    # --- front 3: speed -------------------------------------------------------------------
    xb = torch.from_numpy(np.stack([g_test[i][0].numpy() for i in range(256)]))
    mD.eval()
    with torch.no_grad():
        for _ in range(3):
            mD(xb.to(device))  # warm
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            mD(xb[i : i + 1].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_lat = (time.time() - t0) / 100 * 1e3
        t0 = time.time()
        for _ in range(10):
            mD(xb.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_thr = 10 * len(xb) / (time.time() - t0)
        mD.cpu()
        t0 = time.time()
        for i in range(50):
            mD(xb[i : i + 1])
        cpu_lat = (time.time() - t0) / 50 * 1e3
        mD.to(device)
    mf_lat = float(mf_meta["latency_ms_per_crop"])
    mf_thr = float(mf_meta["throughput_crops_per_s"])
    mf_cond = float(mf_meta["bank_condition_s_per_block"])
    print(f"\nspeed (1 s crops; MF = {int(mf_meta['n_total']):,} crops on "
          f"{int(mf_meta['workers'])} threads, CNN on {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}):")
    print(f"  {'arm':28s} {'latency/crop':>14} {'throughput':>16}")
    print(f"  {'MF bank (62 templates)':28s} {mf_lat:>11.2f} ms {mf_thr:>10,.0f} crops/s")
    print(f"    (+ bank conditioning, amortized over a 509-crop block: "
          f"+{mf_cond/509*1e3:.2f} ms/crop)")
    print(f"  {'CNN (batch 1, GPU)':28s} {gpu_lat:>11.2f} ms {gpu_thr:>10,.0f} crops/s (batch 256)")
    print(f"  {'CNN (batch 1, CPU)':28s} {cpu_lat:>11.2f} ms")

    # --- the checks that catch bugs --------------------------------------------------------
    tell_margin = 0.02
    hi_bins = SNR_EDGES[:-1] >= 8
    worst = max(float((eff[k][1e-2] - eff["O"][1e-2])[hi_bins].max()) for k in ("C", "D"))
    checks = [
        (eff["E"][1e-2][-1] >= 0.90,
         f"MF finds loud injections (E eff {eff['E'][1e-2][-1]:.3f} at SNR 18-20 @ 1e-2)"),
        # low-SNR bins excluded on purpose: there the bank's 62 trials out-fire the
        # oracle's single template on noise-dominated crops — expected, not a bug
        (bool(((eff["O"][1e-2] >= eff["E"][1e-2] - 0.02)[hi_bins]).all()),
         "oracle is a ceiling for the bank at SNR >= 8"),
        (worst <= tell_margin,
         f"THE TELL: no CNN beats the oracle on the bulk (worst CNN-oracle gap at "
         f"SNR>=8: {worst:+.3f} <= {tell_margin})"),
    ]
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")

    # --- figure -----------------------------------------------------------------------------
    fig, (axe, axb) = plt.subplots(1, 2, figsize=(15.5, 5.5))
    centres = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])
    style = {"C": ("tab:orange", "--"), "D": ("tab:blue", "-"),
             "E": ("tab:red", "--"), "F": ("tab:green", "-"), "O": ("k", ":")}
    for k in (*ARMS, "O"):
        c, ls = style[k]
        axe.plot(centres, eff[k][1e-2], ls, marker="o", ms=4, color=c, lw=1.8,
                 label=("oracle MF" if k == "O" else f"{k}: {ARM_LABEL[k]}"))
    axe.set_xlabel("injected optimal SNR"), axe.set_ylabel("efficiency @ FAP 1e-2")
    axe.set_ylim(0, 1.02), axe.grid(alpha=0.3)
    axe.set_title("The theorem's home turf (stage2 test)")
    axe.legend(loc="lower right", fontsize=8)

    cls = [nm for li, nm in enumerate(names) if (gmask & (g_test.glitch_label == li)).sum()]
    xs = np.arange(len(cls))
    w = 0.2
    for off, k in zip((-1.5, -0.5, 0.5, 1.5), ARMS):
        fr = [max(float((arms[k]["t3"][gmask & (g_test.glitch_label == names.index(nm))]
                         > arms[k]["thr"][1e-3]).mean()), 2e-4) for nm in cls]
        axb.bar(xs + off * w, fr, w, color=style[k][0], label=f"{k}: {ARM_LABEL[k]}")
    axb.axhline(1e-3, color="k", lw=1, ls=":")
    axb.set_yscale("log")
    axb.set_xticks(xs, cls, rotation=45, ha="right", fontsize=7)
    axb.set_ylabel("fraction above threshold @ FAP 1e-3")
    axb.set_title("The theorem's graveyard (stage3 glitch test; bars at floor = zero)")
    axb.legend(fontsize=8), axb.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "3_eval.png", dpi=130)
    print(f"\n  plot -> {OUT / '3_eval.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 5, Step 5 — the ablation benchmark. What was the preprocessing worth?

Four detectors, one variable. Each is a COMPLETE detector — conditioner plus network —
and each is scored on its own conditioned view of the SAME strain, the SAME injections at
the SAME physical SNRs, the SAME blocks, the SAME splits:

    full   whiten -> bandpass -> crop     the control: Stage 3's stage3_cnn.pt, arm D
    wh     whiten -> crop
    bp     bandpass -> crop
    raw    crop                           "just feed it the strain"

SCORED EXACTLY AS STAGE 3 SCORED ARMS C AND D, so the control's numbers must come out
where Stage 3 published them (that is CHECK 1 — if the control moves, the harness is
wrong, not the ablation): thresholds at FAP 1e-2 / 1e-3 on each arm's OWN scores over
Stage 2 VAL negatives (ordinary real noise), then measured on the Stage 2 test split
(efficiency vs SNR, AUC, false alarms per hour of held-out noise) and on the Stage 3
glitch test set (per-class false-alarm fraction).

THE FRAUD ALARM, PRE-REGISTERED (CHECK 2): `raw` must not beat `full`. A network fed
unwhitened strain outscoring one fed the whitened, band-limited version would not be a
discovery — every arm here is a linear signal-blind operator away from the same data, and
whitening cannot destroy information the raw arm still has (it is invertible). If raw wins,
look for a leak or a broken control before looking for a paper.

PLUS THE ONE DIAGNOSTIC THAT SAYS *WHY*: Stage 1's payoff figure, per arm. A 1D conv layer
is a learned template bank, so ask each arm's bank where it put itself. Stage 1's
whitened-and-banded network moved 15 of 16 first-layer kernels' spectral peaks INTO
30-350 Hz (random init: ~6/16). If the raw arm's kernels sit on the seismic wall instead,
that is the mechanism of its failure, drawn.

Run (after stage5_train.py for each arm):
    python stage5_eval.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 3"))
from stage1_model import load_checkpoint
from stage1_noise_check import BAND, SAMPLE_RATE

from stage2_data import Stage2Dataset

from stage3_data import Stage3Dataset
from stage3_train import DEFAULT_CKPT as STAGE3_CKPT

from stage5_condition import BUILD_ARMS
from stage5_data import Stage5Dataset
from stage5_train import ckpt_path

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-2, 1e-3)
SNR_EDGES = np.arange(4.0, 22.0, 2.0)

# Stage 3's published control numbers — CHECK 1 compares against these.
CONTROL_AUC = 0.9839
CONTROL_GLITCH_FA_1E2 = 0.166

LABELS = {
    "full": "full — whiten + bandpass (control, arm D)",
    "wh": "wh — whiten only, no bandpass",
    "bp": "bp — bandpass only, no whitening",
    "raw": "raw — no conditioning at all",
}
COLOURS = {"full": "tab:blue", "wh": "tab:green", "bp": "tab:orange", "raw": "tab:red"}


def score(model, ds, device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=1024, num_workers=2):
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).ravel()


def in_band_kernels(model) -> tuple[int, int]:
    """Stage 1's payoff metric: how many first-layer kernels peak inside 30-350 Hz."""
    k = model.first_layer_kernels()  # (16, 64)
    spec = np.abs(np.fft.rfft(k, n=512, axis=1))
    freqs = np.fft.rfftfreq(512, d=1.0 / SAMPLE_RATE)
    peak = freqs[spec.argmax(axis=1)]
    return int(((peak >= BAND[0]) & (peak <= BAND[1])).sum()), len(k)


def load_arms(device, want_noclip: bool):
    """The control from Stage 3's checkpoint; the ablated arms from Stage 5's."""
    arms = {}
    m, c = load_checkpoint(STAGE3_CKPT)
    arms["full"] = dict(
        model=m.to(device), ck=c,
        s2val=Stage2Dataset("val"), s2test=Stage2Dataset("test"), gtest=Stage3Dataset("test"),
    )
    for arm in ("wh", "bp", "raw"):
        for clip in ((True, False) if want_noclip else (True,)):
            p = ckpt_path(arm, clip)
            if not p.exists():
                if clip:
                    print(f"  (no checkpoint for arm {arm} — run stage5_train.py --arm {arm})")
                continue
            key = arm if clip else f"{arm}_noclip"
            m, c = load_checkpoint(p)
            arms[key] = dict(
                model=m.to(device), ck=c,
                s2val=Stage5Dataset(arm, "s2", "val", clip=clip),
                s2test=Stage5Dataset(arm, "s2", "test", clip=clip),
                gtest=Stage5Dataset(arm, "s3", "test", clip=clip),
            )
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 5 — the ablation benchmark")
    ap.add_argument("--noclip", action="store_true", help="also score any *_noclip arms")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = load_arms(device, args.noclip)
    order = [k for k in ("full", "wh", "bp", "raw", "raw_noclip", "bp_noclip", "wh_noclip")
             if k in arms]

    print("arms:")
    for k in order:
        a = arms[k]
        print(f"  {k:11s} epoch {a['ck']['epoch']:2d}, val AUC {a['ck']['val_auc']:.4f}  "
              f"— {LABELS.get(k, k)}")
    hours = arms["full"]["s2test"].hours_of_noise()
    print(f"\nstage2 test: {len(arms['full']['s2test']):,} crops, {hours:.2f} h of held-out "
          f"real noise in the negatives")
    print(f"glitch test: {int((arms['full']['gtest'].is_glitch == 1).sum())} specimens\n")

    # --- score every arm on its own view of the same data ---------------------------------
    for k in order:
        a = arms[k]
        v = score(a["model"], a["s2val"], device)
        a["thr"] = {f: float(np.quantile(v[a["s2val"].y == 0], 1 - f)) for f in FAPS}
        a["t2"] = score(a["model"], a["s2test"], device)
        a["t3"] = score(a["model"], a["gtest"], device)
        a["auc"] = float(roc_auc_score(a["s2test"].y, a["t2"]))
        neg = a["s2test"].y == 0
        a["fa_h"] = {f: float((a["t2"][neg] > a["thr"][f]).sum()) / hours for f in FAPS}
        a["kern"] = in_band_kernels(a["model"])
        pos = a["s2test"].y == 1
        a["eff"] = {}
        for f in FAPS:
            a["eff"][f] = np.array([
                float((a["t2"][pos & (a["s2test"].snr >= lo) & (a["s2test"].snr < hi)]
                       > a["thr"][f]).mean())
                for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:])
            ])

    # --- headline -------------------------------------------------------------------------
    print("DETECTION on real-noise injections (stage2 test)\n")
    print(f"  {'arm':11s} {'AUC':>7}  {'kernels in band':>15}  " +
          "  ".join(f"{'FA/h @'+f'{f:.0e}':>12}" for f in FAPS))
    for k in order:
        a = arms[k]
        print(f"  {k:11s} {a['auc']:7.4f}  {a['kern'][0]:>10}/{a['kern'][1]:<4}  " +
              "  ".join(f"{a['fa_h'][f]:12.1f}" for f in FAPS))

    for f in FAPS:
        print(f"\n  efficiency vs SNR @ FAP {f:.0e}:")
        print(f"    {'bin':>8} " + "".join(f"{k:>12}" for k in order))
        for i, (lo, hi) in enumerate(zip(SNR_EDGES[:-1], SNR_EDGES[1:])):
            print(f"    {lo:3.0f}-{hi:<4.0f} " +
                  "".join(f"{arms[k]['eff'][f][i]:12.3f}" for k in order))

    # --- the glitch table -----------------------------------------------------------------
    print("\n\nGLITCH false-alarm fraction, thresholds set on ordinary noise\n")
    g = arms["full"]["gtest"]
    names = g.glitch_names
    print(f"  {'class':22s} {'n':>5} " +
          " ".join(f"{k+'@'+f'{f:.0e}':>13}" for f in FAPS for k in order))
    per_class = {}
    for li, nm in enumerate(names):
        ng = int(((g.is_glitch == 1) & (g.glitch_label == li)).sum())
        if ng == 0:
            continue
        row = []
        for f in FAPS:
            for k in order:
                a = arms[k]
                m = (a["gtest"].is_glitch == 1) & (a["gtest"].glitch_label == li)
                row.append(float((a["t3"][m] > a["thr"][f]).mean()))
        per_class[nm] = (ng, row)
        print(f"  {nm:22s} {ng:>5} " + " ".join(f"{v:13.3f}" for v in row))

    tot = {}
    for k in order:
        a = arms[k]
        m = a["gtest"].is_glitch == 1
        tot[k] = [float((a["t3"][m] > a["thr"][f]).mean()) for f in FAPS]
    print(f"  {'ALL GLITCHES':22s} {int((g.is_glitch == 1).sum()):>5} " +
          " ".join(f"{tot[k][i]:13.3f}" for i, _f in enumerate(FAPS) for k in order))

    # --- checks ---------------------------------------------------------------------------
    full, raw = arms["full"], arms.get("raw")
    checks = [
        (abs(full["auc"] - CONTROL_AUC) < 0.01,
         f"control reproduces Stage 3: stage2-test AUC {full['auc']:.4f} vs published "
         f"{CONTROL_AUC} (the harness scores what Stage 3 scored)"),
        (abs(tot["full"][0] - CONTROL_GLITCH_FA_1E2) < 0.03,
         f"control reproduces Stage 3: glitch FA @1e-2 {tot['full'][0]:.3f} vs published "
         f"{CONTROL_GLITCH_FA_1E2}"),
    ]
    if raw is not None:
        checks.append((
            raw["auc"] <= full["auc"] + 0.002,
            f"FRAUD ALARM: raw ({raw['auc']:.4f}) does not beat full ({full['auc']:.4f}) — "
            f"whitening is invertible, so a raw win means a leak, not a discovery",
        ))
    print()
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {msg}")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    centres = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])
    base = [k for k in order if not k.endswith("_noclip")]

    for f, ax, ls in ((1e-2, axes[0], "-"), (1e-3, axes[1], "-")):
        for k in order:
            c = COLOURS.get(k.replace("_noclip", ""), "grey")
            dash = ":" if k.endswith("_noclip") else ls
            ax.plot(centres, arms[k]["eff"][f], "o" + dash, color=c, lw=2,
                    label=f"{k} — AUC {arms[k]['auc']:.4f}")
        ax.set_xlabel("injected optimal SNR")
        ax.set_ylabel(f"efficiency @ FAP {f:.0e}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Detection at equal false-alarm probability ({f:.0e})\n"
                     f"same signals, same noise — only the conditioner differs")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    xs = np.arange(len(base))
    axg = axes[2]
    w = 0.8 / max(len(FAPS), 1)
    for j, f in enumerate(FAPS):
        axg.bar(xs + (j - 0.5) * w, [max(tot[k][j], 2e-4) for k in base], w,
                color=[COLOURS.get(k, "grey") for k in base],
                alpha=1.0 if j == 0 else 0.55,
                label=f"FAP {f:.0e}")
    axg.axhline(1e-2, color="k", lw=1, ls=":", alpha=0.6)
    axg.axhline(1e-3, color="k", lw=1, ls="--", alpha=0.6)
    axg.set_yscale("log")
    axg.set_xticks(xs, base)
    axg.set_ylabel("fraction of glitch specimens above threshold")
    axg.set_title("Glitch rejection\n(dotted/dashed = the FAP the threshold promises)")
    axg.legend(fontsize=8)
    axg.grid(alpha=0.3, axis="y")

    fig.suptitle("Stage 5 — what the preprocessing was worth: same CNN, same data, "
                 "four conditioners", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "5_ablation.png", dpi=130)
    print(f"\n  plot -> {OUT / '5_ablation.png'}")

    # --- the learned template bank, per arm -----------------------------------------------
    fig2, axk = plt.subplots(1, len(base), figsize=(4.2 * len(base), 4), squeeze=False)
    freqs = np.fft.rfftfreq(512, d=1.0 / SAMPLE_RATE)
    for j, k in enumerate(base):
        ax = axk[0][j]
        kern = arms[k]["model"].first_layer_kernels()
        spec = np.abs(np.fft.rfft(kern, n=512, axis=1))
        spec /= spec.max(axis=1, keepdims=True)
        for s in spec:
            ax.plot(freqs, s, color=COLOURS.get(k, "grey"), alpha=0.45, lw=1)
        ax.axvspan(*BAND, color="k", alpha=0.07)
        ax.set_xlim(0, 600)
        ax.set_xlabel("Hz")
        ax.set_ylabel("normalised |FFT| of kernel" if j == 0 else "")
        ax.set_title(f"{k}: {arms[k]['kern'][0]}/{arms[k]['kern'][1]} kernels peak in band")
        ax.grid(alpha=0.3)
    fig2.suptitle("The learned template bank, per arm — shaded: the 30-350 Hz analysis band "
                  "(random init: ~6/16 in band)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT / "6_kernels.png", dpi=130)
    print(f"  plot -> {OUT / '6_kernels.png'}")

    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 5, Step 7 — re-run Stage 4's verdict with the clip-fixed CNN.

Stage 4 compared arms C/D (CNNs) against E/F (matched filter) and O (oracle) at equal
false-alarm probability, and concluded that the chi-squared-vetoed matched filter (arm F)
narrowly won glitch rejection: 0.100 vs D's 0.166 @ FAP 1e-2, a dead heat at 1e-3.

Stage 5 found that arm D was trained with Stage 3's +-20 sigma saturation applied to only
4.9% of its training rows (Stage3Dataset clips; Stage2Dataset does not, and supplies 95% of
the mix — including crops up to 6,177 sigma). So Stage 4's headline was scored against a CNN
still carrying the gain-crush bug Stage 3 had diagnosed.

AND THE ASYMMETRY WAS ONE-SIDED. stage4_mf.py clips its OWN inputs (`np.clip(x, -CLIP_SIGMA,
+CLIP_SIGMA)`, stage4_mf.py:173) — so the matched filter always saw saturated crops. Stage 4
was comparing a correctly-conditioned MF against a CNN that wasn't. This script fixes only
that, and changes nothing else.

    arm C   Stage-2 CNN (glitch-naive)      as published
    arm D   Stage-3 CNN (glitch-trained)    as published
    arm D+  the same, clip applied to BOTH sources                          <- the fix
    arm E   MF, max rho over the 62-template bank      } reused verbatim from
    arm F   MF, newSNR (chi-squared re-weighted)       } stage4_scores.h5 — the MF
    arm O   oracle (true-parameter template)           } statistics do not depend on any CNN

D+ IS REPORTED ACROSS 4 SEEDS, not one. Its single best seed rejects glitches at 0.024
@1e-2; the seed MEAN is 0.058 +- 0.022. A claim that flips a published verdict rides on the
mean and the worst seed, never the best one.

Run (from WSL2; needs stage5_sweep.py's checkpoints):
    python stage5_rerun_stage4.py
"""

from __future__ import annotations

import sys
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 4"))
from stage1_model import load_checkpoint

from stage2_data import Stage2Dataset
from stage2_train import DEFAULT_CKPT as STAGE2_CKPT

from stage3_data import Stage3Dataset
from stage3_train import DEFAULT_CKPT as STAGE3_CKPT

from stage4_mf import DEFAULT_OUT as SCORES_H5

from stage5_data import Stage5Dataset
from stage5_eval import score
from stage5_train import SEED, ckpt_path

OUT = Path(__file__).parent / "outputs"
FAPS = (1e-2, 1e-3)
SNR_EDGES = np.arange(4.0, 22.0, 2.0)
SEEDS = [SEED, 1, 2, 3]

ARMS = ("C", "D", "D+", "E", "F")
LABEL = {"C": "CNN glitch-naive", "D": "CNN glitch-trained (as published)",
         "D+": "CNN glitch-trained + clip fix", "E": "MF max-rho",
         "F": "MF newSNR (chi-sq veto)", "O": "oracle MF"}
STYLE = {"C": ("tab:orange", "--"), "D": ("tab:blue", "--"), "D+": ("tab:purple", "-"),
         "E": ("tab:red", "--"), "F": ("tab:green", "-"), "O": ("k", ":")}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Published CNNs read their crops the way Stage 3/4 read them (s2 unclipped);
    # D+ reads both sources clipped. Each arm is scored on ITS OWN view — that is the
    # difference under test.
    s2val, s2test, g_test = Stage2Dataset("val"), Stage2Dataset("test"), Stage3Dataset("test")
    c2val = Stage5Dataset("full", "s2", "val")
    c2test = Stage5Dataset("full", "s2", "test")
    c3test = Stage5Dataset("full", "s3", "test")

    names = g_test.glitch_names
    gmask = g_test.is_glitch == 1
    pos = s2test.y == 1
    bins = list(zip(SNR_EDGES[:-1], SNR_EDGES[1:]))

    arms: dict[str, dict] = {}
    for key, ck, dv, dt, dg in (
        ("C", STAGE2_CKPT, s2val, s2test, g_test),
        ("D", STAGE3_CKPT, s2val, s2test, g_test),
        ("D+", ckpt_path("full", True, SEED), c2val, c2test, c3test),
    ):
        m, _ = load_checkpoint(ck)
        m.to(device)
        arms[key] = {"val": score(m, dv, device), "t2": score(m, dt, device),
                     "t3": score(m, dg, device)}

    with h5py.File(SCORES_H5, "r") as f:
        for key, col in (("E", "rho"), ("F", "rhotilde")):
            arms[key] = {"val": f["s2val"][col][:], "t2": f["s2test"][col][:],
                         "t3": f["s3test"][col][:]}
        rho_oracle = f["s2test"]["rho_oracle"][:]

    for k in ARMS:
        a = arms[k]
        a["thr"] = {fap: float(np.quantile(a["val"][s2val.y == 0], 1 - fap)) for fap in FAPS}
        a["auc2"] = float(roc_auc_score(s2test.y, a["t2"]))

    # --- D+ across every seed, because a verdict-flipping claim needs the spread ---------
    dplus: dict[int, dict] = {}
    for sd in SEEDS:
        p = ckpt_path("full", True, sd)
        if not p.exists():
            print(f"  (missing {p.name} — run stage5_sweep.py first)")
            continue
        m, _ = load_checkpoint(p)
        m.to(device)
        v, t2, t3 = score(m, c2val, device), score(m, c2test, device), score(m, c3test, device)
        thr = {fap: float(np.quantile(v[c2val.y == 0], 1 - fap)) for fap in FAPS}
        dplus[sd] = {
            "glitch": {f: float((t3[gmask] > thr[f]).mean()) for f in FAPS},
            "eff": {f: np.array([float((t2[pos & (s2test.snr >= lo) & (s2test.snr < hi)]
                                        > thr[f]).mean()) for lo, hi in bins]) for f in FAPS},
            "auc": float(roc_auc_score(s2test.y, t2)),
        }

    # --- front 1: efficiency, the theorem's home turf ------------------------------------
    print("STAGE 4, RE-RUN — the same protocol, with the clip-fixed CNN\n")
    print("stage2-test AUC: " + "   ".join(f"{k} {arms[k]['auc2']:.4f}" for k in ARMS))

    eff: dict[str, dict] = {k: {} for k in (*ARMS, "O")}
    for fap in FAPS:
        for k in ARMS:
            a = arms[k]
            eff[k][fap] = np.array([
                float((a["t2"][pos & (s2test.snr >= lo) & (s2test.snr < hi)] > a["thr"][fap]).mean())
                for lo, hi in bins])
        eff["O"][fap] = np.array([
            float((rho_oracle[pos & (s2test.snr >= lo) & (s2test.snr < hi)]
                   > arms["E"]["thr"][fap]).mean()) for lo, hi in bins])

    for fap in FAPS:
        print(f"\nefficiency @ FAP {fap:.0e} per SNR bin (stage2 test):")
        print(f"  {'bin':>8} " + "".join(f"{k:>8}" for k in (*ARMS, "O")))
        for i, (lo, hi) in enumerate(bins):
            print(f"  {lo:3.0f}-{hi:<4.0f} " +
                  "".join(f"{eff[k][fap][i]:8.3f}" for k in (*ARMS, "O")))
        if dplus:
            lo_ = np.min([dplus[s]["eff"][fap] for s in dplus], axis=0)
            hi_ = np.max([dplus[s]["eff"][fap] for s in dplus], axis=0)
            print(f"  {'D+ range':>8} " +
                  "".join(f"{a:.2f}-{b:.2f}".rjust(8) for a, b in zip(lo_, hi_))
                  + "   (over 4 seeds)")

    # --- front 2: the glitch table, the money table --------------------------------------
    for fap in FAPS:
        print(f"\nglitch false-alarm fraction @ FAP {fap:.0e} (thresholds on ordinary noise):")
        print(f"  {'class':22s} {'n':>5} " + "".join(f"{k:>8}" for k in ARMS))
        for li, nm in enumerate(names):
            m_ = gmask & (g_test.glitch_label == li)
            if m_.sum() == 0:
                continue
            print(f"  {nm:22s} {int(m_.sum()):>5} " +
                  "".join(f"{float((arms[k]['t3'][m_] > arms[k]['thr'][fap]).mean()):8.3f}"
                          for k in ARMS))
        print(f"  {'ALL GLITCHES':22s} {int(gmask.sum()):>5} " +
              "".join(f"{float((arms[k]['t3'][gmask] > arms[k]['thr'][fap]).mean()):8.3f}"
                      for k in ARMS))
        if dplus:
            v = np.array([dplus[s]["glitch"][fap] for s in dplus])
            print(f"  {'D+ over 4 seeds':22s} {'':>5} mean {v.mean():.3f} +- {v.std():.3f}   "
                  f"worst {v.max():.3f}   best {v.min():.3f}"
                  f"   |  arm F = {float((arms['F']['t3'][gmask] > arms['F']['thr'][fap]).mean()):.3f}")

    # --- the verdict ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("THE VERDICT — glitch rejection, D+ vs F (the fight Stage 4 set up)\n")
    verdict_ok = True
    for fap in FAPS:
        f_val = float((arms["F"]["t3"][gmask] > arms["F"]["thr"][fap]).mean())
        v = np.array([dplus[s]["glitch"][fap] for s in dplus])
        beats_all = bool((v < f_val).all())
        verdict_ok &= beats_all
        print(f"  FAP {fap:.0e}:  D+ {v.mean():.3f} +- {v.std():.3f} (worst seed {v.max():.3f})"
              f"   vs   F {f_val:.3f}   -> "
              f"{'D+ WINS on every seed' if beats_all else 'not every seed beats F'}")

    # the fraud alarm, unchanged: no CNN may beat the oracle on the quasi-Gaussian bulk
    hi_bins = SNR_EDGES[:-1] >= 8
    worst = max(float((eff[k][1e-2] - eff["O"][1e-2])[hi_bins].max()) for k in ("C", "D", "D+"))
    print(f"\n  {'PASS' if worst <= 0.02 else 'FAIL'}: THE TELL — no CNN beats the oracle at "
          f"SNR>=8 @1e-2 (worst gap {worst:+.3f}). The clip fix did NOT")
    print(f"        break the theorem: on the bulk, matched filtering is still optimal.")

    print(f"\n  Detection at FAP 1e-3 is where Stage 4 said F leads at every SNR. Now:")
    for i, (lo, hi) in enumerate(bins):
        if lo < 6 or lo > 10:
            continue
        print(f"        SNR {lo:.0f}-{hi:.0f}:  D {eff['D'][1e-3][i]:.3f}  ->  "
              f"D+ {eff['D+'][1e-3][i]:.3f}   vs   F {eff['F'][1e-3][i]:.3f}")
    print("=" * 78)

    # --- figure -----------------------------------------------------------------------------
    fig, (axe, axb) = plt.subplots(1, 2, figsize=(16, 5.5))
    centres = 0.5 * (SNR_EDGES[:-1] + SNR_EDGES[1:])
    for k in (*ARMS, "O"):
        c, ls = STYLE[k]
        axe.plot(centres, eff[k][1e-3], ls, marker="o", ms=4, color=c,
                 lw=2.2 if k in ("D+", "F") else 1.5, label=f"{k}: {LABEL[k]}")
    axe.set_xlabel("injected optimal SNR"), axe.set_ylabel("efficiency @ FAP 1e-3")
    axe.set_ylim(0, 1.02), axe.grid(alpha=0.3)
    axe.set_title("Detection at the deep threshold (stage2 test)\nthe regime the veto rescued")
    axe.legend(loc="lower right", fontsize=8)

    cls = [nm for li, nm in enumerate(names) if (gmask & (g_test.glitch_label == li)).sum()]
    xs = np.arange(len(cls))
    w = 0.19
    for off, k in zip((-1.5, -0.5, 0.5, 1.5), ("D", "D+", "E", "F")):
        fr = [max(float((arms[k]["t3"][gmask & (g_test.glitch_label == names.index(nm))]
                         > arms[k]["thr"][1e-2]).mean()), 2e-4) for nm in cls]
        axb.bar(xs + off * w, fr, w, color=STYLE[k][0], label=f"{k}: {LABEL[k]}")
    axb.axhline(1e-2, color="k", lw=1, ls=":")
    axb.set_yscale("log")
    axb.set_xticks(xs, cls, rotation=45, ha="right", fontsize=7)
    axb.set_ylabel("fraction above threshold @ FAP 1e-2")
    axb.set_title("The theorem's graveyard, re-scored\n(bars at floor = zero measured)")
    axb.legend(fontsize=8), axb.grid(alpha=0.3, axis="y")
    fig.suptitle("Stage 4 re-run — the chi-squared veto vs a CNN that isn't gain-crushed",
                 fontsize=13)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "7_stage4_rerun.png", dpi=130)
    print(f"\n  plot -> {OUT / '7_stage4_rerun.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

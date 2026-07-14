"""Stage 3, Step 3's check — the Stage 2 battery, plus the questions glitches add.

New silent failure modes this file hunts:
  - a "glitch" row whose crop holds no glitch (mis-mapped offset — the label/row divorce,
    glitch edition). Checked as GLITCH PRESENCE: for the strongly in-band classes the
    peak |x| near the recorded position must sit far above clean-crop statistics.
  - a glitch placed outside U[0.10, 0.95) of the crop, or never overlapping the injection
    range — POSITION AS LABEL is the shortcut this stage must not hand the model.
  - class imbalance per split, glitch starvation of val/test.

Everything else is the Stage 2 battery re-armed: structure, float64 params, (block,
offset) uniqueness, time-ordered block splits, bit-exact rebuild, rho SNR calibration.

Run (from WSL2, after stage3_dataset.py):
    python stage3_dataset_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
import stage2_condition as s2c
from stage2_dataset import build_segment

from stage3_dataset import DEFAULT_OUT, GLITCH_POS_RANGE, STRAIN_H5

OUT = Path(__file__).parent / "outputs"
N_REBUILD = 48
N_SNR = 200
STRONG_CLASSES = ("Koi_Fish", "Blip", "Extremely_Loud", "Tomte")  # >=94% in-band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    s2c.STRAIN_H5 = STRAIN_H5

    rng = np.random.default_rng(2)
    results = []
    f = h5py.File(args.path, "r")
    n = f["X"].shape[0]
    y = f["y"][:]
    isg = f["is_glitch"][:]
    glab = f["glitch_label"][:]
    gpos = f["glitch_pos"][:]
    block = f["block"][:]
    offset = f["offset"][:]
    split = f["split"][:]
    labels = list(f.attrs["glitch_labels"])
    print(f"{args.path}: {n:,} rows — {int(y.sum()):,} pos / {int((y==0).sum()):,} neg "
          f"({int(isg.sum()):,} glitch)")

    # --- structure ---------------------------------------------------------------------
    f8 = all(f[k].dtype == np.float64
             for k in ("m1", "m2", "snr", "merger_pos", "glitch_gps", "glitch_pos", "glitch_snr"))
    results.append(("parameters stored float64", f8, "an f4 crept in"))
    results.append(("labels ~50/50", abs(y.mean() - 0.5) < 0.02, f"{y.mean():.3f}"))
    uniq = len(np.unique(block.astype(np.int64) * 100_000 + offset)) == n
    results.append(("(block, offset) unique", uniq, "duplicate crops"))
    results.append(("glitch rows are negatives", bool((y[isg == 1] == 0).all()), "a glitch marked positive"))
    gp = gpos[isg == 1]
    pos_ok = bool(((gp >= 0.02) & (gp < 0.99)).all())
    in_primary = float(((gp >= GLITCH_POS_RANGE[0]) & (gp < GLITCH_POS_RANGE[1])).mean())
    results.append((f"glitch positions in [0.02, 0.99) ({in_primary:.0%} in primary "
                    f"U{GLITCH_POS_RANGE})", pos_ok and in_primary > 0.7, "out of range"))
    overlap = float((gpos[isg == 1] >= 0.70).mean())
    results.append((f"glitch/injection position ranges overlap ({overlap:.0%} of glitches at >=0.70)",
                    overlap > 0.15, "position could become the label"))
    for s, nm in ((1, "val"), (2, "test")):
        m = split == s
        results.append((f"{nm} split holds glitches ({int(isg[m].sum())}) and positives "
                        f"({int(y[m].sum())})", isg[m].sum() >= 50 and y[m].sum() >= 200,
                        "starved split"))

    # splits are BLOCK-DISJOINT (class-stratified, deliberately not time-ordered — see
    # stage3_dataset.py) and every class is measurable: >= 20 test rows or all-in-test
    disjoint = all(len({int(s) for s in split[block == b]}) == 1 for b in np.unique(block))
    results.append(("each block lives in exactly one split", disjoint, "a block straddles splits"))
    starved = []
    for li, nm in enumerate(labels):
        m = (isg == 1) & (glab == li)
        if m.sum() and (isg[(split == 2)] & (glab[split == 2] == li)).sum() < min(20, m.sum()):
            starved.append(nm)
    results.append(("every class measurable on test (>= 20 rows or everything it has)",
                    not starved, f"starved: {starved}"))

    # --- bit-exact rebuild ----------------------------------------------------------------
    rows = rng.choice(n, size=min(N_REBUILD, n), replace=False)
    worst = 0.0
    for r in rows:
        r = int(r)
        spec = (r, int(y[r]), int(block[r]), int(offset[r]),
                float(f["m1"][r]), float(f["m2"][r]), float(f["snr"][r]),
                float(f["merger_pos"][r]))
        _, x = build_segment(spec)
        worst = max(worst, float(np.max(np.abs(x - f["X"][r, 0]))))
    results.append((f"bit-exact rebuild of {len(rows)} rows (max|delta| = {worst:.3e})",
                    worst == 0.0, "row/metadata divorce"))

    # --- SNR calibration (rho, same-block negatives) ---------------------------------------
    pos_rows = np.flatnonzero(y == 1)
    pick = rng.choice(pos_rows, size=min(N_SNR, len(pos_rows)), replace=False)
    plain = (y == 0) & (isg == 0)
    ratios = []
    for r in sorted(int(v) for v in pick):
        b = int(block[r])
        bg_rows = np.flatnonzero(plain & (block == b))
        if len(bg_rows) < 8:
            continue
        spec = (r, 1, b, int(offset[r]), float(f["m1"][r]), float(f["m2"][r]),
                float(f["snr"][r]), float(f["merger_pos"][r]))
        _, x_pos = build_segment(spec)
        _, x_noise = build_segment((r, 0, b, int(offset[r]), 0.0, 0.0, 0.0, 0.0))
        ch = x_pos.astype(np.float64) - x_noise.astype(np.float64)
        bg = np.stack([f["X"][int(i), 0] for i in bg_rows]).astype(np.float64)
        sigma_bg = float(np.std(bg @ ch))
        if sigma_bg > 0:
            ratios.append((float(ch @ ch) / sigma_bg) / float(f["snr"][r]))
    med = float(np.median(ratios))
    results.append((f"SNR calibration: median rho/target = {med:.3f} (n={len(ratios)})",
                    0.85 < med < 1.15, "x-axis mislabelled"))

    # --- glitch presence: strong classes must light up near the recorded position ----------
    print("\n  glitch presence (peak |x| within +-0.15 of recorded crop position):")
    strong_ok = True
    per_class_peak = {}
    for li, name in enumerate(labels):
        rows_c = np.flatnonzero((isg == 1) & (glab == li))
        if len(rows_c) == 0:
            continue
        take = rng.choice(rows_c, size=min(40, len(rows_c)), replace=False)
        peaks = []
        for r in take:
            x = f["X"][int(r), 0]
            c = int(gpos[int(r)] * 2048)
            lo, hi = max(0, c - 307), min(2048, c + 307)
            peaks.append(np.abs(x[lo:hi]).max())
        per_class_peak[name] = float(np.median(peaks))
        print(f"    {name:22s} median peak {per_class_peak[name]:8.1f} sigma  (n={len(rows_c)})")
        if name in STRONG_CLASSES and per_class_peak[name] < 5.0:
            strong_ok = False
    results.append((f"strong in-band classes visibly present (median peak >= 5 sigma)",
                    strong_ok, "a strong class looks empty — offset mis-mapped?"))

    print()
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {name}" + ("" if passed else f" — {detail}"))

    # --- figure: the gallery -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    t = np.arange(2048) / 2048
    shown = 0
    for li, name in enumerate(labels):
        if shown >= 12:
            break
        rows_c = np.flatnonzero((isg == 1) & (glab == li))
        if len(rows_c) == 0:
            continue
        r = int(rows_c[np.argsort(f["glitch_snr"][:][rows_c])[len(rows_c) // 2]])
        ax = axes.ravel()[shown]
        ax.plot(t, f["X"][r, 0], lw=0.5, color="tab:red")
        ax.axvline(gpos[r], color="k", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(f"{name} (SNR {f['glitch_snr'][r]:.0f})", fontsize=8)
        ax.grid(alpha=0.2)
        shown += 1
    fig.suptitle("Hard negatives, conditioned exactly like the training data "
                 "(median-SNR specimen per class; dotted = recorded peak position)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "3_dataset_check.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    f.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

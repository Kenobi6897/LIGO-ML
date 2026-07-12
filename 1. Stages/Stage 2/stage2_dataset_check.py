"""Stage 2, Step 3's check — the checks that caught Stage 1's four silent bugs, re-armed.

Everything here compares the file against something known independently:

  STRUCTURE      shapes, dtypes (f8 params — bug #4's rule), 50/50 within tolerance,
                 negatives carry no waveform metadata, (block, offset) pairs unique,
                 splits are truly TIME-ORDERED: every val gps > every train gps, every
                 test gps > every val gps.
  BIT-EXACT      a stored row must be reproducible from its metadata + the raw strain
  REBUILD        file, to zero. Not 1e-5 — zero. (Stage 1 bug #4: a tolerance that
                 "looks like round-off" is how a writer/label divorce hides.)
  SNR            the empirical-background rho statistic, unchanged from Stage 1 —
  CALIBRATION    correlate the conditioned template against conditioned negatives from
                 the SAME BLOCK, so sigma_bg is measured in the same noise the signal
                 sits in. It ported unchanged precisely because it never assumed the
                 noise was white, Gaussian, or stationary.
  UNIT VARIANCE  (C(h) . x) / sigma_bg == rho + N(0,1) on stored positives: proves
                 sigma_bg is the right normalisation AND that each row contains the
                 signal its metadata claims, at the offset we think.

Run (from WSL2, after stage2_dataset.py; a few minutes):
    python stage2_dataset_check.py [--path /tmp/s2sm.h5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_condition import CROP_N

import stage2_condition as s2c
from stage2_dataset import DEFAULT_OUT, build_segment, MERGER_RANGE, SNR_RANGE

OUT = Path(__file__).parent / "outputs"
N_REBUILD = 64        # rows rebuilt bit-exactly
N_SNR = 300           # positives for the rho calibration
N_BG_PER_BLOCK = 120  # same-block negatives used to measure sigma_bg


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 3 — check the dataset")
    ap.add_argument("--path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strain", type=Path, default=None,
                    help="override the strain file (smoke test)")
    args = ap.parse_args()
    if args.strain is not None:
        s2c.STRAIN_H5 = args.strain

    rng = np.random.default_rng(1)
    results: list[tuple[str, bool, str]] = []
    f = h5py.File(args.path, "r")
    n = f["X"].shape[0]
    y = f["y"][:]
    snr = f["snr"][:]
    block = f["block"][:]
    offset = f["offset"][:]
    gps = f["gps"][:]
    split = f["split"][:]
    print(f"{args.path}: {n:,} rows, attrs norm={f.attrs['norm']:.6e}")

    # --- structure -----------------------------------------------------------------
    results.append(("X shape/dtype", f["X"].shape == (n, 1, CROP_N) and f["X"].dtype == np.float32,
                    f"{f['X'].shape} {f['X'].dtype}"))
    f8 = all(f[k].dtype == np.float64 for k in ("m1", "m2", "snr", "merger_pos", "gps"))
    results.append(("waveform params stored float64 (bug #4's rule)", f8, "an f4 crept in"))
    bal = abs(y.mean() - 0.5)
    results.append(("labels ~50/50", bal < 0.01, f"positive fraction {y.mean():.3f}"))
    neg = y == 0
    clean = all((f[k][:][neg] == 0).all() for k in ("m1", "m2", "snr", "merger_pos"))
    results.append(("negatives carry no waveform metadata", clean, "nonzero params on y=0 rows"))
    pos = ~neg
    inr = ((snr[pos] >= SNR_RANGE[0]) & (snr[pos] <= SNR_RANGE[1])).all()
    mr = ((f["merger_pos"][:][pos] >= MERGER_RANGE[0]) & (f["merger_pos"][:][pos] < MERGER_RANGE[1])).all()
    results.append(("SNR and merger_pos in range", bool(inr and mr), "out-of-range draw"))
    uniq = len(np.unique(block.astype(np.int64) * 100_000 + offset)) == n
    results.append(("(block, offset) pairs unique — no crop stored twice", uniq, "duplicates!"))

    # --- the split is time itself ----------------------------------------------------
    spans = [(nm, gps[split == i]) for i, nm in enumerate(("train", "val", "test"))]
    nonempty = [(nm, g) for nm, g in spans if len(g)]
    ordered = all(a[1].max() < b[1].min() for a, b in zip(nonempty, nonempty[1:]))
    results.append(("splits time-ordered: train < val < test in GPS", bool(ordered),
                    " | ".join(f"{nm} {g.min():.0f}..{g.max():.0f}" for nm, g in nonempty)))
    for nm, g in nonempty:
        print(f"  {nm:5s} spans GPS {g.min():.0f} .. {g.max():.0f}")
    print(f"  test noise: {(y[split==2]==0).sum()/3600:.2f} h")

    # --- bit-exact rebuild -------------------------------------------------------------
    rows = rng.choice(n, size=min(N_REBUILD, n), replace=False)
    worst = 0.0
    for r in rows:
        r = int(r)
        spec = (r, int(y[r]), int(block[r]), int(offset[r]),
                float(f["m1"][r]), float(f["m2"][r]), float(snr[r]), float(f["merger_pos"][r]))
        _, x = build_segment(spec)
        worst = max(worst, float(np.max(np.abs(x - f["X"][r, 0]))))
    results.append((f"bit-exact rebuild of {len(rows)} rows (max|delta| = {worst:.3e})",
                    worst == 0.0, "a stored row is not what its metadata builds"))

    # --- SNR calibration: the empirical-background rho ---------------------------------
    # Group sampled positives by block so sigma_bg comes from the same noise regime.
    pos_rows = np.flatnonzero(pos)
    pick = rng.choice(pos_rows, size=min(N_SNR, len(pos_rows)), replace=False)
    neg_by_block: dict[int, np.ndarray] = {}
    ratios, zs = [], []
    for r in sorted(int(r) for r in pick):
        b = int(block[r])
        if b not in neg_by_block:
            cand = np.flatnonzero(neg & (block == b))
            neg_by_block[b] = rng.choice(cand, size=min(N_BG_PER_BLOCK, len(cand)), replace=False)
        bg_rows = neg_by_block[b]
        if len(bg_rows) < 30:
            continue
        # the signal actually in the row: conditioned, scaled template (linearity)
        spec = (r, 1, b, int(offset[r]), float(f["m1"][r]), float(f["m2"][r]),
                float(snr[r]), float(f["merger_pos"][r]))
        _, x_pos = build_segment(spec)
        spec_n = (r, 0, b, int(offset[r]), 0.0, 0.0, 0.0, 0.0)
        _, x_noise = build_segment(spec_n)
        ch = (x_pos.astype(np.float64) - x_noise.astype(np.float64))  # C(h), exactly

        bg = np.stack([f["X"][int(i), 0] for i in bg_rows]).astype(np.float64)
        sigma_bg = float(np.std(bg @ ch))
        rho = float(ch @ ch) / sigma_bg
        ratios.append(rho / snr[r])
        zs.append((float(f["X"][r, 0].astype(np.float64) @ ch) / sigma_bg - rho))
    ratios, zs = np.array(ratios), np.array(zs)
    med = float(np.median(ratios))
    lo, hi = np.percentile(ratios, [5, 95])
    results.append((f"SNR calibration: median rho/target = {med:.3f} (5-95% {lo:.3f}-{hi:.3f})",
                    0.9 < med < 1.1, "the money plot's x-axis is mislabelled"))
    zm, zsd = float(zs.mean()), float(zs.std())
    results.append((f"stored rows contain their signal: z = rho + N(0,1) -> {zm:+.3f} +- {zsd:.3f}",
                    abs(zm) < 0.3 and 0.8 < zsd < 1.25, "row/label divorce or bad sigma_bg"))

    # --- figure -------------------------------------------------------------------------
    fig, (axr, axz) = plt.subplots(1, 2, figsize=(13, 5))
    tgt = snr[sorted(int(r) for r in pick)][: len(ratios)]
    axr.plot([4, 20], [4, 20], "k--", lw=1, label="ideal")
    axr.scatter(tgt, tgt * ratios, s=10, alpha=0.5, color="tab:blue",
                label=f"recovered rho (median ratio {med:.3f})")
    axr.set_xlabel("requested SNR (vs the block's measured PSD)")
    axr.set_ylabel("recovered rho from the file")
    axr.set_title("SNR calibration on real O3 noise\n(sigma_bg from same-block negatives)")
    axr.legend()
    axr.grid(alpha=0.3)

    axz.hist(zs, bins=30, density=True, color="tab:orange", alpha=0.8,
             label=f"measured: {zm:+.2f} +- {zsd:.2f}")
    zg = np.linspace(-4, 4, 200)
    axz.plot(zg, np.exp(-zg**2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1) prediction")
    axz.set_xlabel("(C(h).x)/sigma_bg - rho")
    axz.set_ylabel("density")
    axz.set_title("Each stored row contains the signal its metadata claims\n"
                  "(unit variance proves sigma_bg is the right normalisation)")
    axz.legend()
    axz.grid(alpha=0.3)

    print()
    all_ok = True
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + ("" if ok else f" — {detail}"))
        all_ok &= ok

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "3_dataset_check.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    f.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

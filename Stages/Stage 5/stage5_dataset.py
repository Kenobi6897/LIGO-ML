"""Stage 5, Step 2 — rebuild both datasets under the three alternative conditioners.

    python stage5_dataset.py --source s2     # ~100k crops, Stage 2's geometry  -> stage5_s2.h5
    python stage5_dataset.py --source s3     # ~7.3k crops, Stage 3's geometry  -> stage5_s3.h5

THE SPECS ARE NOT REDRAWN — THEY ARE RE-DERIVED. Both builders call the ORIGINAL
`make_specs()` (stage2_dataset's, stage3_dataset's) with the ORIGINAL seeds, so every
row of stage5_s2.h5 is the same block, the same offset, the same masses, the same target
SNR, the same merger position and the same split as the corresponding row of stage2.h5.
The injection is even placed and scaled by the same code path (build_segment's body,
lifted here only because it must now emit three crops instead of one). stage5_dataset_check.py
asserts this bit-exactly against the files on disk, and refuses to pass at 1e-5 — Stage 1
bug #4's rule, pointed at the one thing this stage's fairness rests on.

So: same noise, same signals, same splits. Three conditioners, three X arrays, one pass —
the PSD, the waveform generation and the sigma() integral are the expensive parts and they
are shared, which is why building three arms costs barely more than building one.

    X_raw   (N, 1, 2048) f4    crop only
    X_bp    (N, 1, 2048) f4    bandpass 30-350 -> crop
    X_wh    (N, 1, 2048) f4    whiten -> crop
    + every metadata column of the source dataset, copied from the same specs

The `full` arm is NOT rebuilt: it is stage2.h5 / stage3.h5, already on disk, already the
thing Stages 2-4 reported. Rebuilding the control would risk moving it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 3"))
from stage1_condition import CROP_N
from stage1_dataset import place_in_buffer
from stage1_injection_check import make_waveform
from stage1_noise_check import BAND, SAMPLE_RATE

import stage2_condition as s2c
import stage2_dataset as s2d
import stage3_dataset as s3d
from stage2_fetch import DEFAULT_OUT as STRAIN_S2
from stage3_fetch import DEFAULT_OUT as STRAIN_S3

import stage5_condition as s5c
from stage5_condition import BUILD_ARMS

OUT_DIR = Path.home() / "ligo-data"
WRITE_BLOCK = 512

# (source dataset on disk, strain file, the metadata columns to carry over)
S2_COLS = ["y", "block", "offset", "gps", "m1", "m2", "snr", "merger_pos", "split"]
S3_COLS = S2_COLS + ["is_glitch", "glitch_label", "glitch_gps", "glitch_pos", "glitch_snr"]


def build_row(spec) -> tuple[int, np.ndarray]:
    """One row, three arms. stage2_dataset.build_segment's body, emitting a stack.

    The injection block is character-for-character Stage 2's: same waveform, same
    in-band sigma against the same block PSD, same amplitude. It happens ONCE, upstream
    of every conditioner, which is what makes 'SNR 8' mean the same physical signal in
    all three arms (and in the control on disk).
    """
    idx, label, block, offset, m1, m2, target_snr, merger_pos = spec

    buf = s2c.raw_buffer(int(block), int(offset))
    if label:
        placed = place_in_buffer(make_waveform(float(m1), float(m2)), float(merger_pos))
        s = s2d.in_band_sigma(placed, int(block))
        if not s > 0:
            raise RuntimeError(f"segment {idx}: no in-band power (m1={m1}, m2={m2})")
        buf = buf + placed * (float(target_snr) / s)

    return int(idx), np.stack([s5c.condition(buf, int(block), a) for a in BUILD_ARMS])


def _init(strain_path: Path) -> None:
    s5c.reset_caches(strain_path)


def specs_s2(seed: int):
    """Stage 2's specs, re-derived. Same seed -> same rows, and the check proves it."""
    rec = s2d.make_specs(seed, STRAIN_S2, None)
    tuples = s2d.spec_tuples(rec)
    cols = {
        "y": np.asarray(rec.label, "i1"), "block": np.asarray(rec.block, "i4"),
        "offset": np.asarray(rec.offset, "i4"), "gps": np.asarray(rec.gps, "f8"),
        "m1": np.asarray(rec.m1, "f8"), "m2": np.asarray(rec.m2, "f8"),
        "snr": np.asarray(rec.snr, "f8"), "merger_pos": np.asarray(rec.merger_pos, "f8"),
        "split": np.asarray(rec.split, "i1"),
    }
    return tuples, cols, STRAIN_S2, {}


def specs_s3(seed: int):
    """Stage 3's specs, re-derived — including its class-stratified split."""
    rec, split, labels, bgps = s3d.make_specs(seed)
    tuples = [
        (i, int(r.label), int(r.block), int(r.offset), float(r.m1), float(r.m2),
         float(r.snr), float(r.merger_pos))
        for i, r in enumerate(rec)
    ]
    cols = {
        "y": np.asarray(rec.label, "i1"), "block": np.asarray(rec.block, "i4"),
        "offset": np.asarray(rec.offset, "i4"),
        "gps": np.asarray(bgps[rec.block] + rec.offset, "f8"),
        "m1": np.asarray(rec.m1, "f8"), "m2": np.asarray(rec.m2, "f8"),
        "snr": np.asarray(rec.snr, "f8"), "merger_pos": np.asarray(rec.merger_pos, "f8"),
        "split": np.asarray(split, "i1"),
        "is_glitch": np.asarray(rec.is_glitch, "i1"),
        "glitch_label": np.asarray(rec.glitch_label, "i2"),
        "glitch_gps": np.asarray(rec.glitch_gps, "f8"),
        "glitch_pos": np.asarray(rec.glitch_pos, "f8"),
        "glitch_snr": np.asarray(rec.glitch_snr, "f8"),
    }
    return tuples, cols, STRAIN_S3, {"glitch_labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 2 — the three ablated datasets")
    ap.add_argument("--source", choices=("s2", "s3"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    args = ap.parse_args()

    seed = s2d.DATASET_SEED if args.source == "s2" else s3d.DATASET_SEED
    tuples, cols, strain, extra_attrs = (specs_s2 if args.source == "s2" else specs_s3)(seed)

    if args.limit:
        tuples = tuples[: args.limit]
        cols = {k: v[: args.limit] for k, v in cols.items()}
    n = len(tuples)
    out = args.out or OUT_DIR / f"stage5_{args.source}.h5"

    n_pos = int((cols["y"] == 1).sum())
    print(f"source {args.source}: {n:,} rows ({n_pos:,} + / {n - n_pos:,} -), "
          f"arms {BUILD_ARMS}, {args.workers} workers")
    print(f"  strain  {strain}")
    print(f"  splits  {(cols['split']==0).sum():,} / {(cols['split']==1).sum():,} / "
          f"{(cols['split']==2).sum():,}")
    print(f"  ->      {out}\n")

    # Point stage2_condition at the right strain file BEFORE any handle is opened, so the
    # workers inherit it across the fork (Stage 2's h5py-across-fork rule).
    s2c.STRAIN_H5 = strain
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")

    from tqdm import tqdm

    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        X = {
            a: f.create_dataset(f"X_{a}", (n, 1, CROP_N), dtype="f4", chunks=(1, 1, CROP_N))
            for a in BUILD_ARMS
        }
        buf = np.empty((WRITE_BLOCK, len(BUILD_ARMS), CROP_N), dtype=np.float32)
        fill = base = 0
        with Pool(args.workers, initializer=_init, initargs=(strain,)) as pool:
            for _idx, x in tqdm(pool.imap(build_row, tuples, chunksize=16),
                                total=n, unit="seg"):
                buf[fill] = x
                fill += 1
                if fill == WRITE_BLOCK:
                    for k, a in enumerate(BUILD_ARMS):
                        X[a][base : base + fill] = buf[:fill, k][:, None, :]
                    base += fill
                    fill = 0
        if fill:
            for k, a in enumerate(BUILD_ARMS):
                X[a][base : base + fill] = buf[:fill, k][:, None, :]

        for name, arr in cols.items():
            f.create_dataset(name, data=arr)

        # The Pool is gone; the parent may touch DSP now. One norm per arm, recorded.
        s5c.reset_caches(strain)
        f.attrs.update(
            source=args.source, arms=list(BUILD_ARMS), detector="H1", run="O3a",
            sample_rate=SAMPLE_RATE, band_low=BAND[0], band_high=BAND[1],
            strain_file=str(strain), dataset_seed=seed, n_total=n,
            control="stage2.h5" if args.source == "s2" else "stage3.h5",
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra_attrs,
        )
        for a in BUILD_ARMS:
            f.attrs[f"norm_{a}"] = s5c.norm(a)

    tmp.replace(out)
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min ({n/dt:.0f} seg/s) — {out.stat().st_size/1e6:.0f} MB")
    for a in BUILD_ARMS:
        print(f"  norm_{a:3s} = {s5c.norm(a):.6e}")
    print("\nnow run:  python stage5_dataset_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

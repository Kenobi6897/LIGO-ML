"""Stage 2, Step 3 — write the dataset: Stage 1's injections, real O3 noise.

~100k crops (509 per data block), 50/50. A positive is a negative with a waveform added;
both classes go through the identical `condition_real()` call. Output `~/ligo-data/stage2.h5`:

    X          (N, 1, 2048) float32   the CNN's input — conditioned real strain
    y          (N,)         int8      1 = injection + real noise, 0 = real noise
    snr        (N,)         float64   injected optimal SNR vs the BLOCK'S MEASURED PSD
    m1, m2     (N,)         float64   component masses, Msun (0.0 for negatives)
    merger_pos (N,)         float64   merger location as a fraction of the 1 s crop
    block      (N,)         int32     row in stage2_strain.h5 the crop came from
    offset     (N,)         int32     whole seconds into the block (0..508)
    gps        (N,)         float64   GPS of the crop's first sample
    split      (N,)         int8      0 = train, 1 = val, 2 = test — TIME-ORDERED

EVERYTHING STAGE 1 LEARNED THE HARD WAY STILL APPLIES
  - Waveform parameters stored float64, at the precision they were used — the bit-exact
    rebuild check depends on it (Stage 1 bug #4).
  - The merger is placed relative to the CROP, not the buffer (Stage 1 decision #1).
  - SNR is integrated over the crop, in 30-350 Hz — but against the BLOCK'S measured
    PSD, not the design curve. "SNR 8" must mean SNR 8 in the noise the segment actually
    sits in; the design curve would mislabel the money plot's x-axis by the time-varying
    ratio of real to design sensitivity.
  - The parent draws every spec up front and NEVER touches the strain file or an FFT
    before forking the Pool (the FFTW-fork deadlock, and the h5py-handle-across-fork
    rule, in one). Workers open their own handles and warm their own caches.

WHAT IS NEW: THE SPLIT IS TIME, NOT A SHUFFLE
Splits are assigned at BLOCK level, in time order: first 80% of data blocks -> train,
next 10% -> val, last 10% -> test. Val and test are strictly LATER than every training
sample, and no two blocks share a raw sample (stage2_fetch's geometry), so the split is
airtight. Noise drift across the boundary is not a leak — it is the phenomenon under
study, and eval reports it. Labels and injection parameters are drawn uniformly within
every block, so both classes see identical noise conditions.

Run (from WSL2, ~1 h on the 3700X):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 2"
    python stage2_dataset.py                                # all complete blocks
    python stage2_dataset.py --blocks 4 --out /tmp/s2sm.h5  # smoke test
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
from pycbc.filter import sigma
from pycbc.types import TimeSeries

from stage1_condition import CROP, CROP_N
from stage1_dataset import place_in_buffer
from stage1_injection_check import MASS_RANGE, make_waveform
from stage1_noise_check import BAND, SAMPLE_RATE

import stage2_condition as s2c
from stage2_fetch import BLOCK_LEN, DEFAULT_OUT as STRAIN_H5

# --- Stage 2 decisions, pinned in the plan note --------------------------------
SNR_RANGE = (4.0, 20.0)     # unchanged from Stage 1 — the noise is the only variable
MERGER_RANGE = (0.7, 0.95)  # unchanged
SPLITS = (0.8, 0.1, 0.1)    # of DATA BLOCKS, in time order
DATASET_SEED = 20260712

DEFAULT_OUT = Path.home() / "ligo-data" / "stage2.h5"
WRITE_BLOCK = 1024


def in_band_sigma(placed: TimeSeries, row: int) -> float:
    """Optimal SNR of the signal that survives to the CNN's input, in THIS block's noise.

    Same two restrictions as Stage 1 (the crop; the band) — the PSD is the block's
    measured one, because that is the noise the signal actually sits in.
    """
    snippet = TimeSeries(placed.numpy()[CROP].copy(), delta_t=1.0 / SAMPLE_RATE)
    return float(
        sigma(
            snippet,
            psd=s2c.snr_psd(row),
            low_frequency_cutoff=BAND[0],
            high_frequency_cutoff=BAND[1],
        )
    )


def build_segment(spec) -> tuple[int, np.ndarray]:
    """One row of X. Positives and negatives differ by exactly one added array."""
    idx, label, block, offset, m1, m2, target_snr, merger_pos = spec

    buf = s2c.raw_buffer(int(block), int(offset))

    if label:
        placed = place_in_buffer(make_waveform(float(m1), float(m2)), float(merger_pos))
        s = in_band_sigma(placed, int(block))
        if not s > 0:
            raise RuntimeError(f"segment {idx}: no in-band power (m1={m1}, m2={m2})")
        buf = buf + placed * (float(target_snr) / s)

    # The identical call for both classes — with the block's fixed, causal filter.
    return int(idx), s2c.condition_real(buf, int(block))


def make_specs(seed: int, strain_path: Path, n_blocks: int | None):
    """Draw every crop's parameters in the parent — WITHOUT opening anything that a
    forked worker will later use. The strain file is read in a scoped handle, closed
    before any Pool exists; no DSP happens here at all."""
    with h5py.File(strain_path, "r") as f:
        n_done = int(f.attrs.get("n_done", f["strain"].shape[0]))
        role = f["role"][:n_done]
        gps = f["gps"][:n_done]
    blocks = np.flatnonzero(role == 1)
    if n_blocks is not None:
        blocks = blocks[:n_blocks]
    if len(blocks) < 10 and n_blocks is None:
        raise SystemExit(f"only {len(blocks)} complete data blocks — fetch still running?")

    n_per = s2c.N_BUFFERS_PER_BLOCK  # 509
    n = len(blocks) * n_per

    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int8)
    labels[: n // 2] = 1
    rng.shuffle(labels)

    block_col = np.repeat(blocks.astype(np.int32), n_per)
    offset_col = np.tile(np.arange(n_per, dtype=np.int32), len(blocks))
    gps_col = gps[block_col] + offset_col  # buffer start; crop starts 1.5 s later

    m1 = rng.uniform(*MASS_RANGE, size=n)
    m2 = rng.uniform(*MASS_RANGE, size=n)
    snr = rng.uniform(*SNR_RANGE, size=n)
    merger = rng.uniform(*MERGER_RANGE, size=n)
    neg = labels == 0
    m1[neg] = m2[neg] = snr[neg] = merger[neg] = 0.0

    # Time-ordered split at block level: the first 80% of data blocks are train, and
    # every val/test sample is later than every train sample. Rows are already in time
    # order, so the split ranges are contiguous and trivially inspectable — same
    # property Stage 1's shuffled-then-contiguous split had, by different means.
    n_tr_b = int(SPLITS[0] * len(blocks))
    n_va_b = int(SPLITS[1] * len(blocks))
    split_of_block = np.full(len(blocks), 2, dtype=np.int8)
    split_of_block[:n_tr_b] = 0
    split_of_block[n_tr_b : n_tr_b + n_va_b] = 1
    split = np.repeat(split_of_block, n_per)

    return np.rec.fromarrays(
        [np.arange(n), labels, block_col, offset_col, gps_col, m1, m2, snr, merger, split],
        names="idx,label,block,offset,gps,m1,m2,snr,merger_pos,split",
    )


def spec_tuples(specs) -> list[tuple]:
    """The 8 fields build_segment needs. `split` stays out — a worker that could see
    the split could treat it differently (Stage 1's rule)."""
    return [
        (int(r.idx), int(r.label), int(r.block), int(r.offset),
         float(r.m1), float(r.m2), float(r.snr), float(r.merger_pos))
        for r in specs
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 3 — write the dataset")
    ap.add_argument("--blocks", type=int, default=None, help="cap data blocks (smoke test)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strain", type=Path, default=STRAIN_H5)
    ap.add_argument("--seed", type=int, default=DATASET_SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    specs = make_specs(args.seed, args.strain, args.blocks)
    n = len(specs)
    n_pos = int(specs.label.sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".h5.tmp")

    hours = n / 3600.0
    print(f"writing {n:,} crops ({n_pos:,} + / {n - n_pos:,} -) from "
          f"{len(np.unique(specs.block))} blocks ({hours:.1f} h of O3 H1) "
          f"with {args.workers} workers")
    print(f"  SNR U{SNR_RANGE} vs each block's measured PSD, masses U{MASS_RANGE} Msun")
    print(f"  splits (time-ordered): {(specs.split==0).sum():,} / "
          f"{(specs.split==1).sum():,} / {(specs.split==2).sum():,}")
    print(f"  -> {args.out}\n")

    from tqdm import tqdm

    if str(args.strain) != str(STRAIN_H5):
        s2c.STRAIN_H5 = args.strain  # workers inherit this by fork — before any open

    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        X = f.create_dataset("X", (n, 1, CROP_N), dtype="f4", chunks=(1, 1, CROP_N))

        buf = np.empty((WRITE_BLOCK, 1, CROP_N), dtype=np.float32)
        fill = 0
        base = 0
        with Pool(args.workers, initializer=_worker_init, initargs=(args.strain,)) as pool:
            for idx, x in tqdm(
                pool.imap(build_segment, spec_tuples(specs), chunksize=32), total=n, unit="seg"
            ):
                buf[fill, 0, :] = x
                fill += 1
                if fill == WRITE_BLOCK:
                    X[base : base + fill] = buf[:fill]
                    base += fill
                    fill = 0
        if fill:
            X[base : base + fill] = buf[:fill]

        # float64 for every waveform parameter — Stage 1 bug #4's rule.
        for name, dtype in [
            ("y", "i1"), ("block", "i4"), ("offset", "i4"), ("gps", "f8"),
            ("m1", "f8"), ("m2", "f8"), ("snr", "f8"), ("merger_pos", "f8"), ("split", "i1"),
        ]:
            src = specs.label if name == "y" else specs[name]
            f.create_dataset(name, data=np.asarray(src, dtype=dtype))

        # The Pool is gone; the parent may do DSP now. norm() here is the same float
        # every worker computed (deterministic), recorded so downstream consumers can
        # condition new data (e.g. an event check) identically without recomputing.
        f.attrs.update(
            detector="H1", run="O3a",
            sample_rate=SAMPLE_RATE, crop_len=1.0, block_len=BLOCK_LEN,
            band_low=BAND[0], band_high=BAND[1],
            snr_low=SNR_RANGE[0], snr_high=SNR_RANGE[1],
            mass_low=MASS_RANGE[0], mass_high=MASS_RANGE[1],
            merger_low=MERGER_RANGE[0], merger_high=MERGER_RANGE[1],
            approximant="IMRPhenomD",
            psd="per-block causal median Welch (previous 512 s block)",
            snr_definition="optimal SNR of the placed waveform, over the crop, "
                           "in 30-350 Hz, vs the block's measured PSD",
            norm=s2c.norm(),
            n_norm_blocks=s2c.N_NORM_BLOCKS,
            strain_file=str(args.strain),
            dataset_seed=args.seed, n_total=n,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    tmp.replace(args.out)
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min ({n/dt:.0f} seg/s) — "
          f"{args.out.stat().st_size/1e6:.0f} MB")
    print("\nnow run:  python stage2_dataset_check.py")
    return 0


def _worker_init(strain_path: Path) -> None:
    """Fork hygiene: whatever handle or cache state leaked over the fork, drop it.
    Each worker opens its own strain file and warms its own PSD/norm caches."""
    s2c.STRAIN_H5 = strain_path
    s2c._file = None
    s2c.block_table.cache_clear()
    s2c._measured_psd.cache_clear()
    s2c.whitening_psd.cache_clear()
    s2c.snr_psd.cache_clear()


if __name__ == "__main__":
    raise SystemExit(main())

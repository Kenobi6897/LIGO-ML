"""Stage 1, Step 4 — write the dataset.

100k segments, 50/50. A positive is a negative with a waveform added; both classes go
through the *identical* call. The output is `~/ligo-data/stage1.h5`:

    X          (N, 1, 2048) float32   the CNN's input — conditioned strain
    y          (N,)         int8      1 = signal + noise, 0 = noise
    snr        (N,)         float32   injected optimal SNR in band (0.0 for negatives)
    m1, m2     (N,)         float32   component masses, Msun (0.0 for negatives)
    merger_pos (N,)         float32   merger location as a fraction of the 1 s crop
    seed       (N,)         int64     the noise realisation — unique across the dataset
    split      (N,)         int8      0 = train, 1 = val, 2 = test

Everything except X is metadata you cannot recover later and will need at eval time.
`snr` above all: it is the x-axis of the money plot (efficiency vs. SNR), and there is no
way to reconstruct it from a conditioned segment after the fact.

WHAT ONE SEGMENT IS
    4 s of coloured Gaussian noise  ->  (if positive) inject a waveform scaled to a target
    SNR  ->  condition()  ->  keep the central 2048 samples.

The 4 s buffer is [[Stage 1#Step 3]]'s rule: whitening and band-passing are convolutions
and corrupt both ends, so we crop the central 1 s, 1.5 s clear of either edge.

TWO THINGS THIS FILE DECIDES THAT STEPS 1-3 DIDN'T HAVE TO
Both follow from the crop, and both would be silent errors:

1. THE MERGER IS PLACED RELATIVE TO THE CROP, NOT THE BUFFER. `merger_pos` in [0.7, 0.95)
   is a fraction of the 1 s window the CNN sees — so it maps to buffer sample
   CROP_START + merger_pos * CROP_N. Placing it at 0.7-0.95 of the *buffer* would put the
   merger at ~3 s, outside the crop, and every "positive" would be a segment whose signal
   had been cropped away. It would train, and it would learn nothing.

   The early inspiral is still injected into the buffer *outside* the crop, and is meant
   to be. A real 1 s segment cut from a detector stream has the earlier inspiral present
   in the data on either side of it, and the 0.5 s whitening filter reaches into it. We
   discard those samples; we do not pretend they were never there.

2. SNR IS INTEGRATED OVER 30-350 Hz — THE BAND THE CNN ACTUALLY SEES. `condition()`
   band-passes to BAND, so any signal power outside it is filtered away before the network
   gets the segment. Integrating `sigma` from 30 Hz with no upper cutoff — which is what
   [[Stage 1#Step 2]]'s check did — would count ringdown power above 350 Hz that the CNN
   never receives, and would overstate the SNR of exactly the lightest systems (whose
   merger frequency is highest). The money plot's x-axis has to mean "the SNR available in
   the data the model is handed", so we integrate over the same band we keep.

   `stage1_dataset_check.py` closes this loop empirically: it conditions the pure waveform
   (legal — `condition()` is signal-blind) and confirms its norm comes back as the SNR we
   asked for.

Reproducibility: the parent process draws every segment's parameters up front from one
seed, so the dataset is identical regardless of how many workers run. Noise seeds are
unique across all N — a repeated noise realisation split across train and test is a leak.

Run (from WSL2, ~10 min on the 3700X):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_dataset.py                 # the real thing: 100k
    python stage1_dataset.py --n 400 --out /tmp/smoke.h5    # smoke test
"""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from pycbc.filter import sigma
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from tqdm import tqdm

from stage1_condition import BUFFER_LEN, BUFFER_N, CROP, CROP_LEN, CROP_N, CROP_START, condition
from stage1_injection_check import MASS_RANGE, make_waveform, place
from stage1_noise_check import BAND, FLOW, SAMPLE_RATE, generate_noise

# --- Stage 1 decisions, pinned in the plan note -------------------------------
N_TOTAL = 100_000  # 50/50; ~820 MB of float32 at 2048 samples
SNR_RANGE = (4.0, 20.0)  # train across the range, evaluate per-SNR
MERGER_RANGE = (0.7, 0.95)  # fraction of the CROP — never a fixed index (footgun #2)
SPLITS = (0.8, 0.1, 0.1)  # train / val / test
DATASET_SEED = 20260712

DEFAULT_OUT = Path.home() / "ligo-data" / "stage1.h5"
WRITE_BLOCK = 1024  # rows buffered in RAM before flushing to HDF5


@lru_cache(maxsize=1)
def snr_psd():
    """Design PSD at the crop's frequency resolution — for sigma() and matched filters.

    delta_f = 1/CROP_LEN, because the SNR we quote is the SNR of the signal in the 1 s
    window the network sees, not in the 4 s buffer we threw most of away.
    """
    delta_f = 1.0 / CROP_LEN
    flen = int(SAMPLE_RATE / 2 / delta_f) + 1
    return aLIGOZeroDetHighPower(flen, delta_f, FLOW)


def place_in_buffer(hp, merger_pos: float) -> TimeSeries:
    """Place a waveform in the 4 s buffer with its merger inside the 1 s crop.

    `merger_pos` is a fraction of the CROP. Converting to a fraction of the BUFFER is the
    whole job — get it wrong and the signal lands outside the window the CNN is handed.
    """
    frac = (CROP_START + merger_pos * CROP_N) / BUFFER_N
    return place(hp, BUFFER_N, frac)


def in_band_sigma(placed: TimeSeries) -> float:
    """Optimal SNR of the part of the signal that survives to the CNN's input.

    Two restrictions, both deliberate, both explained in the module docstring:
      - the CROP: only the samples we keep;
      - BAND: only 30-350 Hz, because condition() filters the rest away.
    """
    snippet = TimeSeries(placed.numpy()[CROP].copy(), delta_t=1.0 / SAMPLE_RATE)
    return float(
        sigma(
            snippet,
            psd=snr_psd(),
            low_frequency_cutoff=BAND[0],
            high_frequency_cutoff=BAND[1],
        )
    )


def spec_tuples(specs) -> list[tuple]:
    """The 7 fields `build_segment` needs, as plain tuples.

    `split` is bookkeeping for the reader and deliberately not one of them: a worker that
    could see which split a segment belongs to is a worker that could treat it differently.
    """
    return [
        (int(r.idx), int(r.label), int(r.seed), float(r.m1), float(r.m2),
         float(r.snr), float(r.merger_pos))
        for r in specs
    ]


def build_segment(spec) -> tuple[int, np.ndarray]:
    """One row of X. Positives and negatives differ by exactly one added array."""
    idx, label, seed, m1, m2, target_snr, merger_pos = spec

    buf = generate_noise(BUFFER_LEN, seed=int(seed))

    if label:
        placed = place_in_buffer(make_waveform(float(m1), float(m2)), float(merger_pos))
        s = in_band_sigma(placed)
        if not s > 0:
            raise RuntimeError(f"segment {idx}: waveform has no in-band power (m1={m1}, m2={m2})")
        buf = buf + placed * (float(target_snr) / s)

    # The identical call for both classes. This is the whole ballgame — see Step 3.
    return int(idx), condition(buf)


def make_specs(n: int, seed: int) -> np.ndarray:
    """Draw every segment's parameters in the parent, so workers are pure functions.

    Consequence: the dataset depends only on (n, seed), not on the number of workers, the
    chunk size, or the order results come back in.
    """
    rng = np.random.default_rng(seed)

    labels = np.zeros(n, dtype=np.int8)
    labels[: n // 2] = 1
    rng.shuffle(labels)

    # Unique noise seeds. A repeated realisation that straddles the train/test boundary is
    # a leak — the model would meet the same noise twice and could memorise it.
    pool = np.unique(np.random.SeedSequence(seed).generate_state(int(n * 1.3) + 64))
    if len(pool) < n:
        raise RuntimeError(f"could not draw {n} unique noise seeds (got {len(pool)})")
    rng.shuffle(pool)
    seeds = pool[:n].astype(np.int64)

    m1 = rng.uniform(*MASS_RANGE, size=n)
    m2 = rng.uniform(*MASS_RANGE, size=n)
    snr = rng.uniform(*SNR_RANGE, size=n)
    merger = rng.uniform(*MERGER_RANGE, size=n)

    # Negatives carry no waveform, so they carry no waveform metadata. Zeros, not stale
    # draws — a nonzero snr on a y=0 row would be a trap for anything reading this file.
    neg = labels == 0
    m1[neg] = m2[neg] = snr[neg] = merger[neg] = 0.0

    # labels/seeds/params are already in random order, so a contiguous split is a random
    # split — and it makes the split ranges trivially inspectable.
    n_tr = int(SPLITS[0] * n)
    n_va = int(SPLITS[1] * n)
    split = np.full(n, 2, dtype=np.int8)
    split[:n_tr] = 0
    split[n_tr : n_tr + n_va] = 1

    return np.rec.fromarrays(
        [np.arange(n), labels, seeds, m1, m2, snr, merger, split],
        names="idx,label,seed,m1,m2,snr,merger_pos,split",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 Step 4 — write the dataset")
    ap.add_argument("--n", type=int, default=N_TOTAL, help="segments (half positive)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DATASET_SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    n = args.n
    if n % 2:
        raise SystemExit("--n must be even (the dataset is 50/50)")

    specs = make_specs(n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".h5.tmp")

    print(f"writing {n:,} segments ({n//2:,} + / {n//2:,} -) with {args.workers} workers")
    print(f"  {BUFFER_LEN:.0f}s buffer @ {SAMPLE_RATE} Hz -> condition() -> central {CROP_N} samples")
    print(f"  SNR U{SNR_RANGE} over {BAND[0]:.0f}-{BAND[1]:.0f} Hz, masses U{MASS_RANGE} Msun")
    print(f"  -> {args.out}\n")

    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        # chunks=(1,1,CROP_N): one segment per chunk, so the lazy __getitem__ in
        # stage1_data.py reads exactly one 8 KB chunk per random access. Uncompressed —
        # the file is ~820 MB, and gzip would tax every single batch read to save disk we
        # are not short of.
        X = f.create_dataset("X", (n, 1, CROP_N), dtype="f4", chunks=(1, 1, CROP_N))

        buf = np.empty((WRITE_BLOCK, 1, CROP_N), dtype=np.float32)
        fill = 0
        base = 0
        with Pool(args.workers) as pool:
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

        for name, dtype in [
            ("y", "i1"), ("seed", "i8"), ("m1", "f4"), ("m2", "f4"),
            ("snr", "f4"), ("merger_pos", "f4"), ("split", "i1"),
        ]:
            src = specs.label if name == "y" else specs[name]
            f.create_dataset(name, data=np.asarray(src, dtype=dtype))

        f.attrs.update(
            sample_rate=SAMPLE_RATE,
            buffer_len=BUFFER_LEN,
            crop_len=CROP_LEN,
            crop_start=CROP_START,
            band_low=BAND[0],
            band_high=BAND[1],
            f_low_generate=FLOW,
            snr_low=SNR_RANGE[0],
            snr_high=SNR_RANGE[1],
            mass_low=MASS_RANGE[0],
            mass_high=MASS_RANGE[1],
            merger_low=MERGER_RANGE[0],
            merger_high=MERGER_RANGE[1],
            approximant="IMRPhenomD",
            psd="aLIGOZeroDetHighPower",
            snr_definition="optimal SNR of the placed waveform, over the crop, in 30-350 Hz",
            dataset_seed=args.seed,
            n_total=n,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    tmp.replace(args.out)  # atomic: a half-written dataset never has the real name
    dt = time.time() - t0
    size_mb = args.out.stat().st_size / 1e6
    print(f"\ndone in {dt/60:.1f} min ({n/dt:.0f} seg/s) — {size_mb:.0f} MB")
    print(f"  train/val/test: {(specs.split==0).sum():,} / {(specs.split==1).sum():,} / {(specs.split==2).sum():,}")
    print("\nnow run:  python stage1_dataset_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

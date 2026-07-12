"""Stage 3, Step 3 — the hard-negative dataset: glitches by name, chirps beside them.

For every data block (a block holding >= 1 selected glitch), FOUR kinds of row, all
through the identical `condition_real()` call, all with the block's causal PSD:

    glitch negatives   y=0  crop placed so the glitch peak sits at U[0.10, 0.95) of it
    plain  negatives   y=0  clean crops (no Gravity Spy trigger of ANY confidence inside)
    injected positives y=1  Stage-2 injections into OTHER clean crops of the same block

Per block with G usable glitches: G glitch rows + G plain rows + 2G injected rows -> 4G
rows, exactly 50/50. THE ANTI-SHORTCUT RULE ([[Stage 3]] plan): both classes come from
every block, so "this stretch's texture" cannot become the label. Glitch placement
U[0.10, 0.95) deliberately OVERLAPS the injection merger range U[0.70, 0.95) — if
glitches only ever appeared where mergers never do, position would become the label.

CLEAN means clean by the whole CSV: crops are rejected if any Gravity Spy trigger (any
confidence, any class, including the ones we didn't select) falls within CLEAN_MARGIN of
them. Storm blocks may not have 3G clean crops; the block then contributes fewer rows,
trimmed to keep pos == neg. Catalog events cannot appear in data blocks (selection vetoed
glitches to EVENT_VETO + BLOCK_LEN, which keeps whole data blocks >= 128 s clear).

Schema = Stage 2's + the glitch columns (rebuild rule unchanged: parameters at the
precision used, float64; a glitch row rebuilds as a pure conditioned crop — label 0):

    is_glitch     i1    1 = glitch-centred negative
    glitch_label  i2    index into attrs['glitch_labels'] (-1 for non-glitch rows)
    glitch_gps    f8    Gravity Spy peak_time (0.0)
    glitch_pos    f8    where the peak sits in the crop, as a fraction (0.0)
    glitch_snr    f8    Gravity Spy's Omicron SNR (0.0)

Run (from WSL2, ~10 min):
    python stage3_dataset.py
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
from stage1_condition import CROP_N
from stage1_injection_check import MASS_RANGE
from stage1_noise_check import BAND, SAMPLE_RATE

import stage2_condition as s2c
from stage2_dataset import MERGER_RANGE, SNR_RANGE, _worker_init, build_segment, spec_tuples
from stage2_fetch import BLOCK_LEN, PAD

from stage3_fetch import DEFAULT_OUT as STRAIN_H5, plan_blocks
from stage3_select import CSV, DEFAULT_OUT as SELECTION_H5

DEFAULT_OUT = Path.home() / "ligo-data" / "stage3.h5"
SPLITS = (0.6, 0.1, 0.3)  # test-heavy: the per-class FA table lives on test statistics
DATASET_SEED = 20260713
GLITCH_POS_RANGE = (0.10, 0.95)
# +-1 s: a trigger a second OUTSIDE the crop is the reality every real search injects
# into. The first build used +-3 s and starved storm blocks of clean crops so badly the
# balancing trimmed 74% of all glitch rows (2,991 -> 780).
CLEAN_MARGIN = 1.0
WRITE_BLOCK = 512


def glitch_offset(p: float, rng) -> tuple[int, float] | None:
    """Integer buffer offset placing peak-position p (s into the block) at a crop
    fraction inside GLITCH_POS_RANGE. Returns (offset, fraction) or None.

    The preferred range's width (0.85 s) misses an integer offset for ~15% of peak
    positions; those fall back to a slightly wider window rather than being dropped.
    """
    for lo, hi in (GLITCH_POS_RANGE, (0.02, 0.99)):
        cands = [o for o in range(int(np.floor(p - 1.5 - hi)), int(np.floor(p - 1.5 - lo)) + 1)
                 if 0 <= o <= int(BLOCK_LEN - 4) and lo <= p - 1.5 - o < hi]
        if cands:
            o = int(rng.choice(cands))
            return o, p - 1.5 - o
    return None


def make_specs(seed: int):
    """Everything the workers need, drawn in the parent. No strain access, no DSP."""
    rng = np.random.default_rng(seed)

    blocks = plan_blocks(SELECTION_H5)  # (gps, span, role) in stage3_strain row order
    bgps = np.array([b[0] for b in blocks])
    brole = np.array([b[2] for b in blocks])

    with h5py.File(SELECTION_H5, "r") as f:
        g_gps = f["gps"][:]
        g_label = f["label_id"][:]
        g_snr = f["snr"][:]
        labels = list(f.attrs["labels"])

    import pandas as pd

    csv = pd.read_csv(CSV, usecols=["peak_time", "peak_time_ns"])
    all_trig = np.sort(csv.peak_time.values + csv.peak_time_ns.values * 1e-9)

    rows = []  # (label, block_row, offset, m1, m2, snr, merger_pos, is_g, g_lab, g_gps, g_pos, g_snr)
    data_rows = np.flatnonzero(brole == 1)
    for r in data_rows:
        r = int(r)
        b0 = bgps[r]
        in_blk = np.flatnonzero((g_gps >= b0 + 2.0) & (g_gps <= b0 + BLOCK_LEN - 2.6))

        used = set()
        glitch_rows = []
        for gi in in_blk:
            place = glitch_offset(float(g_gps[gi] - b0), rng)
            if place is None or place[0] in used:
                continue
            used.add(place[0])
            glitch_rows.append((0, r, place[0], 0.0, 0.0, 0.0, 0.0,
                                1, int(g_label[gi]), float(g_gps[gi]), place[1], float(g_snr[gi])))
        G = len(glitch_rows)
        if G == 0:
            continue

        # clean offsets: crop [o+1.5, o+2.5] with CLEAN_MARGIN, no CSV trigger inside
        lo = np.searchsorted(all_trig, b0 - 8)
        hi = np.searchsorted(all_trig, b0 + BLOCK_LEN + 8)
        trig = all_trig[lo:hi] - b0
        clean = [o for o in range(0, int(BLOCK_LEN - 4))
                 if o not in used
                 and not np.any((trig > o + 1.5 - CLEAN_MARGIN) & (trig < o + 2.5 + CLEAN_MARGIN))]
        rng.shuffle(clean)
        # balance within the block: negatives = G glitches + P plain, positives = G + P
        # injected, needing G + 2P clean offsets. Storm blocks short on clean crops trim
        # the glitch rows first (their storms are over-represented anyway).
        if len(clean) < G:
            glitch_rows = glitch_rows[: len(clean)]
            G = len(glitch_rows)
            if G == 0:
                continue
        P = min(G, max(0, (len(clean) - G) // 2))
        pos_offsets = clean[: G + P]
        plain_offsets = clean[G + P : G + 2 * P]

        for o in pos_offsets:
            rows.append((1, r, o, rng.uniform(*MASS_RANGE), rng.uniform(*MASS_RANGE),
                         rng.uniform(*SNR_RANGE), rng.uniform(*MERGER_RANGE),
                         0, -1, 0.0, 0.0, 0.0))
        for o in plain_offsets:
            rows.append((0, r, o, 0.0, 0.0, 0.0, 0.0, 0, -1, 0.0, 0.0, 0.0))
        rows.extend(glitch_rows)

    names = "label,block,offset,m1,m2,snr,merger_pos,is_glitch,glitch_label,glitch_gps,glitch_pos,glitch_snr"
    rec = np.rec.fromrecords(rows, names=names)

    # CLASS-STRATIFIED block splits — deliberately NOT time-ordered, and said out loud:
    # glitch classes cluster in time (storms), so a time-ordered split gave one class 0
    # training rows and 84 test rows, another 139/0/1. The glitch benchmark is a
    # population census, not a stream test — Stage 2's time-ordered split remains the
    # deployment guard. Block-level disjointness (no shared raw samples) still holds.
    # Rarest classes assign their blocks first; a block's assignment is final, so
    # multi-class blocks inherit the rarest class's stratification.
    rng2 = np.random.default_rng(seed + 1)
    g = rec[rec.is_glitch == 1]
    assign: dict[int, int] = {}
    class_counts = {li: int((g.glitch_label == li).sum()) for li in set(g.glitch_label)}
    for li in sorted(class_counts, key=class_counts.get):
        blks = [int(b) for b in np.unique(g.block[g.glitch_label == li]) if int(b) not in assign]
        if not blks:
            continue
        rng2.shuffle(blks)
        n = len(blks)
        if n == 1:
            assign[blks[0]] = 2  # unmeasurable is worse than untrainable
            continue
        n_te = max(1, round(SPLITS[2] * n))
        n_va = round(SPLITS[1] * n)
        for i, b in enumerate(blks):
            assign[b] = 2 if i < n_te else (1 if i < n_te + n_va else 0)
    split = np.array([assign[int(b)] for b in rec.block], dtype=np.int8)
    return rec, split, labels, bgps


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Step 3 — write the hard-negative dataset")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DATASET_SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    rec, split, labels, bgps = make_specs(args.seed)
    n = len(rec)
    n_pos = int((rec.label == 1).sum())
    n_glitch = int(rec.is_glitch.sum())
    print(f"{n:,} rows: {n_pos:,} injected + / {n - n_pos:,} - "
          f"({n_glitch:,} glitch, {n - n_pos - n_glitch:,} plain)")
    print(f"  splits: {(split==0).sum():,} / {(split==1).sum():,} / {(split==2).sum():,}")
    if abs(n_pos / n - 0.5) > 0.02:
        raise SystemExit(f"class balance broke: {n_pos/n:.3f} positive")

    specs = [(i, int(r.label), int(r.block), int(r.offset),
              float(r.m1), float(r.m2), float(r.snr), float(r.merger_pos))
             for i, r in enumerate(rec)]

    s2c.STRAIN_H5 = STRAIN_H5  # workers inherit by fork, before any open
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".h5.tmp")

    from tqdm import tqdm

    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        X = f.create_dataset("X", (n, 1, CROP_N), dtype="f4", chunks=(1, 1, CROP_N))
        buf = np.empty((WRITE_BLOCK, 1, CROP_N), dtype=np.float32)
        fill = base = 0
        with Pool(args.workers, initializer=_worker_init, initargs=(STRAIN_H5,)) as pool:
            for idx, x in tqdm(pool.imap(build_segment, specs, chunksize=16),
                               total=n, unit="seg"):
                buf[fill, 0, :] = x
                fill += 1
                if fill == WRITE_BLOCK:
                    X[base : base + fill] = buf[:fill]
                    base += fill
                    fill = 0
        if fill:
            X[base : base + fill] = buf[:fill]

        f.create_dataset("y", data=np.asarray(rec.label, dtype="i1"))
        for name, dt in [("block", "i4"), ("offset", "i4"), ("m1", "f8"), ("m2", "f8"),
                         ("snr", "f8"), ("merger_pos", "f8"), ("is_glitch", "i1"),
                         ("glitch_label", "i2"), ("glitch_gps", "f8"),
                         ("glitch_pos", "f8"), ("glitch_snr", "f8")]:
            f.create_dataset(name, data=np.asarray(rec[name], dtype=dt))
        f.create_dataset("gps", data=bgps[rec.block] + rec.offset)
        f.create_dataset("split", data=split)
        f.attrs.update(
            detector="H1", run="O3a", sample_rate=SAMPLE_RATE,
            band_low=BAND[0], band_high=BAND[1],
            snr_low=SNR_RANGE[0], snr_high=SNR_RANGE[1],
            glitch_labels=labels, glitch_pos_low=GLITCH_POS_RANGE[0],
            glitch_pos_high=GLITCH_POS_RANGE[1], clean_margin=CLEAN_MARGIN,
            strain_file=str(STRAIN_H5), selection_file=str(SELECTION_H5),
            dataset_seed=args.seed, n_total=n,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        f.attrs["norm"] = s2c.norm()  # Pool is closed; the parent may do DSP now

    tmp.replace(args.out)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min — {args.out.stat().st_size/1e6:.0f} MB")
    print("\nnow run:  python stage3_dataset_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

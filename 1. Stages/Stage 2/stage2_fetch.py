"""Stage 2, Step 1 — fetch real O3 noise. The one variable this stage changes.

Downloads H1 strain from GWOSC (O3a, in time order from the run's start), keeps only
science-mode data with every catalog event vetoed, resamples 4096 -> 2048 Hz, and stores
it as fixed-length BLOCKS in `~/ligo-data/stage2_strain.h5`:

    strain      (n_blocks, 1048576) float64   512 s of 2048 Hz strain per block
    gps         (n_blocks,)         float64   GPS of each block's first sample
    stretch_id  (n_blocks,)         int32     which contiguous science stretch it came from
    role        (n_blocks,)         int8      0 = PSD source, 1 = data

WHY BLOCKS, AND WHY BLOCK 0 OF EVERY STRETCH IS PSD-ONLY
Real noise is non-stationary: the spectrum drifts over hours. Stage 2's conditioning
whitens block k with a Welch PSD measured on block k-1 — causal, and never the block
itself, so the whitener is signal-blind by construction (injections are added after PSD
estimation, downstream of this file entirely). The first block of every stretch has no
predecessor, so it serves only as a PSD source. Blocks never span a stretch boundary and
buffers never span a block boundary, so no two blocks share a single raw sample — which
is what makes a block-level train/val/test split airtight at the sample level.

WHY THE EVENT VETO IS NOT OPTIONAL
O3a contains real signals. A catalog event inside a "noise" block teaches the model that
chirps are noise — a label error aimed exactly at the thing being detected. ±128 s around
every GWOSC event dataset in the span is cut before blocks are laid down.

WHY RESAMPLING HAPPENS ON PADDED FETCHES
`resample` is a FIR filter: it corrupts the ends of whatever it is given. Every fetch
carries PAD s of extra strain on each side, trimmed after resampling, so no filter
transient ever lands inside a stored block. Blocks are laid from stretch_start + PAD so
the padding always exists inside science-mode data.

Storage is float64: the conditioning pipeline stays in doubles end-to-end (Stage 1's
bit-exact-rebuild lesson — store what you compute with). ~1.9 GB on ext4, never in git.

Run (from WSL2; ~30-60 min, network-bound, safe to re-run — skips finished work):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 2"
    python stage2_fetch.py                     # the real thing (~197 data blocks)
    python stage2_fetch.py --blocks 4          # smoke test: 4 data blocks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_noise_check import SAMPLE_RATE  # 2048 Hz — the one Stage 1 decision reused here

# --- Stage 2 decisions, pinned in the plan note --------------------------------
O3A_START = 1238166018  # 2019-04-01, the first second of O3a
SEARCH_SPAN = 14 * 86400  # widened automatically if it ever comes up short
BLOCK_LEN = 512.0  # s — the PSD/non-stationarity granularity
BLOCK_N = int(BLOCK_LEN * SAMPLE_RATE)  # 1,048,576 samples
EVENT_VETO = 128.0  # s cut on each side of every catalog event
PAD = 8.0  # s of extra strain fetched around every group, trimmed after resample
MIN_STRETCH = 2 * BLOCK_LEN + 2 * PAD  # a stretch must hold >= 1 PSD block + 1 data block
FETCH_RATE = 4096  # Hz — what GWOSC serves; resampled down to SAMPLE_RATE
GROUP_BLOCKS = 8  # blocks fetched per network call (4096 s + padding, ~140 MB in RAM)

# 509 crops per data block (4 s buffers hopping 1 s inside 512 s) -> ~100k crops
CROPS_PER_BLOCK = int(BLOCK_LEN) - 3
N_CROPS_TARGET = 100_000

DEFAULT_OUT = Path.home() / "ligo-data" / "stage2_strain.h5"


def science_stretches(span_end: float) -> tuple[list[tuple[float, float]], list[tuple[str, float]]]:
    """H1 science segments from O3a start, with catalog events vetoed out.

    Returns (stretches, events): event-free science spans in time order, and the
    (name, gps) list that was vetoed — stored in the file so the check can re-verify.
    """
    from gwosc.datasets import event_gps, find_datasets
    from gwosc.timeline import get_segments

    window = (O3A_START, int(span_end))
    segs = get_segments("H1_DATA", *window)

    names = find_datasets(type="events", segment=window)
    events: dict[str, float] = {}
    for n in names:
        base = n.split("-")[0]  # 'GW190403_051519-v2' and '-v1' are one event
        if base not in events:
            events[base] = float(event_gps(n))

    vetoes = [(g - EVENT_VETO, g + EVENT_VETO) for g in events.values()]
    out: list[tuple[float, float]] = []
    for a, b in segs:
        pieces = [(float(a), float(b))]
        for va, vb in vetoes:
            nxt = []
            for pa, pb in pieces:
                if vb <= pa or va >= pb:  # veto misses this piece
                    nxt.append((pa, pb))
                    continue
                if pa < va:
                    nxt.append((pa, va))
                if vb < pb:
                    nxt.append((vb, pb))
            pieces = nxt
        out.extend(p for p in pieces if p[1] - p[0] >= MIN_STRETCH)
    return out, sorted(events.items(), key=lambda kv: kv[1])


def plan_blocks(stretches, n_data_target: int):
    """Lay blocks onto stretches: (gps, stretch_id, role) rows, in time order.

    Blocks start at stretch_start + PAD (so every fetch can pad inside science data) and
    the first block of each stretch is role 0: a PSD source, never data.
    """
    rows = []
    n_data = 0
    for sid, (a, b) in enumerate(stretches):
        n_here = int((b - a - 2 * PAD) // BLOCK_LEN)
        for k in range(n_here):
            if n_data >= n_data_target:
                return rows
            role = 0 if k == 0 else 1
            rows.append((a + PAD + k * BLOCK_LEN, sid, role))
            n_data += role
    return rows


def fetch_group(gps0: float, n_blocks: int) -> np.ndarray:
    """One network call: n_blocks contiguous blocks + PAD each side, resampled, trimmed.

    Returns (n_blocks, BLOCK_N) float64. The resample transient lives entirely in the
    padding, which is trimmed here and never stored.
    """
    from gwpy.timeseries import TimeSeries as GwpyTS

    a, b = gps0 - PAD, gps0 + n_blocks * BLOCK_LEN + PAD
    ts = GwpyTS.fetch_open_data("H1", a, b, sample_rate=FETCH_RATE, cache=True)
    if np.isnan(ts.value).any():
        raise RuntimeError(f"NaNs in fetched strain at {a}-{b} — science segment lied?")
    ts = ts.resample(SAMPLE_RATE)
    x = ts.value.astype(np.float64)
    i0 = int(round((gps0 - float(ts.t0.value)) * SAMPLE_RATE))
    need = n_blocks * BLOCK_N
    if i0 < 0 or i0 + need > len(x):
        raise RuntimeError(f"fetched stretch {a}-{b} too short after trim")
    return x[i0 : i0 + need].reshape(n_blocks, BLOCK_N)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 1 — fetch real O3 noise")
    ap.add_argument("--blocks", type=int,
                    default=int(np.ceil(N_CROPS_TARGET / CROPS_PER_BLOCK)),
                    help="data blocks to fetch (default sized for ~100k crops)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"planning: {args.blocks} data blocks of {BLOCK_LEN:.0f}s "
          f"({args.blocks * CROPS_PER_BLOCK:,} crops), H1 O3a from GPS {O3A_START}")

    span = SEARCH_SPAN
    while True:
        stretches, events = science_stretches(O3A_START + span)
        rows = plan_blocks(stretches, args.blocks)
        if sum(r[2] for r in rows) >= args.blocks:
            break
        span *= 2
        print(f"  not enough science time in {span // (2*86400)} d — widening to {span // 86400} d")

    n_total = len(rows)
    gps = np.array([r[0] for r in rows])
    sid = np.array([r[1] for r in rows], dtype=np.int32)
    role = np.array([r[2] for r in rows], dtype=np.int8)
    print(f"  planned {n_total} blocks ({int(role.sum())} data + {int((role == 0).sum())} PSD-only) "
          f"across {sid.max() + 1} stretches, ending GPS {gps[-1] + BLOCK_LEN:.0f}")
    print(f"  vetoed {len(events)} events: {', '.join(n for n, _ in events)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resumable: rows are written in order, n_done says how many are already on disk.
    if args.out.exists():
        with h5py.File(args.out, "r") as f:
            same_plan = (f["gps"].shape[0] == n_total
                         and np.allclose(f["gps"][:], gps))
            n_done = int(f.attrs.get("n_done", 0)) if same_plan else 0
        if not same_plan:
            print("  existing file has a different plan — starting over")
            args.out.unlink()
    else:
        n_done = 0

    if n_done == 0 or not args.out.exists():
        with h5py.File(args.out, "w") as f:
            f.create_dataset("strain", (n_total, BLOCK_N), dtype="f8",
                             chunks=(1, BLOCK_N))
            f.create_dataset("gps", data=gps)
            f.create_dataset("stretch_id", data=sid)
            f.create_dataset("role", data=role)
            f.attrs.update(
                detector="H1", run="O3a", sample_rate=SAMPLE_RATE,
                block_len=BLOCK_LEN, pad=PAD, event_veto=EVENT_VETO,
                o3a_start=O3A_START, n_done=0,
                event_names=[n for n, _ in events],
                event_gps=[g for _, g in events],
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    from tqdm import tqdm

    t0 = time.time()
    with h5py.File(args.out, "a") as f:
        pbar = tqdm(total=n_total, initial=n_done, unit="block")
        i = n_done
        while i < n_total:
            # group: consecutive planned blocks that are contiguous in GPS (same stretch)
            j = i + 1
            while (j < n_total and j - i < GROUP_BLOCKS and sid[j] == sid[i]
                   and np.isclose(gps[j], gps[i] + (j - i) * BLOCK_LEN)):
                j += 1
            f["strain"][i:j] = fetch_group(gps[i], j - i)
            f.attrs["n_done"] = j
            f.flush()
            pbar.update(j - i)
            i = j
        pbar.close()

    dt = time.time() - t0
    size_gb = args.out.stat().st_size / 1e9
    print(f"\ndone in {dt/60:.1f} min — {size_gb:.2f} GB at {args.out}")
    print("\nnow run:  python stage2_fetch_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

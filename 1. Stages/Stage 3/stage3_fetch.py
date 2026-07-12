"""Stage 3, Step 2 — fetch the strain the selection planned. Stage 2's machinery, reused.

Reads `stage3_selection.h5` (spans + glitches) and writes `stage3_strain.h5` with exactly
the schema of `stage2_strain.h5` — same 512 s blocks, same padded fetches, same resumable
layout — so `stage2_condition` can be pointed at it UNCHANGED and every leak property
proven there carries over.

ROLES: a block is data (role 1) iff it holds at least one selected glitch's buffer;
otherwise it is a PSD source (role 0). The selection's grid closure guarantees every
data block's immediate predecessor is in the same span, and that the first block of
every span is glitch-free — so "every stretch leads with a PSD block" holds here too.

EVENTS: selected glitches sit >= EVENT_VETO + BLOCK_LEN from every catalog event, which
puts every point of every DATA block >= 128 s from any event. A role-0 PSD block MAY
contain an event — harmless to a median Welch, and role-0 blocks never become crops.

Run (from WSL2, ~2 h for ~100 GWOSC files; detached + resumable, safe to re-run):
    python stage3_fetch.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage2_fetch import BLOCK_LEN, BLOCK_N, GROUP_BLOCKS, PAD, fetch_group

from stage3_select import DEFAULT_OUT as SELECTION_H5

DEFAULT_OUT = Path.home() / "ligo-data" / "stage3_strain.h5"


def plan_blocks(sel_path: Path):
    """(gps, stretch_id, role) for every block of every span, plus the glitch table."""
    with h5py.File(sel_path, "r") as f:
        spans = np.stack([f["span_start"][:], f["span_end"][:]], axis=1)
        glitch_gps = f["gps"][:]

    rows = []
    for sid, (a, b) in enumerate(spans):
        n = int(round((b - a - 2 * PAD) / BLOCK_LEN))
        for k in range(n):
            g0 = a + PAD + k * BLOCK_LEN
            has = bool(np.any((glitch_gps >= g0 + 2.0) & (glitch_gps <= g0 + BLOCK_LEN - 2.6)))
            rows.append((g0, sid, 1 if has else 0))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Step 2 — fetch strain for the selection")
    ap.add_argument("--selection", type=Path, default=SELECTION_H5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = plan_blocks(args.selection)
    n_total = len(rows)
    gps = np.array([r[0] for r in rows])
    sid = np.array([r[1] for r in rows], dtype=np.int32)
    role = np.array([r[2] for r in rows], dtype=np.int8)
    print(f"planned {n_total} blocks ({int(role.sum())} data + {int((role==0).sum())} PSD-only) "
          f"across {sid.max()+1} spans — {n_total*BLOCK_LEN/3600:.1f} h")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        with h5py.File(args.out, "r") as f:
            same = f["gps"].shape[0] == n_total and np.allclose(f["gps"][:], gps)
            n_done = int(f.attrs.get("n_done", 0)) if same else 0
        if not same:
            print("  existing file has a different plan — starting over")
            args.out.unlink()
    else:
        n_done = 0

    if n_done == 0 or not args.out.exists():
        with h5py.File(args.out, "w") as f:
            f.create_dataset("strain", (n_total, BLOCK_N), dtype="f8", chunks=(1, BLOCK_N))
            f.create_dataset("gps", data=gps)
            f.create_dataset("stretch_id", data=sid)
            f.create_dataset("role", data=role)
            f.attrs.update(
                detector="H1", run="O3a", sample_rate=2048,
                block_len=BLOCK_LEN, pad=PAD, n_done=0,
                selection=str(args.selection),
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    from tqdm import tqdm

    t0 = time.time()
    with h5py.File(args.out, "a") as f:
        pbar = tqdm(total=n_total, initial=n_done, unit="block")
        i = n_done
        while i < n_total:
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

    print(f"\ndone in {(time.time()-t0)/60:.1f} min — "
          f"{args.out.stat().st_size/1e9:.2f} GB at {args.out}")
    print("\nnow run:  python stage3_fetch_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

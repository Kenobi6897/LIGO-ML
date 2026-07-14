"""Stage 3, Step 2's check — is the fetched strain what the selection ordered?

Beyond Stage 2's integrity checks (finite, disjoint, PSD-block-led), the Stage 3 file
carries a promise the dataset step will silently depend on: EVERY selected glitch lands
in a role-1 block with a contiguous same-stretch predecessor, and role-1 blocks stay
>= 128 s clear of every catalog event (role-0 PSD blocks are allowed to contain events —
they never become crops, and a median Welch shrugs them off).

Run (from WSL2, after stage3_fetch.py):
    python stage3_fetch_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage2_fetch import BLOCK_LEN, EVENT_VETO

from stage3_fetch import DEFAULT_OUT
from stage3_select import DEFAULT_OUT as SELECTION_H5, O3A


def main() -> int:
    results = []
    with h5py.File(DEFAULT_OUT, "r") as f:
        n_total = f["strain"].shape[0]
        n_done = int(f.attrs.get("n_done", 0))
        if n_done < n_total:
            print(f"file is PARTIAL: {n_done}/{n_total} — checking what exists")
        gps = f["gps"][:n_done]
        sid = f["stretch_id"][:n_done]
        role = f["role"][:n_done]

        bad = zero = 0
        for i in range(n_done):
            x = f["strain"][i]
            if not np.isfinite(x).all():
                bad += 1
            if (x == 0.0).mean() > 0.001:
                zero += 1
    results.append(("all strain finite", bad == 0, f"{bad} bad blocks"))
    results.append(("no zero-filled gaps", zero == 0, f"{zero} suspicious blocks"))

    order = np.argsort(gps)
    results.append(("blocks disjoint in time",
                    bool((np.diff(gps[order]) >= BLOCK_LEN - 1e-9).all()), "overlap"))
    lead_ok = all(role[np.flatnonzero(sid == s)[0]] == 0 for s in np.unique(sid))
    results.append(("every span leads with a PSD block", lead_ok, "a span starts with data"))

    with h5py.File(SELECTION_H5, "r") as f:
        ggps = f["gps"][:]

    # every glitch in a role-1 block with a contiguous predecessor
    ok_g = 0
    for g in ggps:
        i = np.searchsorted(gps, g) - 1
        if (0 < i < n_done and role[i] == 1 and gps[i] <= g < gps[i] + BLOCK_LEN
                and sid[i - 1] == sid[i] and np.isclose(gps[i - 1] + BLOCK_LEN, gps[i])):
            ok_g += 1
    results.append((f"all {len(ggps)} glitches in role-1 blocks with causal predecessors",
                    ok_g == len(ggps), f"only {ok_g}"))

    # role-1 blocks clear of events; every block inside science segments
    from gwosc.datasets import event_gps, find_datasets
    from gwosc.timeline import get_segments

    ev = np.unique([float(event_gps(n)) for n in find_datasets(type="events", segment=O3A)])
    data_gps = gps[role == 1]
    dmin = np.min(np.abs(data_gps[:, None] + BLOCK_LEN / 2 - ev[None, :]), axis=1) - BLOCK_LEN / 2
    results.append((f"every data block >= {EVENT_VETO:.0f}s from all {len(ev)} events",
                    bool((dmin >= EVENT_VETO).all()), f"closest {dmin.min():.0f}s"))

    segs = np.array(get_segments("H1_DATA", *O3A), dtype=float)
    i = np.clip(np.searchsorted(segs[:, 1], gps), 0, len(segs) - 1)
    in_sci = (segs[i, 0] <= gps) & (gps + BLOCK_LEN <= segs[i, 1])
    results.append(("every block inside a science segment", bool(in_sci.all()),
                    f"{int((~in_sci).sum())} outside"))

    print()
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {name}" + ("" if passed else f" — {detail}"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 3, Step 1's check — is the selection what it claims to be?

Re-verifies every constraint from primary sources (the CSV, GWOSC) rather than trusting
the selector's own bookkeeping. Silent selection bugs to catch: a Chirp slipping through
as a negative, a glitch inside an event veto, a glitch whose PSD block cannot exist, a
class over its cap, a span that doesn't actually cover its glitches.

Run (from WSL2, after stage3_select.py):
    python stage3_select_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage2_fetch import BLOCK_LEN, EVENT_VETO, PAD

from stage3_select import CSV, DEFAULT_OUT, DROP_CLASSES, LEAD, MIN_CONF, O3A, TAIL


def main() -> int:
    with h5py.File(DEFAULT_OUT, "r") as f:
        gps = f["gps"][:]
        label_id = f["label_id"][:]
        conf = f["conf"][:]
        labels = list(f.attrs["labels"])
        cap = int(f.attrs["cap"])
        spans = np.stack([f["span_start"][:], f["span_end"][:]], axis=1)

    names = np.array(labels)[label_id]
    results = []
    print(f"{len(gps):,} selected glitches, {len(spans)} spans, "
          f"{(spans[:,1]-spans[:,0]).sum()/3600:.2f} h")

    # confidence + class rules, re-checked against the raw CSV
    df = pd.read_csv(CSV, usecols=["peak_time", "peak_time_ns", "ml_label", "ml_confidence"])
    df["gps"] = (df.peak_time + df.peak_time_ns * 1e-9).round(6)
    # the CSV holds rows with duplicate times — verify by (gps, label) membership, and
    # take the max confidence among duplicates for the confidence gate
    pairs = set(zip(df.gps, df.ml_label))
    have = all((g, l) in pairs for g, l in zip(np.round(gps, 6), names))
    results.append(("every selected glitch exists in the CSV with matching label",
                    have, "label mismatch"))
    confmax = df[df.ml_confidence >= MIN_CONF]
    conf_pairs = set(zip(confmax.gps, confmax.ml_label))
    conf_ok = all((g, l) in conf_pairs for g, l in zip(np.round(gps, 6), names))
    results.append(("confidence >= 0.9 for all (per the CSV, not the selector)", conf_ok,
                    "a low-confidence specimen slipped in"))
    results.append(("no dropped class present", not bool(np.isin(names, DROP_CLASSES).any()),
                    "a Chirp/No_Glitch slipped in"))
    over = [(c, n) for c, n in pd.Series(names).value_counts().items() if n > cap]
    results.append((f"class caps respected (cap {cap})", not over, f"over: {over}"))

    # geometry + vetoes, re-derived from GWOSC
    from gwosc.datasets import event_gps, find_datasets
    from gwosc.timeline import get_segments

    segs = np.array(get_segments("H1_DATA", *O3A), dtype=float)
    i = np.clip(np.searchsorted(segs[:, 1], gps), 0, len(segs) - 1)
    geo = (segs[i, 0] + LEAD <= gps) & (gps <= segs[i, 1] - TAIL)
    results.append(("every glitch >= LEAD into and >= TAIL before the end of a science segment",
                    bool(geo.all()), f"{int((~geo).sum())} violate"))

    ev = []
    for n in find_datasets(type="events", segment=O3A):
        ev.append(float(event_gps(n)))
    ev = np.unique(np.round(ev, 3))
    dmin = np.min(np.abs(gps[:, None] - ev[None, :]), axis=1)
    results.append((f"no glitch within +-{EVENT_VETO:.0f}s of any of {len(ev)} event datasets",
                    bool((dmin >= EVENT_VETO).all()), f"closest {dmin.min():.0f}s"))

    # spans cover their glitches with room for the causal PSD block, in complete blocks
    covered = np.zeros(len(gps), dtype=bool)
    for a, b in spans:
        n_blocks = int((b - a - 2 * PAD) // BLOCK_LEN)
        block_end = a + PAD + n_blocks * BLOCK_LEN
        m = (gps >= a + LEAD) & (gps + TAIL / 2 <= block_end)
        covered |= m
    results.append(("every glitch inside a span, past block 0, within complete blocks",
                    bool(covered.all()), f"{int((~covered).sum())} uncovered"))
    results.append(("spans disjoint", bool((spans[1:, 0] >= spans[:-1, 1]).all()), "overlap"))

    print()
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {name}" + ("" if passed else f" — {detail}"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

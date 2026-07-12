"""Stage 3, Step 1 — choose the glitches, and plan the cheapest download that covers them.

Input: the Gravity Spy H1 O3a table (Zenodo 5649212, cached CSV). Output:
`~/ligo-data/stage3_selection.h5` — the selected glitches plus the merged, block-aligned
strain spans that stage3_fetch.py will pull.

WHY SELECTION IS A STEP OF ITS OWN: 65k confident glitches spread over 183 days, and a
GWOSC fetch costs ~a minute per 4096 s file. Fetching around every glitch would take days;
fetching everything the greedy file-coverage pass picks takes ~a couple of hours. Glitches
cluster (storms), so the densest files buy hundreds of specimens each.

FILTERS (every one re-verified by stage3_select_check.py):
  - ml_confidence >= 0.9
  - class kept iff >= MIN_CLASS specimens after filters; capped at CAP_PER_CLASS
  - dropped classes: No_Glitch (ordinary noise — Stage 2 holds 50k of it),
    None_of_the_Above (means "unlabelled"), Chirp (chirp-shaped by definition; as a
    NEGATIVE it is a label error aimed at the thing being detected)
  - ±128 s event veto (same as Stage 2)
  - geometry: the glitch must sit >= LEAD s into its science segment (its 512 s causal
    PSD block plus padding must exist before it) and >= TAIL s from the end

SPAN GEOMETRY (the part that bit during design): a naive span [g-532, g+16] holds the
glitch's PSD block but NOT necessarily the whole 512 s block the glitch itself lives in —
stage2_fetch lays only COMPLETE blocks, so the merged span's end is rounded UP to finish
the glitch's block (clipped to the science segment; glitches whose block still cannot be
completed are dropped, counted, and reported). Every kept glitch is then re-checked to
land in block index >= 1 of its stretch.

Run (from WSL2; seconds, no network beyond two GWOSC metadata queries):
    python stage3_select.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage2_fetch import BLOCK_LEN, EVENT_VETO, PAD  # 512.0, 128.0, 8.0

CSV = Path.home() / "ligo-data" / "gravityspy_H1_O3a.csv"
DEFAULT_OUT = Path.home() / "ligo-data" / "stage3_selection.h5"

O3A = (1238166018, 1253977218)
MIN_CONF = 0.9
MIN_CLASS = 150
TARGET_PER_CLASS = 400  # enough for per-class FA statistics with room to train
DROP_CLASSES = ("No_Glitch", "None_of_the_Above", "Chirp")
DEDUP_S = 2.0  # triggers closer than this are one physical event; keep max confidence
MAX_FILES = 200         # download budget: unique 4096 s GWOSC files (~1.2 min each) —
                        # THE binding constraint; download time is what we are rationing
MAX_SPAN_HOURS = 40.0   # safety cap on stored strain (~2.7 GB); storms are span-heavy
                        # but disk and conditioning are cheap next to network

LEAD = 532.0  # s of segment required before a glitch: PAD + one PSD block + margin
TAIL = 16.0   # s required after it


def merged_spans(gs: np.ndarray) -> list[tuple[float, float]]:
    """[g-LEAD, g+TAIL] for each glitch, merged."""
    spans = sorted((g - LEAD, g + TAIL) for g in gs)
    out = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_hours(gs: np.ndarray) -> float:
    return sum(b - a for a, b in merged_spans(gs)) / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Step 1 — select glitches, plan spans")
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    ap.add_argument("--cap", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(CSV, usecols=["peak_time", "peak_time_ns", "duration", "peak_frequency",
                                   "snr", "ml_label", "ml_confidence"])
    df["gps"] = df.peak_time + df.peak_time_ns * 1e-9
    print(f"{len(df):,} glitches in the table")

    df = df[df.ml_confidence >= MIN_CONF]
    df = df[~df.ml_label.isin(DROP_CLASSES)]
    counts = df.ml_label.value_counts()
    df = df[df.ml_label.isin(counts[counts >= MIN_CLASS].index)]
    print(f"{len(df):,} after confidence/class filters ({df.ml_label.nunique()} classes)")

    # DEDUPLICATE: the CSV lists the same physical trigger many times (near-identical
    # peak_times) — the first build discovered this as 1,777 offset collisions out of
    # 2,991 "selected glitches". Cluster within DEDUP_S, keep the highest confidence,
    # class-agnostic: one time, one specimen.
    df = df.sort_values("gps").reset_index(drop=True)
    cluster_id = np.concatenate([[0], np.cumsum(np.diff(df.gps.values) > DEDUP_S)])
    df["cluster"] = cluster_id
    df = df.loc[df.groupby("cluster").ml_confidence.idxmax()].drop(columns="cluster")
    df = df.reset_index(drop=True)
    print(f"{len(df):,} after deduplication (clusters within {DEDUP_S}s collapsed)")

    # --- GWOSC metadata: science segments + event veto (independent of Stage 2's) -----
    from gwosc.datasets import event_gps, find_datasets
    from gwosc.timeline import get_segments

    segs = np.array(get_segments("H1_DATA", *O3A), dtype=float)
    events = {}
    for n in find_datasets(type="events", segment=O3A):
        base = n.split("-")[0]
        if base not in events:
            events[base] = float(event_gps(n))
    ev = np.array(sorted(events.values()))
    print(f"{len(segs)} science segments, {len(ev)} events vetoed")

    g = df.gps.values
    i = np.searchsorted(segs[:, 1], g)  # segment each glitch would live in
    i = np.clip(i, 0, len(segs) - 1)
    in_seg = (segs[i, 0] + LEAD <= g) & (g <= segs[i, 1] - TAIL)
    near_ev = np.zeros(len(g), dtype=bool)
    for e in ev:
        near_ev |= np.abs(g - e) < EVENT_VETO + BLOCK_LEN  # generous: block-width margin
    df = df[in_seg & ~near_ev].copy()
    df["seg_start"] = segs[i, 0][in_seg & ~near_ev]
    df["seg_end"] = segs[i, 1][in_seg & ~near_ev]
    print(f"{len(df):,} after geometry + event veto")

    # --- per-class round-robin greedy under a FILE budget --------------------------------
    # The first version greedily maximised raw coverage and promptly spent the whole
    # budget on three Scattered_Light storm files (1,240 of one class, 10 Blips). The
    # objective has to be BALANCE: cycle through under-target classes, each picking the
    # unpicked file richest in that class; every pick opportunistically takes whatever
    # other under-target specimens the file holds.
    df["file"] = (df.gps // 4096).astype(int)
    picked_files: set[int] = set()
    class_left = {c: args.cap for c in df.ml_label.unique()}
    chosen = []
    by_file = {f: sub for f, sub in df.groupby("file")}
    file_counts = {f: sub.ml_label.value_counts().to_dict() for f, sub in by_file.items()}

    def take_file(f: int) -> None:
        picked_files.add(f)
        for c, grp in by_file[f].groupby("ml_label"):
            if class_left[c] <= 0:
                continue
            take = grp.nlargest(min(class_left[c], len(grp)), "ml_confidence")
            class_left[c] -= len(take)
            chosen.append(take)

    stalled: set[str] = set()
    while len(picked_files) < args.max_files:
        active = [c for c, left in class_left.items() if left > 0 and c not in stalled]
        if not active:
            break
        # neediest class first: largest remaining shortfall
        c = max(active, key=lambda k: class_left[k])
        best, best_n = None, 0
        for f, cnts in file_counts.items():
            if f not in picked_files and cnts.get(c, 0) > best_n:
                best, best_n = f, cnts.get(c, 0)
        if best is None:
            stalled.add(c)  # no unpicked file holds this class any more
            continue
        take_file(best)
        if total_hours(pd.concat(chosen).gps.values) > MAX_SPAN_HOURS:
            print(f"  span-hours cap reached after {len(picked_files)} files")
            break

    print(f"  picked {len(picked_files)} files")
    sel = pd.concat(chosen).sort_values("gps").reset_index(drop=True)

    # --- spans from a PER-SEGMENT block grid --------------------------------------------
    # Blocks live on a fixed grid anchored to each science segment: seg_start + PAD +
    # k*BLOCK_LEN. Each glitch needs exactly two grid blocks — the one holding its
    # buffer (k) and its causal PSD source (k-1) — so the fetch list is the union of
    # those, grouped into runs of consecutive blocks. No end-rounding, no feedback:
    # two earlier versions of this section respectively broke span disjointness and
    # cascaded 56 h of need into 214 h of downloads.
    seg_start = sel.seg_start.values
    p_in_seg = sel.gps.values - (seg_start + PAD)
    k = (p_in_seg // BLOCK_LEN).astype(int)
    p_in_block = p_in_seg - k * BLOCK_LEN
    # placeable: the whole 4 s buffer (glitch at 0.1-0.95 of the crop) fits in block k,
    # k >= 1 so the causal PSD block exists, and block k is COMPLETE inside the science
    # segment (a glitch near segment end lives in a partial block no fetch can finish)
    block_complete = (seg_start + PAD + (k + 1) * BLOCK_LEN + PAD) <= sel.seg_end.values
    keep = (k >= 1) & (p_in_block >= 2.0) & (p_in_block <= BLOCK_LEN - 2.6) & block_complete
    dropped = int((~keep).sum())
    sel = sel[keep].reset_index(drop=True)
    seg_start, k = seg_start[keep], k[keep]

    needed = sorted({(s, b) for s, kk in zip(seg_start, k) for b in (kk - 1, kk)})
    spans = []
    run = [needed[0]]
    for s, b in needed[1:]:
        if s == run[-1][0] and b == run[-1][1] + 1:
            run.append((s, b))
        else:
            spans.append(run)
            run = [(s, b)]
    spans.append(run)
    spans = [
        (s0 + PAD + r[0][1] * BLOCK_LEN - PAD, s0 + PAD + (r[-1][1] + 1) * BLOCK_LEN + PAD)
        for r in spans
        for s0 in (r[0][0],)
    ]

    hours = sum(b - a for a, b in spans) / 3600.0
    files = len({int(t // 4096) for a, b in spans for t in np.arange(a, b, 2048)})
    print(f"\nselected {len(sel):,} glitches ({dropped} dropped at block-completion), "
          f"{len(spans)} spans, {hours:.2f} h of strain, ~{files} GWOSC files")
    print(sel.ml_label.value_counts().to_string())

    labels = sorted(sel.ml_label.unique())
    label_id = {c: i for i, c in enumerate(labels)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("gps", data=sel.gps.values)
        f.create_dataset("label_id", data=sel.ml_label.map(label_id).values.astype(np.int16))
        f.create_dataset("conf", data=sel.ml_confidence.values)
        f.create_dataset("snr", data=sel.snr.values)
        f.create_dataset("duration", data=sel.duration.values)
        f.create_dataset("peak_frequency", data=sel.peak_frequency.values)
        f.create_dataset("span_start", data=np.array([a for a, _ in spans]))
        f.create_dataset("span_end", data=np.array([b for _, b in spans]))
        f.attrs.update(
            labels=labels, min_conf=MIN_CONF, cap=args.cap,
            max_files=args.max_files, lead=LEAD, tail=TAIL,
            source="zenodo 5649212 H1_O3a.csv",
            event_names=sorted(events), event_gps=[events[k] for k in sorted(events)],
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    print(f"\n-> {args.out}\nnow run:  python stage3_select_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

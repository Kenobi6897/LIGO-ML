"""Stage 2, Step 1's check — is the real noise on disk actually what we claim it is?

Four things can be silently wrong with `stage2_strain.h5`, and none would crash anything
downstream — they would just make every Stage 2 number quietly meaningless:

  1. NaNs / non-finite strain (a gap that slipped through) -> poisoned conditioning
  2. A block overlapping an event veto                     -> real signals labelled noise
  3. A block outside science-mode data                     -> not detector noise at all
  4. Blocks sharing samples                                 -> the airtight split, holed

Checks 2 and 3 re-derive the segment list and event list from GWOSC *now*, rather than
trusting the lists the fetcher stored — the check should not share its premise with the
thing it checks (needs network).

Also produces the stage's premise as a figure: the measured O3 ASD against the
aLIGOZeroDetHighPower design curve Stage 1 trained on, plus the per-block in-band RMS
over time — the non-stationarity we are choosing to live with, made visible.

Run (from WSL2, after stage2_fetch.py):
    python stage2_fetch_check.py
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
from stage1_noise_check import BAND, SAMPLE_RATE, design_psd

from stage2_fetch import BLOCK_LEN, BLOCK_N, DEFAULT_OUT, EVENT_VETO, O3A_START

OUT = Path(__file__).parent / "outputs"


def welch_psd(x: np.ndarray, seg_len: int, seg_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-averaged Welch PSD (Hann), matching pycbc's avg_method='median' scaling.

    Median, not mean: a single glitch in one 4 s segment should not drag the whole
    block's spectrum with it. (numpy-only so the check does not import the code under
    test's DSP stack.)
    """
    from scipy.signal import welch as scipy_welch

    freqs, psd = scipy_welch(
        x, fs=SAMPLE_RATE, window="hann", nperseg=seg_len,
        noverlap=seg_len - seg_stride, average="median",
    )
    return freqs, psd


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 1 — check the fetched noise")
    ap.add_argument("--path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offline", action="store_true",
                    help="skip the GWOSC re-queries (checks 2 and 3)")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        n_total = f["strain"].shape[0]
        n_done = int(f.attrs.get("n_done", 0))
        gps = f["gps"][:]
        role = f["role"][:]
        sid = f["stretch_id"][:]
        if n_done < n_total:
            print(f"file is PARTIAL: {n_done}/{n_total} blocks — checking what exists")
        n = n_done
        gps, role, sid = gps[:n], role[:n], sid[:n]

        results: list[tuple[str, bool, str]] = []

        # --- 1. every stored sample finite, and none exactly zero-run (a gap tell) ----
        bad = 0
        zero_runs = 0
        rms = np.empty(n)
        for i in range(n):
            x = f["strain"][i]
            if not np.isfinite(x).all():
                bad += 1
            if (x == 0.0).mean() > 0.001:
                zero_runs += 1
            rms[i] = x.std()
        results.append(("all strain finite", bad == 0, f"{bad} blocks with non-finite samples"))
        results.append(("no zero-filled gaps", zero_runs == 0, f"{zero_runs} blocks >0.1% exact zeros"))

        # --- 4. no two blocks share a sample ------------------------------------------
        order = np.argsort(gps)
        gap_ok = bool((np.diff(gps[order]) >= BLOCK_LEN - 1e-9).all())
        results.append(("blocks disjoint in time", gap_ok, "blocks overlap!"))

        # every stretch leads with a PSD-source block
        lead_ok = all(role[np.flatnonzero(sid == s)[0]] == 0 for s in np.unique(sid))
        results.append(("every stretch leads with a PSD block", lead_ok, "a stretch's first block is data"))

        # --- 2 + 3. re-derive vetoes and science segments from GWOSC ------------------
        if not args.offline:
            from gwosc.datasets import event_gps as ev_gps, find_datasets
            from gwosc.timeline import get_segments

            window = (O3A_START, int(gps[-1] + BLOCK_LEN + 1))
            segs = get_segments("H1_DATA", *window)
            in_science = sum(
                any(a <= g and g + BLOCK_LEN <= b for a, b in segs) for g in gps
            )
            results.append((f"all {n} blocks inside H1 science segments",
                            in_science == n, f"only {in_science}/{n} are"))

            names = find_datasets(type="events", segment=window)
            egps = {nm.split("-")[0]: float(ev_gps(nm)) for nm in names}
            hits = sum(
                any(g - EVENT_VETO < gg + BLOCK_LEN and gg < g + EVENT_VETO for gg in gps)
                for g in egps.values()
            )
            results.append((f"no block within +-{EVENT_VETO:.0f}s of any of {len(egps)} events",
                            hits == 0, f"{hits} events touch a block"))

        # --- the premise figure ---------------------------------------------------------
        psd_rows = np.flatnonzero(role == 0)[:6]
        fig, (axa, axr) = plt.subplots(1, 2, figsize=(13.5, 5))

        freqs = None
        for i in psd_rows:
            freqs, psd = welch_psd(f["strain"][i], 4 * SAMPLE_RATE, 2 * SAMPLE_RATE)
            keep = (freqs >= 20) & (freqs <= 1024)
            axa.loglog(freqs[keep], np.sqrt(psd[keep]), lw=0.7, alpha=0.7,
                       label=f"block @ GPS {gps[i]:.0f}")
        d = design_psd(freqs[1] - freqs[0])
        dn, df = d.numpy(), d.sample_frequencies.numpy()
        keep = (df >= 20) & (df <= 1024) & (dn > 0)
        axa.loglog(df[keep], np.sqrt(dn[keep]), "k", lw=2.2,
                   label="aLIGOZeroDetHighPower (Stage 1 trained on this)")
        axa.axvspan(*BAND, color="tab:orange", alpha=0.12, label=f"analysis band {BAND[0]:.0f}-{BAND[1]:.0f} Hz")
        axa.set_xlabel("frequency [Hz]")
        axa.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
        axa.set_title("Stage 2's premise: real O3 H1 noise is not the design curve\n"
                      "(lines, a different floor, and it moves between blocks)")
        axa.legend(fontsize=7, loc="upper right")
        axa.grid(alpha=0.3, which="both")

        t_hr = (gps - gps[0]) / 3600.0
        axr.plot(t_hr[role == 1], rms[role == 1] * 1e21, ".", ms=3, color="tab:blue", label="data blocks")
        axr.plot(t_hr[role == 0], rms[role == 0] * 1e21, "x", ms=5, color="tab:red", label="PSD-source blocks")
        axr.set_xlabel(f"hours since GPS {gps[0]:.0f}")
        axr.set_ylabel(r"block RMS strain  [$\times 10^{-21}$]")
        axr.set_title("Non-stationarity we are living with\n(train | val | test will be split left to right)")
        axr.legend()
        axr.grid(alpha=0.3)

    print()
    all_ok = True
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + ("" if ok else f" — {detail}"))
        all_ok &= ok

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "1_fetch_check.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

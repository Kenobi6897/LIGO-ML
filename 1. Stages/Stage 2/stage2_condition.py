"""Stage 2, Step 2 — condition real noise: whiten -> bandpass -> crop, per block.

Stage 1's `condition()` with exactly one substitution: the *known design PSD* becomes a
*measured Welch PSD* — and because real noise is non-stationary, "the" PSD becomes one
PSD per 512 s block. Everything else is byte-for-byte the Stage 1 recipe (which
`stage1_gw150914.py` already field-tested on real O1 data): 4 s buffer, whiten, 0.5 s
inverse-spectrum truncation, 30-350 Hz zero-phase FIRs, central 1 s crop, one global
scale constant.

    THE RULE, UNCHANGED: the conditioner never looks at the segment's own content.

How that survives contact with a measured PSD:

  - CAUSAL, PER BLOCK. Block k is whitened with the PSD of block k-1 — the previous
    512 s of the same science stretch. The PSD never sees the data it whitens. (The
    first block of every stretch is a PSD source only; stage2_fetch.py guarantees it.)
  - INJECTIONS HAPPEN DOWNSTREAM. The PSD is estimated from raw detector strain before
    any waveform is added, so it cannot carry a signal fingerprint even in principle.
  - ONE FIXED FILTER PER BLOCK, both classes. Within a block the operator is linear and
    signal-blind, so C(n + h) == C(n) + C(h) exactly — the identity
    stage2_condition_check.py tests to machine precision, per Stage 1's playbook.
  - ONE GLOBAL SCALE. `norm()` is measured once, on raw noise from the earliest data
    blocks (guaranteed train-split: splits are time-ordered), and applied to every
    segment forever. Never per-segment — that is the variance leak.

WHY PER-BLOCK AND NOT ONE GLOBAL PSD: O3 noise drifts on the scale of hours. One global
PSD would whiten Tuesday's noise with Monday's spectrum — still leak-free, but the
whitening residue would grow with time and hand the split boundary to the model as a
feature. 512 s tracks the drift; causality keeps it honest.

Everything here reads `~/ligo-data/stage2_strain.h5` lazily, one handle per process,
because the dataset writer forks 16 workers (same h5py-across-fork rule as stage1_data).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.types import TimeSeries

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_condition import (
    BUFFER_LEN,
    BUFFER_N,
    CROP,
    CROP_LEN,
    CROP_N,
    CROP_START,
    FIR_ORDER,
    INV_SPEC_LEN,
)
from stage1_noise_check import BAND, FLOW, SAMPLE_RATE

from stage2_fetch import BLOCK_LEN, BLOCK_N, DEFAULT_OUT as STRAIN_H5

# A 4 s buffer slides in 1 s hops inside a 512 s block: starts 0..508, crops disjoint.
N_BUFFERS_PER_BLOCK = int(BLOCK_LEN - BUFFER_LEN) + 1  # 509

# The global scale constant is measured on the first N_NORM_BLOCKS data blocks — the
# earliest data, which the time-ordered split guarantees is train. 64 crops (~130k
# samples) pin the std to ~0.2%; every dataset worker recomputes this deterministically
# on first use, so it is kept cheap on purpose.
N_NORM_BLOCKS = 8
NORM_CROP_STRIDE = 64  # every 64th crop -> 8 crops per norm block

_file: h5py.File | None = None


def strain_file(path: Path | str | None = None) -> h5py.File:
    """Per-process lazy handle. Never opened at import, never pickled across a fork.

    The default path is read from the module attribute AT CALL TIME, so a test or a
    smoke run can point the whole module elsewhere with `stage2_condition.STRAIN_H5 =
    other` before anything opens (stage2_dataset relies on this across its fork).
    """
    global _file
    if _file is None:
        _file = h5py.File(path if path is not None else STRAIN_H5, "r")
    return _file


@lru_cache(maxsize=1)
def block_table() -> np.recarray:
    """(row, gps, stretch_id, role) for every COMPLETE block on disk."""
    f = strain_file()
    n = int(f.attrs.get("n_done", f["strain"].shape[0]))
    return np.rec.fromarrays(
        [np.arange(n), f["gps"][:n], f["stretch_id"][:n], f["role"][:n]],
        names="row,gps,stretch_id,role",
    )


def data_rows() -> np.ndarray:
    """Rows usable as data (role 1), each guaranteed a same-stretch predecessor."""
    return block_table().row[block_table().role == 1]


def _psd_source_row(row: int) -> int:
    """The causal PSD source for a data block: the immediately preceding block of the
    same stretch. Fails loudly rather than whiten with the wrong spectrum."""
    t = block_table()
    if t.role[row] != 1:
        raise ValueError(f"block {row} is a PSD source, not data")
    prev = row - 1
    if t.stretch_id[prev] != t.stretch_id[row] or not np.isclose(
        t.gps[prev] + BLOCK_LEN, t.gps[row]
    ):
        raise RuntimeError(f"block {row} has no contiguous predecessor — fetch plan broken")
    return prev


@lru_cache(maxsize=64)
def _measured_psd(row: int):
    """The one Welch estimate everything about block `row` derives from: a median
    Welch (4 s segments, 2 s stride, ~250 averages) on the block's causal predecessor.
    Median, so a single glitch in the PSD block cannot drag the whole spectrum."""
    src = _psd_source_row(row)
    raw = TimeSeries(strain_file()["strain"][src], delta_t=1.0 / SAMPLE_RATE)
    return welch(raw, seg_len=4 * SAMPLE_RATE, seg_stride=2 * SAMPLE_RATE)


@lru_cache(maxsize=64)
def whitening_psd(row: int):
    """Block `row`'s whitening PSD, truncated to a 0.5 s filter (same reason as
    Stage 1: bound the corruption to a length we can afford to crop away)."""
    psd = interpolate(_measured_psd(row), 1.0 / BUFFER_LEN, BUFFER_N // 2 + 1)
    return inverse_spectrum_truncation(
        psd,
        max_filter_len=int(INV_SPEC_LEN * SAMPLE_RATE),
        low_frequency_cutoff=FLOW,
        trunc_method="hann",
    )


@lru_cache(maxsize=64)
def snr_psd(row: int):
    """The same measured spectrum at the crop's delta_f, for sigma()/matched filters.

    NOT the truncated one: truncation reshapes the PSD to make a short time-domain
    filter, which is a whitening concern. SNR integrals want the spectrum as measured.
    """
    return interpolate(_measured_psd(row), 1.0 / CROP_LEN,
                       int(SAMPLE_RATE / 2 / (1.0 / CROP_LEN)) + 1)


def raw_buffer(row: int, offset: int) -> TimeSeries:
    """The 4 s buffer starting `offset` whole seconds into block `row` (0..508).

    Buffers never leave their block, so no crop anywhere shares a raw sample with a
    crop in any other block — the fact the block-level split rests on.
    """
    if not 0 <= offset < N_BUFFERS_PER_BLOCK:
        raise ValueError(f"offset {offset} outside 0..{N_BUFFERS_PER_BLOCK - 1}")
    i0 = offset * SAMPLE_RATE
    x = strain_file()["strain"][row, i0 : i0 + BUFFER_N]
    return TimeSeries(np.asarray(x, dtype=np.float64), delta_t=1.0 / SAMPLE_RATE)


def _condition_unnormed(buf: TimeSeries, psd) -> np.ndarray:
    """whiten -> bandpass -> crop. Identical shape to Stage 1's, PSD passed in."""
    white = (buf.to_frequencyseries() / psd**0.5).to_timeseries()
    band = highpass_fir(white, BAND[0], FIR_ORDER)
    band = lowpass_fir(band, BAND[1], FIR_ORDER)
    return band.numpy()[CROP]


@lru_cache(maxsize=1)
def norm() -> float:
    """One global scale so conditioned real NOISE has ~unit variance.

    Measured on raw (pre-injection) crops from the earliest data blocks — train split
    by construction, noise by construction. One number for every segment forever; it
    commutes with everything and can encode nothing about any given segment.
    """
    rows = data_rows()[:N_NORM_BLOCKS]
    if len(rows) == 0:
        raise RuntimeError("no complete data blocks on disk yet — run stage2_fetch.py")
    stds = [
        _condition_unnormed(raw_buffer(int(r), off), whitening_psd(int(r))).std()
        for r in rows
        for off in range(0, N_BUFFERS_PER_BLOCK, NORM_CROP_STRIDE)
    ]
    return 1.0 / float(np.mean(stds))


def condition_real(buf: TimeSeries, row: int) -> np.ndarray:
    """The CNN's input: 2048 float32 samples, conditioned with block `row`'s fixed filter."""
    if len(buf) != BUFFER_N:
        raise ValueError(f"expected a {BUFFER_LEN}s buffer ({BUFFER_N} samples), got {len(buf)}")
    return (_condition_unnormed(buf, whitening_psd(row)) * norm()).astype(np.float32)


if __name__ == "__main__":
    t = block_table()
    rows = data_rows()
    print(f"{len(t)} complete blocks on disk ({len(rows)} data), norm blocks = {rows[:N_NORM_BLOCKS]}")
    print(f"norm = {norm():.6e}")
    x = condition_real(raw_buffer(int(rows[0]), 100), int(rows[0]))
    print(f"one conditioned crop: shape {x.shape}, dtype {x.dtype}, std {x.std():.3f}")

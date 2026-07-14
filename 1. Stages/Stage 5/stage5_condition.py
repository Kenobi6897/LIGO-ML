"""Stage 5, Step 1 — four conditioners. The ONE variable this stage changes.

Stages 1-4 asked what the network learns. This one asks what the *physics we handed it*
was worth, by taking it away. Same strain, same blocks, same injections at the same
physical amplitudes, same architecture, same recipe — only the operator between the
detector and the CNN changes:

    raw   crop only                      no whitening, no bandpass. "Just feed it strain."
    bp    bandpass 30-350 -> crop        band-limited, still coloured inside the band
    wh    whiten -> crop                 flat spectrum, no band limit (15-1024 Hz)
    full  whiten -> bandpass -> crop     Stages 2-4's condition_real(). The control.

WHY THIS IS A FAIR FIGHT, AND NOT A STRAWMAN. Three ways a raw-data arm can be rigged to
fail, all closed here:

  1. SCALE. Raw strain is ~1e-19; a network fed 1e-19 fails for reasons that have nothing
     to do with spectra. Every arm therefore gets its OWN global scale constant, measured
     by the same signal-blind recipe as Stage 2's norm() (mean std of pure-noise crops
     from the earliest train blocks, one number forever, never per-segment). Each arm's
     input is O(1). BatchNorm after conv1 would mostly absorb a scale error anyway; this
     removes the argument entirely.
  2. THE SIGNAL. Injection amplitude is set upstream, in strain, against the block's
     measured PSD — identically for every arm (stage2_dataset.build_segment's code path,
     reused verbatim). "SNR 8" is the same physical waveform in the same noise in all four
     arms. The conditioner is the only thing that differs.
  3. LINEARITY. All four operators are linear and signal-blind, so Stage 1's identity
     C(n + h) == C(n) + C(h) holds for each of them, and stage5_dataset_check.py proves it
     to machine precision per arm. No arm gets a leak the others don't.

WHAT WE EXPECT TO BREAK, STATED BEFORE THE RUN (so a surprise is recognisable):
raw strain's variance is dominated by the seismic/suspension wall below ~30 Hz, orders of
magnitude above the 100-300 Hz bucket the signals live in. An SNR-8 chirp is ~1e-4 of a
raw crop's standard deviation. A 64-tap conv kernel CAN in principle learn a high-pass —
what it cannot learn is Sigma^-1/2, because whitening here is DATA-ADAPTIVE (a causal
per-block Welch PSD, because the noise drifts), and a convolution is a fixed filter.
The prediction: `bp` recovers most of `full`, `raw` collapses, and the collapse is worst
exactly where Stage 2 showed real noise is worst — in the tail, at deep thresholds.

Everything reads through stage2_condition (blocks, causal PSDs, raw_buffer), so the PSD
discipline — block k whitened by block k-1, never its own content — is inherited, not
reimplemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.types import TimeSeries

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_condition import BUFFER_N, CROP, FIR_ORDER
from stage1_noise_check import BAND

import stage2_condition as s2c

# The three arms this stage BUILDS. `full` already exists on disk as stage2.h5 / stage3.h5
# and is never rebuilt — rebuilding the control would risk moving it.
BUILD_ARMS = ("raw", "bp", "wh")
ARMS = BUILD_ARMS + ("full",)

# Read-time saturation, inherited from Stage 3 unchanged. Applied identically to every
# arm (in that arm's own units), because a difference in read-time treatment would be a
# second variable. stage5_dataset_check.py reports how much each arm actually clips.
CLIP_SIGMA = 20.0

_needs_psd = {"wh", "full"}
_norms: dict[tuple[str, str], float] = {}


def _unnormed(buf: TimeSeries, psd, arm: str) -> np.ndarray:
    """The arm's operator, without the global scale. Linear and signal-blind, all four."""
    if arm == "raw":
        # No filter at all. The crop is taken at the same samples as every other arm, so
        # the arms differ by their operator and by nothing else — not even by geometry.
        return buf.numpy()[CROP]
    if arm == "bp":
        band = highpass_fir(buf, BAND[0], FIR_ORDER)
        band = lowpass_fir(band, BAND[1], FIR_ORDER)
        return band.numpy()[CROP]
    if arm == "wh":
        # inverse_spectrum_truncation already kills everything below FLOW=15 Hz, so this
        # is "flat 15-1024 Hz", not "flat DC-1024 Hz".
        return (buf.to_frequencyseries() / psd**0.5).to_timeseries().numpy()[CROP]
    if arm == "full":
        return s2c._condition_unnormed(buf, psd)
    raise ValueError(f"unknown arm {arm!r} — expected one of {ARMS}")


def norm(arm: str) -> float:
    """One global scale per arm, so that arm's conditioned NOISE has ~unit variance.

    Stage 2's recipe, unchanged and re-derived per arm: pure noise crops (pre-injection,
    by construction) from the first N_NORM_BLOCKS data blocks, which the time-ordered
    split guarantees are train. One number for every segment forever.

    Keyed on the strain file as well as the arm, because stage3_dataset.py points
    stage2_condition at a different strain file — so Stage 3's crops already carried a
    different constant from Stage 2's. Mirrored here rather than "fixed": the control arm
    on disk was built that way, and the control is not a thing this stage gets to move.
    """
    key = (arm, str(s2c.STRAIN_H5))
    if key not in _norms:
        rows = s2c.data_rows()[: s2c.N_NORM_BLOCKS]
        if len(rows) == 0:
            raise RuntimeError(f"no complete data blocks in {s2c.STRAIN_H5}")
        stds = [
            _unnormed(
                s2c.raw_buffer(int(r), off),
                s2c.whitening_psd(int(r)) if arm in _needs_psd else None,
                arm,
            ).std()
            for r in rows
            for off in range(0, s2c.N_BUFFERS_PER_BLOCK, s2c.NORM_CROP_STRIDE)
        ]
        _norms[key] = 1.0 / float(np.mean(stds))
    return _norms[key]


def condition(buf: TimeSeries, row: int, arm: str) -> np.ndarray:
    """The CNN's input under `arm`: 2048 float32 samples from block `row`'s buffer."""
    if len(buf) != BUFFER_N:
        raise ValueError(f"expected a {BUFFER_N}-sample buffer, got {len(buf)}")
    psd = s2c.whitening_psd(row) if arm in _needs_psd else None
    return (_unnormed(buf, psd, arm) * norm(arm)).astype(np.float32)


def reset_caches(strain_path) -> None:
    """Fork hygiene, Stage 2's rule plus this module's own cache."""
    from stage2_dataset import _worker_init

    _worker_init(strain_path)
    _norms.clear()


if __name__ == "__main__":
    rows = s2c.data_rows()
    r = int(rows[0])
    print(f"{len(rows)} data blocks in {s2c.STRAIN_H5}\n")
    print(f"{'arm':>5}  {'norm':>12}  {'crop std':>9}  {'crop max|x|':>11}")
    for arm in ARMS:
        x = condition(s2c.raw_buffer(r, 100), r, arm)
        print(f"{arm:>5}  {norm(arm):12.4e}  {x.std():9.3f}  {np.abs(x).max():11.2f}")

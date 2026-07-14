"""Stage 1, Step 3 — condition(): whiten -> bandpass -> crop. The one the CNN eats.

This is the most dangerous function in the project, so it is the smallest.

    THE RULE: `condition()` never looks at the segment's own content.

It whitens with the *known* design PSD — the one we generated the noise from — and with
a fixed global scale. That makes it a **linear, signal-blind operator**: the same matrix
multiplies every segment, positive or negative, and

    condition(noise + signal) == condition(noise) + condition(signal)

exactly. `stage1_condition_check.py` tests that identity to machine precision, because
it is the property that separates a correct pipeline from the 99%-AUC fraud. Anything
estimated per-segment -- a Welch PSD, a unit-variance rescale -- breaks the identity, and
the CNN learns the fingerprint of the whitener instead of the shape of a chirp.

See [[Stage 1#Footguns]]. `condition_leaky()` at the bottom is the wrong version, kept
only so the check can show it failing.

WHY A 4 s BUFFER FOR A 1 s SEGMENT: whitening and band-passing are convolutions, so they
corrupt both ends of whatever you hand them. We generate 4 s, condition it, and crop the
central 1 s -- which is 1.5 s clear of either edge. Crop from a longer buffer; never
"fix" the edges of a short one.
"""

from functools import lru_cache

import numpy as np
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import TimeSeries

from stage1_noise_check import BAND, FLOW, SAMPLE_RATE, design_psd

# --- geometry -----------------------------------------------------------------
BUFFER_LEN = 4.0  # s generated per segment
CROP_LEN = 1.0  # s handed to the CNN  — the Stage 1 decision
BUFFER_N = int(BUFFER_LEN * SAMPLE_RATE)  # 8192
CROP_N = int(CROP_LEN * SAMPLE_RATE)  # 2048
CROP_START = (BUFFER_N - CROP_N) // 2  # centred: samples 3072:5120
CROP = slice(CROP_START, CROP_START + CROP_N)

# Corruption budgets. Both are < 0.5 s; the crop sits 1.5 s from each edge.
INV_SPEC_LEN = 0.5  # s — inverse spectrum truncation: kills the whitening filter's tails
FIR_ORDER = 256  # samples corrupted each side by each FIR (0.125 s)


@lru_cache(maxsize=1)
def whitening_psd():
    """The design PSD, truncated to a short filter. Known, fixed, identical for all data.

    `inverse_spectrum_truncation` matters: 1/ASD has a huge dynamic range below 30 Hz, so
    the raw inverse filter rings for seconds. Truncating it to 0.5 s bounds the corrupted
    region to a length we can afford to crop away.
    """
    psd = design_psd(1.0 / BUFFER_LEN)
    return inverse_spectrum_truncation(
        psd,
        max_filter_len=int(INV_SPEC_LEN * SAMPLE_RATE),
        low_frequency_cutoff=FLOW,
        trunc_method="hann",
    )


@lru_cache(maxsize=1)
def _norm() -> float:
    """One global scale factor, so conditioned NOISE has ~unit variance.

    Two things this is not:
      - not per-segment (that is the [[Stage 1#Footguns|unit-variance leak]]: adding a
        signal raises the variance, so a per-segment rescale tells the CNN the label);
      - not fitted to signals. It is measured on pure noise, from a fixed seed, once.

    Same number for every segment forever, so it commutes with everything and cannot
    encode anything about a given segment. It exists only to hand the network O(1) inputs.
    """
    from stage1_noise_check import generate_noise

    stds = [
        _condition_unnormed(generate_noise(BUFFER_LEN, seed=900_000 + i)).std()
        for i in range(8)
    ]
    return 1.0 / float(np.mean(stds))


def _condition_unnormed(buf: TimeSeries) -> np.ndarray:
    """whiten -> bandpass -> crop, without the global scale (which _norm() calibrates)."""
    # Whiten: divide by the KNOWN amplitude spectral density. Never one estimated here.
    white = (buf.to_frequencyseries() / whitening_psd() ** 0.5).to_timeseries()

    # Bandpass to the band we care about. Whitening flattens everything, including the
    # near-empty 400-1024 Hz region, so without this the network's input is mostly
    # high-frequency noise it can do nothing with.
    band = highpass_fir(white, BAND[0], FIR_ORDER)
    band = lowpass_fir(band, BAND[1], FIR_ORDER)

    # Crop away the corrupted edges. Everything above is a convolution; this is the part
    # of the buffer no filter tail ever reached.
    return band.numpy()[CROP]


def condition(buf: TimeSeries) -> np.ndarray:
    """The CNN's input: 2048 float32 samples of whitened, band-limited strain.

    Linear and signal-blind by construction — see the module docstring.
    """
    if len(buf) != BUFFER_N:
        raise ValueError(f"expected a {BUFFER_LEN}s buffer ({BUFFER_N} samples), got {len(buf)}")
    return (_condition_unnormed(buf) * _norm()).astype(np.float32)


# --- the wrong one ------------------------------------------------------------


def condition_leaky(buf: TimeSeries) -> np.ndarray:
    """DO NOT USE. Whitens with a PSD estimated from the segment itself — [[Stage 0]]'s
    `.whiten()`, and the headline footgun.

    A segment with a loud injection gets whitened *differently* from a quiet one: the
    signal inflates the PSD estimate in its own band, and the whitener then divides that
    band back down. The operation carries a fingerprint of the signal. Kept here purely so
    `stage1_condition_check.py` can demonstrate it failing the linearity test that
    `condition()` passes.
    """
    white = buf.whiten(2.0, INV_SPEC_LEN, low_frequency_cutoff=FLOW, remove_corrupted=False)
    band = highpass_fir(white, BAND[0], FIR_ORDER)
    band = lowpass_fir(band, BAND[1], FIR_ORDER)
    return band.numpy()[CROP].astype(np.float32)

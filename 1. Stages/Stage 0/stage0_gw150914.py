"""
Stage 0 — See the Chirp.

Fetch real LIGO strain around GW150914, whiten it, bandpass it, and look at it.
No machine learning. The point is to prove the preprocessing chain works on a case
where we already know the right answer, before any of it is load-bearing.

Success criterion is not a number: can I see the chirp?

Outputs three PNGs into outputs/:
    1_chirp_per_detector.png   whitened + bandpassed strain, H1 and L1 separately
    2_overlay.png              H1 and L1 on top of each other  <- the money plot
    3_qtransform.png           time-frequency spectrogram      <- the convincing one

Runs natively on Windows: pure gwpy/scipy, no lalsuite. See requirements-stage0.txt
for the igwn-segments pin that makes `pip install gwpy` work here at all.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

# The merger, to the millisecond. Using 1126259462.4 puts the chirp 23 ms off-centre,
# which is harmless in a plot and quietly misaligns every training segment in Stage 1.
GPS_MERGER = 1126259462.423

# Fetch 32 s to look at 0.3 s. The margin is fuel for the PSD estimate and the crop:
# filtering and PSD estimation corrupt segment edges, so the outer second of each side
# is a filter transient, not data, and gets thrown away.
PAD = 16
START = GPS_MERGER - PAD
END = GPS_MERGER + PAD
CROP = 1.0

DETECTORS = ("H1", "L1")

# GW150914's power lives in 30-350 Hz. Below is seismic, above is shot noise.
BAND = (30, 350)

# Livingston saw the wave first -- it swept across the Earth at c and the detectors are
# ~3000 km apart. And L1's arms are oriented such that its response has opposite sign.
# Correct for both and two independent instruments trace the same squiggle.
L1_TIME_SHIFT = 0.0069
L1_SIGN = -1

ZOOM = (-0.15, 0.05)  # seconds relative to merger

OUTDIR = Path(__file__).parent / "outputs"


def fetch(ifo):
    """Real, public, archival LIGO strain from GWOSC. 4096 Hz."""
    return TimeSeries.fetch_open_data(ifo, START, END, cache=True)


def condition(strain):
    """Whiten, bandpass, drop the corrupted edges.

    Whitening divides by the noise amplitude spectral density so every frequency
    contributes equally. Raw strain is utterly dominated by low-frequency seismic
    noise; without this step there is simply nothing to see.

    NOTE for Stage 1: .whiten() estimates the PSD from the very segment it is
    whitening. That is fine here -- a 0.2 s signal barely perturbs a PSD estimated
    over 32 s -- but it is catastrophic for a training set, because the whitening
    then carries a fingerprint of the injection and the CNN learns the fingerprint
    instead of the chirp. Estimate the PSD off a separate stretch there.
    """
    white = strain.whiten(4, 2)  # fftlength=4 s, overlap=2 s
    band = white.bandpass(*BAND)
    return band.crop(START + CROP, END - CROP)


def relative_times(series):
    return series.times.value - GPS_MERGER


def plot_per_detector(conditioned):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, ifo in zip(axes, DETECTORS):
        ax.plot(relative_times(conditioned[ifo]), conditioned[ifo].value, lw=1)
        ax.set_xlim(*ZOOM)
        ax.set_ylabel("whitened strain")
        ax.set_title(f"{ifo} — whitened, bandpassed {BAND[0]}–{BAND[1]} Hz")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"time from merger (s) — GPS {GPS_MERGER}")
    fig.tight_layout()
    fig.savefig(OUTDIR / "1_chirp_per_detector.png", dpi=150)
    plt.close(fig)


def plot_overlay(conditioned):
    """The physics. Shift L1 forward by 6.9 ms, invert it, lay it on H1."""
    h1, l1 = conditioned["H1"], conditioned["L1"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(relative_times(h1), h1.value, lw=1.5, label="H1")
    ax.plot(
        relative_times(l1) + L1_TIME_SHIFT,
        L1_SIGN * l1.value,
        lw=1.5,
        alpha=0.85,
        label=f"L1 (shifted +{L1_TIME_SHIFT * 1e3:.1f} ms, inverted)",
    )
    ax.set_xlim(*ZOOM)
    ax.set_xlabel(f"time from merger (s) — GPS {GPS_MERGER}")
    ax.set_ylabel("whitened strain")
    ax.set_title("GW150914 — H1 and L1 agree. Two instruments, 3000 km apart.")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "2_overlay.png", dpi=150)
    plt.close(fig)


def plot_qtransform(raw):
    """The convincing one: the chirp as a visible arc sweeping 35 -> 250 Hz.

    Two gotchas, both hit the first time round:
      - q_transform whitens internally, so it takes the RAW strain. Passing the
        already-whitened series whitens twice.
      - normalized energy can go slightly negative, and sqrt of that is NaN, which
        renders as white speckle across the plot. Clip before rooting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, ifo in zip(axes, DETECTORS):
        q = raw[ifo].q_transform(
            outseg=(GPS_MERGER + ZOOM[0], GPS_MERGER + ZOOM[1]),
            frange=(20, 400),
            qrange=(4, 64),
        )
        mesh = ax.pcolormesh(
            q.times.value - GPS_MERGER,
            q.frequencies.value,
            np.sqrt(np.clip(q.value.T, 0, None)),
            shading="auto",
            cmap="viridis",
        )
        ax.set_yscale("log")
        ax.set_ylim(20, 400)
        ax.set_xlabel(f"time from merger (s)")
        ax.set_title(f"{ifo} — Q-transform")
        fig.colorbar(mesh, ax=ax, label="√(normalized energy)")
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle("GW150914 — energy climbing ~35 → 250 Hz, cut off at merger")
    fig.tight_layout()
    fig.savefig(OUTDIR / "3_qtransform.png", dpi=150)
    plt.close(fig)


def main():
    OUTDIR.mkdir(exist_ok=True)

    print(f"fetching {2 * PAD} s around GPS {GPS_MERGER} from GWOSC ...")
    raw = {ifo: fetch(ifo) for ifo in DETECTORS}
    for ifo, series in raw.items():
        print(f"  {ifo}: {len(series)} samples @ {series.sample_rate}")

    print(f"whitening, bandpassing {BAND[0]}–{BAND[1]} Hz, cropping {CROP} s edges ...")
    conditioned = {ifo: condition(series) for ifo, series in raw.items()}

    print("plotting ...")
    plot_per_detector(conditioned)
    plot_overlay(conditioned)
    plot_qtransform(raw)

    print(f"done — three plots in {OUTDIR}")


if __name__ == "__main__":
    main()

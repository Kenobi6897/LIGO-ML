"""Stage 1, Step 7 — the payoff figure: plot the first-layer conv kernels.

A 1D convolution slides a kernel along the strain computing a dot product at every lag —
which is exactly what correlating against a template does. So layer 1 of the trained CNN
is, functionally, a **learned template bank**: 16 kernels of 64 samples (31 ms) that the
network invented on its own, from labels alone, without ever being told what a black hole
is. This figure is the reason the project went 1D ([[1D vs 2D - decision explained]]) —
a 2D network's first layer would be edge detectors over a spectrogram, with no
matched-filtering reading at all.

WHAT TO EXPECT: oscillatory, band-limited, wavelet-like filters — short chirp *snippets*,
not whole chirps. A 31 ms window can hold ~2 cycles at 60 Hz or ~8 at 250 Hz, so no single
kernel can sweep 30->350 Hz; the bank covers the band collectively, the way a template
bank covers a mass range, and deeper layers assemble the sweep from these pieces. For
comparison the figure ends with the same 31 ms of an actual conditioned chirp at merger.

THE CHECK THAT CAN FAIL: at least 12 of 16 kernels must have their spectral peak inside
20-400 Hz (the analysis band, with one bin of slack). The input is band-passed to
30-350 Hz, so a kernel doing useful work concentrates its energy there. Random-init
kernels are spectrally flat — peak uniform over 0-1024 Hz, so ~6/16 would land in band by
luck. 12/16 says the training moved the bank into the band; it is a weak-ish bar on
purpose, because dropout leaves some kernels near their initialisation in any small net.

Run (from WSL2, after stage1_train.py):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_kernels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np

from stage1_model import FIRST_KERNEL, load_checkpoint
from stage1_train import DEFAULT_CKPT

OUT = Path(__file__).parent / "outputs"

SAMPLE_RATE = 2048
BAND = (30.0, 350.0)
PEAK_BAND = (20.0, 400.0)  # the check's window: the band plus one 4 Hz bin of slack
N_PAD = 512  # zero-pad the 64-sample FFT so the spectra are readable (4 Hz bins)


def kernel_spectra(kernels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, power) — zero-padded power spectrum of each kernel, unit peak."""
    power = np.abs(np.fft.rfft(kernels, n=N_PAD, axis=1)) ** 2
    power /= power.max(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(N_PAD, 1.0 / SAMPLE_RATE)
    return freqs, power


def band_limited(kernels: np.ndarray) -> np.ndarray:
    """Each kernel's 30-350 Hz component — the only part of the weights that band-passed
    input can excite. Out-of-band weight content is functionally inert (the data has no
    power there), so this is the part of each kernel that actually does the filtering."""
    F = np.fft.rfft(kernels, axis=1)
    f = np.fft.rfftfreq(kernels.shape[1], 1.0 / SAMPLE_RATE)
    return np.fft.irfft(F * ((f >= BAND[0]) & (f <= BAND[1])), n=kernels.shape[1], axis=1)


def reference_chirp() -> np.ndarray:
    """31 ms of a conditioned 36+29 Msun chirp at SNR 12, ending at merger — what a
    'real' template looks like after the exact preprocessing the CNN sees.
    condition() is signal-blind, so running it on a pure waveform is legal.

    The SNR 12 rescale is not cosmetic: conditioning the RAW IMRPhenomD waveform (pycbc's
    default distance) puts a signal of SNR ~thousands on the axis — the same
    measured-a-signal-that-isn't-in-the-file mistake as [[Stage 1#🐛 Bugs the checks
    caught|bug #3]], here it would merely mislabel the y-axis."""
    from stage1_condition import condition
    from stage1_dataset import in_band_sigma, place_in_buffer
    from stage1_injection_check import make_waveform

    merger_pos = 0.85
    placed = place_in_buffer(make_waveform(36.0, 29.0), merger_pos)
    placed = placed * (12.0 / in_band_sigma(placed))
    conditioned = condition(placed)  # numpy, the central 2048 samples
    end = int(merger_pos * len(conditioned)) + 4  # a hair past the peak
    return conditioned[end - FIRST_KERNEL : end]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 Step 7 — plot the learned template bank")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt)
    kernels = model.first_layer_kernels()  # (16, 64)
    n_k, k_len = kernels.shape
    print(f"checkpoint: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}")
    print(f"first layer: {n_k} kernels x {k_len} samples ({1000*k_len/SAMPLE_RATE:.0f} ms)\n")

    freqs, power = kernel_spectra(kernels)
    peak_hz = freqs[np.argmax(power, axis=1)]
    order = np.argsort(peak_hz)  # display low -> high frequency, like a template bank

    in_band = (peak_hz >= PEAK_BAND[0]) & (peak_hz <= PEAK_BAND[1])
    print("kernel spectral peaks (Hz), sorted:")
    print("  " + "  ".join(f"{peak_hz[i]:5.0f}" for i in order))
    print(f"\n  in {PEAK_BAND[0]:.0f}-{PEAK_BAND[1]:.0f} Hz: {in_band.sum()}/{n_k}"
          f"   (random init would average ~{n_k * (PEAK_BAND[1]-PEAK_BAND[0])/1024:.0f})")

    ok = int(in_band.sum()) >= 12
    print(f"\n  {'PASS' if ok else 'FAIL'}: {in_band.sum()}/{n_k} kernels peak in band "
          f"(want >= 12) — the network {'built' if ok else 'did NOT build'} its template "
          f"bank inside the band the signals live in")

    # --- figure: 4x4 kernel grid + spectra + the reference chirp snippet -------------
    t_ms = np.arange(k_len) / SAMPLE_RATE * 1000
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 1.35], hspace=0.55, wspace=0.25)

    smooth = band_limited(kernels)
    for slot, i in enumerate(order):
        ax = fig.add_subplot(gs[slot // 4, slot % 4])
        ax.plot(t_ms, kernels[i], lw=0.8, alpha=0.35,
                color="tab:blue" if in_band[i] else "tab:gray")
        ax.plot(t_ms, smooth[i], lw=1.5,
                color="tab:blue" if in_band[i] else "tab:gray")
        ax.set_title(f"k{i}  —  peak ≈ {peak_hz[i]:.0f} Hz", fontsize=9)
        ax.set_xticks([0, 15, 31])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if slot >= 12:
            ax.set_xlabel("ms", fontsize=8)

    axs = fig.add_subplot(gs[4, :2])
    for i in order:
        axs.plot(freqs, power[i], lw=1.0,
                 color="tab:blue" if in_band[i] else "tab:gray", alpha=0.6)
    axs.axvspan(*BAND, color="tab:orange", alpha=0.12,
                label=f"analysis band {BAND[0]:.0f}-{BAND[1]:.0f} Hz")
    axs.set_xlim(0, 700)
    axs.set_xlabel("frequency [Hz]")
    axs.set_ylabel("kernel power (unit peak)")
    axs.set_title(f"Where the bank listens — {in_band.sum()}/{n_k} peak in band", fontsize=10)
    axs.legend(loc="upper right", fontsize=8)
    axs.grid(alpha=0.3)

    axc = fig.add_subplot(gs[4, 2:])
    axc.plot(t_ms, reference_chirp(), lw=1.4, color="tab:red")
    axc.set_xlabel("ms")
    axc.set_ylabel("whitened strain [σ]")
    axc.set_title("For comparison: the last 31 ms of a conditioned\n"
                  "36+29 M☉ chirp at SNR 12 (merger at right edge)", fontsize=10)
    axc.grid(alpha=0.3)

    fig.suptitle(
        "Stage 1, Step 7 — the learned template bank\n"
        "16 first-layer conv kernels (a conv kernel IS a matched filter), sorted by peak "
        "frequency.\nDark line = the kernel's 30-350 Hz component, the only part "
        "band-passed data can excite. Nobody told the network what a black hole is.",
        fontsize=12,
    )
    OUT.mkdir(exist_ok=True)
    path = OUT / "7_kernels.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

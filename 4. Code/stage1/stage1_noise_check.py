"""Stage 1, Step 1 — the noise generator, and the check that it is right.

Generates coloured Gaussian noise at aLIGO design sensitivity and verifies that its
*measured* PSD lands back on the design curve it was drawn from.

Why bother: every positive in Stage 1 is "a negative with a waveform added", so this
generator produces BOTH classes. If its spectrum is wrong, the injected SNRs are wrong,
and every number downstream is quietly wrong with them.

Run (from WSL2):
    source ~/venvs/ligo/bin/activate
    python "4. Code/stage1/stage1_noise_check.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, welch

# --- Stage 1 decisions, pinned in the plan note ---
SAMPLE_RATE = 2048  # Hz — Nyquist 1024, well above our 30–350 Hz band
DURATION = 256  # s of noise to test the spectrum against
FLOW = 15.0  # Hz — generate below the analysis band, so the band isn't edge-affected
BAND = (30.0, 350.0)  # the band we actually care about
SEED = 42

OUT = Path(__file__).parent / "outputs"


def design_psd(delta_f: float) -> "pycbc.types.FrequencySeries":
    """The aLIGO design curve — the *known* PSD we colour the noise with."""
    flen = int(SAMPLE_RATE / 2 / delta_f) + 1
    return aLIGOZeroDetHighPower(flen, delta_f, FLOW)


def generate_noise(duration: float, seed: int):
    """Coloured Gaussian noise at design sensitivity.

    This same function makes positives and negatives — a positive is just this, plus a
    waveform. Identical treatment of both classes is what keeps the labels honest.
    """
    delta_f = 1.0 / 16.0  # 16 s PSD resolution; independent of the segment length
    psd = design_psd(delta_f)
    return noise_from_psd(int(duration * SAMPLE_RATE), 1.0 / SAMPLE_RATE, psd, seed=seed)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(f"generating {DURATION}s of noise @ {SAMPLE_RATE} Hz (seed={SEED}) ...")
    noise = generate_noise(DURATION, SEED)

    # Measure the PSD back off the generated noise. Welch over 4 s segments.
    measured = welch(noise, seg_len=4 * SAMPLE_RATE, seg_stride=2 * SAMPLE_RATE)
    truth = interpolate(design_psd(1.0 / 16.0), measured.delta_f, len(measured))

    freqs = measured.sample_frequencies.numpy()
    in_band = (freqs >= BAND[0]) & (freqs <= BAND[1]) & (truth.numpy() > 0)

    ratio = measured.numpy()[in_band] / truth.numpy()[in_band]
    med, lo, hi = np.median(ratio), *np.percentile(ratio, [5, 95])

    print(f"\nmeasured/design PSD ratio over {BAND[0]:.0f}-{BAND[1]:.0f} Hz:")
    print(f"  median      : {med:.3f}   (want ~1.00)")
    print(f"  5th-95th pct: {lo:.3f} - {hi:.3f}   (Welch scatter, not bias)")

    # The median is the real test: Welch estimates are noisy per-bin (chi-squared
    # scatter), but they must not be biased. A median off 1.0 means the colouring
    # is wrong, and every SNR we compute later would inherit the error.
    ok = 0.9 <= med <= 1.1
    print(f"\n  {'PASS' if ok else 'FAIL'}: noise spectrum {'matches' if ok else 'does NOT match'} the design curve")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    band = (freqs >= 20) & (freqs <= 1024) & (truth.numpy() > 0)
    ax.loglog(freqs[band], np.sqrt(measured.numpy()[band]), lw=0.6, alpha=0.8,
              label="measured (Welch, from generated noise)")
    ax.loglog(freqs[band], np.sqrt(truth.numpy()[band]), lw=2.0, color="k",
              label="aLIGOZeroDetHighPower (design)")
    ax.axvspan(*BAND, color="tab:orange", alpha=0.12, label=f"analysis band {BAND[0]:.0f}-{BAND[1]:.0f} Hz")
    ax.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
    ax.set_title(f"Stage 1 noise generator — {DURATION}s @ {SAMPLE_RATE} Hz  (median ratio {med:.3f})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")

    axr.semilogx(freqs[band], measured.numpy()[band] / truth.numpy()[band], lw=0.6, alpha=0.8)
    axr.axhline(1.0, color="k", lw=1.5)
    axr.axvspan(*BAND, color="tab:orange", alpha=0.12)
    axr.set_ylim(0, 3)
    axr.set_ylabel("measured / design")
    axr.set_xlabel("frequency [Hz]")
    axr.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = OUT / "1_psd_check.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

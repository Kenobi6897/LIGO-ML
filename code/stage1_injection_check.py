"""Stage 1, Step 2 — waveform generation + SNR scaling, and the check that it is right.

A positive is a negative with a waveform added at a *known* SNR. This script proves the
"known" part: request SNR x, then measure it back with an independent matched filter and
confirm we get x.

Why this check earns its keep: the SNR we request becomes the x-axis of the money plot
(detection efficiency vs. injected SNR). If the scaling is wrong, every point on that
axis is mislabelled, the curve is nonsense, and nothing about the result would look
wrong -- it would just be wrong.

Run:  source ~/venvs/ligo/bin/activate && python code/stage1_injection_check.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycbc.filter import matched_filter, sigma
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform

from stage1_noise_check import FLOW, SAMPLE_RATE, generate_noise

SEG_LEN = 1.0  # s   — Stage 1 decision
DELTA_F = 1.0 / SEG_LEN
FLEN = int(SAMPLE_RATE / 2 / DELTA_F) + 1
PSD = aLIGOZeroDetHighPower(FLEN, DELTA_F, FLOW)

F_LOWER = 30.0  # Hz — matches the analysis band; also where sigma integrates from
MASS_RANGE = (10.0, 50.0)  # solar masses, per the plan
OUT = Path(__file__).parent / "outputs"


def make_waveform(m1: float, m2: float):
    """IMRPhenomD inspiral-merger-ringdown, at Stage 1's sample rate."""
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOWER,
    )
    return hp


def place(hp, n: int, merger_pos: float) -> TimeSeries:
    """Put the waveform into an n-sample window with its merger at `merger_pos` (frac of window).

    ⚠️ THIS IS THE SUBTLE ONE. At 10-50 Msun from 30 Hz, IMRPhenomD is LONGER than our
    1 s window -- often several seconds. So the window holds only the tail of the chirp,
    and the early inspiral is cut off.

    The naive `hp.resize(n)` is a trap: it keeps the FIRST n samples (the quiet early
    inspiral) and discards the merger -- the opposite of what the window actually sees.
    Do not use it here.

    Align on the waveform's amplitude PEAK (the merger), and let the early inspiral fall
    off the front of the window if it doesn't fit. What's left is what the detector sees.
    """
    wf = hp.numpy()
    x = np.zeros(n)
    merger_idx = int(merger_pos * n)
    start = merger_idx - int(np.argmax(np.abs(wf)))  # align peak -> merger_idx
    lo, hi = max(0, start), min(n, start + len(wf))
    if hi > lo:
        x[lo:hi] = wf[lo - start : hi - start]
    return TimeSeries(x, delta_t=1.0 / SAMPLE_RATE)


def inject(noise, placed: TimeSeries, target_snr: float):
    """Add the placed waveform into `noise`, rescaled to hit `target_snr`.

    sigma() is computed on the PLACED snippet, not the full waveform -- so `target_snr`
    means "the SNR of the signal actually present in this segment", which is the only
    definition the CNN (and the money plot's x-axis) can be held to.
    """
    s = sigma(placed, psd=PSD, low_frequency_cutoff=F_LOWER)
    return noise + placed * (target_snr / s)


def recover_snr(seg, placed: TimeSeries) -> float:
    """Independently measure the peak matched-filter SNR — the ground truth for the check.

    ⚠️ Do NOT crop the edges of the SNR series here. The template is placed at the SAME
    alignment as the signal in the data, so the filter peaks at LAG ZERO -- index 0.
    An innocent-looking `[edge:-edge]` crop deletes precisely the peak we are measuring,
    and you get a flat ~3.6 back for every SNR: the noise background, reported with total
    confidence. (Yes, this bit us.)
    """
    snr = abs(matched_filter(placed, seg, psd=PSD, low_frequency_cutoff=F_LOWER)).numpy()
    return float(snr.max())


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    targets = [4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
    n_trials = 40

    print(f"injecting {n_trials} waveforms per SNR, masses U{MASS_RANGE} Msun\n")
    print(f"{'target':>7} {'recovered (mean+-sd)':>24} {'bias':>8}")

    rows = []
    for target in targets:
        rec = []
        for i in range(n_trials):
            m1, m2 = rng.uniform(*MASS_RANGE, size=2)
            noise = generate_noise(SEG_LEN, seed=int(rng.integers(1 << 30)))
            hp = make_waveform(m1, m2)
            # Randomised merger position — the footgun fix. Fixed placement would teach
            # the CNN a sample index instead of a chirp shape.
            placed = place(hp, len(noise), rng.uniform(0.7, 0.95))
            seg = inject(noise, placed, target)
            rec.append(recover_snr(seg, placed))
        rec = np.array(rec)
        rows.append(rec)
        print(f"{target:7.1f} {rec.mean():14.2f} +- {rec.std():5.2f} {rec.mean()/target:9.3f}x")

    means = np.array([r.mean() for r in rows])
    # Recovered SNR is biased slightly HIGH at low target SNR: the filter peaks on a
    # noise fluctuation near the signal. That is physics, not a bug -- it is exactly
    # why low-SNR detection is hard. Check the bias is small and shrinks with SNR.
    bias = means / np.array(targets)
    ok = bool(np.all(np.abs(bias - 1.0) < 0.35) and abs(bias[-1] - 1.0) < 0.1)
    print(f"\n  {'PASS' if ok else 'FAIL'}: recovered SNR tracks requested SNR"
          f" (bias {bias.min():.2f}x-{bias.max():.2f}x, ->1 at high SNR)")

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(13, 5))

    ax.plot([0, 22], [0, 22], "k--", lw=1, label="ideal (recovered = requested)")
    for t, r in zip(targets, rows):
        ax.scatter([t] * len(r), r, s=8, alpha=0.35, color="tab:blue")
    ax.errorbar(targets, means, yerr=[r.std() for r in rows], fmt="o-", color="tab:red",
                capsize=4, lw=2, label="recovered (mean ± sd)")
    ax.set_xlabel("requested injected SNR")
    ax.set_ylabel("recovered matched-filter SNR")
    ax.set_title("SNR scaling is calibrated\n(high bias at low SNR is physics: the filter peaks on noise)")
    ax.legend()
    ax.grid(alpha=0.3)

    # A look at what the network will actually see.
    hp = make_waveform(36.0, 29.0)  # GW150914-ish
    noise = generate_noise(SEG_LEN, seed=7)
    placed = place(hp, len(noise), 0.85)
    for snr, c in [(20.0, "tab:red"), (8.0, "tab:blue")]:
        seg = inject(noise, placed, snr)
        # Whiten with the KNOWN design PSD — never one estimated from this segment.
        # That estimate is THE leak: it would carry a fingerprint of the injection.
        asd = np.sqrt(PSD.numpy() * SAMPLE_RATE / 2)
        w = np.divide(seg.to_frequencyseries().numpy(), asd,
                      out=np.zeros_like(seg.to_frequencyseries().numpy()), where=asd > 0)
        wt = np.fft.irfft(w, n=len(seg))
        axw.plot(np.arange(len(wt)) / SAMPLE_RATE, wt, lw=0.9, color=c, alpha=0.85,
                 label=f"injected SNR {snr:.0f}")
    axw.set_xlabel("time [s]")
    axw.set_ylabel("whitened strain [σ]")
    axw.set_title("36+29 M☉ chirp in design noise\n(whitened with the KNOWN psd — no leak)")
    axw.legend()
    axw.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT / "stage1_injection_check.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

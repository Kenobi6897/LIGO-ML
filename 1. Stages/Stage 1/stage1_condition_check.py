"""Stage 1, Step 3 — the check that `condition()` is signal-blind.

Three questions, in order of how badly a wrong answer would hurt:

  1. Is the whitened noise actually WHITE? (If not, the CNN can hear the noise colour and
     will happily use it.)
  2. Is the scale sane -- conditioned noise ~ unit variance, from ONE global constant?
  3. **Is the operator linear?**  condition(noise + signal) == condition(noise) + condition(signal)

Question 3 is the important one. A whitener that estimates its PSD from the segment in
front of it is not linear -- it *cannot* be, because the operator itself depends on the
data. That non-linearity is precisely the leak: it stamps a signal-dependent fingerprint
on the whitened noise floor, and the CNN reads the fingerprint instead of the chirp,
scoring ~99% AUC while having learned nothing. See [[Stage 1#Footguns]].

So we test the identity to machine precision, and we run the same test on
`condition_leaky()` to watch it fail.

Run (from WSL2):
    source ~/venvs/ligo/bin/activate
    python "4. Code/stage1/stage1_condition_check.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycbc.filter import sigma
from pycbc.psd import welch
from pycbc.types import TimeSeries

from stage1_condition import (
    BUFFER_LEN,
    BUFFER_N,
    CROP,
    CROP_N,
    condition,
    condition_leaky,
)
from stage1_injection_check import F_LOWER, MASS_RANGE, PSD, make_waveform
from stage1_noise_check import BAND, SAMPLE_RATE, generate_noise

OUT = Path(__file__).parent / "outputs"
DT = 1.0 / SAMPLE_RATE


def place_in_buffer(hp, merger_pos: float) -> TimeSeries:
    """Put the waveform in the 4 s buffer with its merger at `merger_pos` of the CROP.

    Same peak-alignment logic as Step 2 (`hp.resize()` would keep the quiet early inspiral
    and throw the merger away) — but positioned relative to the 1 s window the CNN will
    actually see, not the buffer.
    """
    wf = hp.numpy()
    x = np.zeros(BUFFER_N)
    merger_idx = CROP.start + int(merger_pos * CROP_N)
    start = merger_idx - int(np.argmax(np.abs(wf)))
    lo, hi = max(0, start), min(BUFFER_N, start + len(wf))
    x[lo:hi] = wf[lo - start : hi - start]
    return TimeSeries(x, delta_t=DT)


def scale_to_snr(placed: TimeSeries, target_snr: float) -> TimeSeries:
    """Scale so the signal *inside the crop* has SNR `target_snr`.

    The SNR is defined on what survives the crop, because that is all the CNN is ever
    shown. Signal power that falls outside the crop is discarded by the pipeline, so it
    must not be counted in the label.
    """
    snippet = TimeSeries(placed.numpy()[CROP], delta_t=DT)
    return placed * (target_snr / sigma(snippet, psd=PSD, low_frequency_cutoff=F_LOWER))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    checks = []

    # --- 1. is conditioned noise white, and unit-scale? ------------------------
    crops = np.concatenate(
        [condition(generate_noise(BUFFER_LEN, seed=1000 + i)) for i in range(48)]
    )
    raw = generate_noise(BUFFER_LEN * 48, seed=77)

    p_cond = welch(TimeSeries(crops, delta_t=DT), seg_len=CROP_N, seg_stride=CROP_N // 2)
    p_raw = welch(raw, seg_len=CROP_N, seg_stride=CROP_N // 2)

    def tilt(p) -> float:
        """Mean power at the band's bottom (30-40 Hz) over its top (200-350 Hz).

        White noise -> 1. Design noise -> ~4.3: the aLIGO curve is only mildly tilted
        ACROSS 30-350 Hz -- its violent part (the seismic wall, orders of magnitude) sits
        BELOW the band, where the bandpass throws it away regardless. So this is a modest
        number by nature, and a 4.3x tilt is still plenty for a CNN to exploit if we leave
        it in: it makes low-frequency noise excursions look systematically louder than
        high-frequency ones.
        """
        f = p.sample_frequencies.numpy()
        lo = p.numpy()[(f >= 30) & (f <= 40)].mean()
        hi = p.numpy()[(f >= 200) & (f <= 350)].mean()
        return float(lo / hi)

    tilt_raw, tilt_cond = tilt(p_raw), tilt(p_cond)
    std = float(crops.std())

    print("1. whiteness  (mean power 30-40 Hz / 200-350 Hz; white noise -> 1.0)")
    print(f"     raw aLIGO noise : {tilt_raw:9.2f}x   (the design curve's in-band tilt)")
    print(f"     conditioned     : {tilt_cond:9.2f}x   (want ~1 — colour removed)")
    checks.append(("noise is whitened flat in band", 0.8 <= tilt_cond <= 1.25))

    print(f"\n2. scale      conditioned noise std = {std:.3f}   (want ~1, from ONE global constant)")
    checks.append(("conditioned noise has unit scale", 0.9 <= std <= 1.1))

    # --- 3. the leak test ------------------------------------------------------
    #
    # Stated so that it only ever feeds the operators inputs the real pipeline sees
    # (noise, and noise+signal — never a pure signal, which no detector produces and for
    # which a per-segment whitener is undefined anyway; whitening one by its own spectrum
    # would be a strawman, and the demo has to survive a hostile reading).
    #
    # THE PROPERTY: a signal-blind operator applies the same fixed filter to every
    # segment, so the CHANGE a signal makes cannot depend on the noise it landed in:
    #
    #     C(n1 + h) - C(n1)  ==  C(n2 + h) - C(n2)     for any noise n1, n2
    #
    # If the whitener is built from the segment itself, the two differ — and the
    # difference is a noise-dependent, signal-dependent stamp. That stamp is the leak.
    leak_scale = float(
        np.mean([condition_leaky(generate_noise(BUFFER_LEN, seed=800_000 + i)).std() for i in range(8)])
    )

    print("\n3. leak test  the response to a signal must not depend on the noise it lands in:")
    print("              || [C(n1+h) - C(n1)] - [C(n2+h) - C(n2)] ||  /  || C(n1+h) - C(n1) ||")
    print(f"\n   {'SNR':>5} {'condition()':>16} {'condition_leaky()':>20}")

    resid_ok, resid_leak, panels = [], [], []
    for snr in (8.0, 20.0):
        for k in range(5):
            m1, m2 = rng.uniform(*MASS_RANGE, size=2)
            h = scale_to_snr(place_in_buffer(make_waveform(m1, m2), rng.uniform(0.7, 0.95)), snr)
            n1 = generate_noise(BUFFER_LEN, seed=int(rng.integers(1 << 30)))
            n2 = generate_noise(BUFFER_LEN, seed=int(rng.integers(1 << 30)))

            for fn, bucket, scale in (
                (condition, resid_ok, 1.0),
                (condition_leaky, resid_leak, leak_scale),
            ):
                # Each operator is normalised by its OWN pure-noise std — a global
                # constant, so neither is handicapped by the other's scale convention.
                d1 = (fn(n1 + h) - fn(n1)) / scale
                d2 = (fn(n2 + h) - fn(n2)) / scale
                bucket.append(float(np.linalg.norm(d1 - d2) / np.linalg.norm(d1)))
                if snr == 20.0 and k == 0:
                    panels.append((d1, d2))

        print(f"   {snr:5.0f} {np.mean(resid_ok[-5:]):16.2e} {np.mean(resid_leak[-5:]):20.2e}")

    worst_ok, best_leak = max(resid_ok), min(resid_leak)
    print(f"\n     condition()       : disagreement <= {worst_ok:.1e}  — same signal, same response,")
    print("                           whatever noise it landed in. Signal-blind.")
    print(f"     condition_leaky() : disagreement >= {best_leak:.1e}  — the SAME signal is conditioned")
    print("                           DIFFERENTLY depending on its noise. The operator read the data.")
    print("\n     ^ that disagreement IS the leak: a signal-dependent stamp on the noise floor,")
    print("       and it is far easier for a CNN to learn than a chirp.")
    checks.append(("condition() is signal-blind", worst_ok < 1e-5))
    checks.append(("condition_leaky() is NOT (demo works)", best_leak > 1e-2))

    ok = all(p for _, p in checks)
    print()
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"\n  {'PASS' if ok else 'FAIL'}: condition() is safe to build the dataset with")

    # --- figure ----------------------------------------------------------------
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.8))

    f = p_cond.sample_frequencies.numpy()
    m = (f >= 10) & (f <= 1000)
    a1.loglog(f[m], np.sqrt(p_raw.numpy()[m] / p_raw.numpy()[m].max()), lw=0.7,
              color="tab:gray", label="raw noise (coloured)")
    a1.loglog(f[m], np.sqrt(p_cond.numpy()[m]), lw=0.7, color="tab:blue",
              label="conditioned (whitened + banded)")
    a1.axvspan(*BAND, color="tab:orange", alpha=0.12, label="30-350 Hz band")
    a1.set(xlabel="frequency [Hz]", ylabel="ASD [arb.]",
           title=f"Whitening removes the colour\n(in-band tilt {tilt_raw:.1f}x → {tilt_cond:.2f}x)")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, which="both")

    # what the CNN is actually handed
    t = np.arange(CROP_N) / SAMPLE_RATE
    n = generate_noise(BUFFER_LEN, seed=7)
    h = scale_to_snr(place_in_buffer(make_waveform(36.0, 29.0), 0.85), 12.0)
    a2.plot(t, condition(n), lw=0.7, color="tab:gray", alpha=0.8, label="negative (noise)")
    a2.plot(t, condition(n + h), lw=0.9, color="tab:red", alpha=0.9, label="positive (SNR 12)")
    a2.set(xlabel="time in segment [s]", ylabel="conditioned strain [σ]",
           title="What the CNN eats: 2048 samples\n36+29 M☉, merger at 0.85 s")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    # The leak, drawn: the SAME signal (SNR 20), recovered from two different noise
    # realisations. C(n+h) - C(n) should be the signal and nothing else.
    (ok1, ok2), (lk1, lk2) = panels[0], panels[1]
    a3.plot(t, ok1, lw=1.4, color="tab:green", label="condition(), noise A")
    a3.plot(t, ok2, lw=0.8, color="k", ls="--", label="condition(), noise B  (identical)")
    a3.plot(t, lk1 - lk2, lw=0.8, color="tab:purple", alpha=0.9,
            label="condition_leaky(): A − B  (should be 0)")
    a3.set(xlabel="time in segment [s]", ylabel="C(n+h) − C(n)   [σ]",
           title="THE LEAK, drawn: one signal, two noises\nblind → identical.  leaky → a residue tracking the chirp")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT / "3_condition_check.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

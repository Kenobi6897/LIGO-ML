"""Stage 2, Step 2's check — is the real-noise conditioner leak-free, and does it work?

Three checks that can fail, one measurement that is the stage's thesis:

  1. SIGNAL-BLINDNESS (the one that matters). Within a block the conditioner is one
     fixed linear operator, so the change a signal makes cannot depend on the noise it
     landed in:  C(n1 + h) - C(n1) == C(n2 + h) - C(n2), to machine precision. This is
     Stage 1's leak test verbatim, now with a *measured* PSD in the operator — the
     property survives because the PSD comes from the previous block, never from n1/n2.
  2. WHITENING QUALITY. Conditioned real noise must come out flat across 30-350 Hz.
     Real spectra have lines and drift, and the PSD is 512 s stale by construction —
     flat-ish with residual line structure is expected; a tilt is a bug.
  3. THE GLOBAL SCALE TRANSFERS. norm() is measured on the earliest data blocks; the
     std of conditioned noise in the LATEST blocks must still be ~1. If it isn't, the
     detector drifted more than a global constant can absorb, and the split boundary
     becomes a feature.

  4. NON-GAUSSIANITY, MEASURED NOT ASSERTED. Excess kurtosis and tail rates
     (P(|x|>4), P(|x|>5)) of conditioned real noise, against both the Gaussian
     prediction and Stage 1's conditioned simulated noise. Whatever the CNN's false
     alarms turn out to be, this is where they come from — quantified before any model
     has touched the data.

Run (from WSL2, after stage2_fetch.py has at least ~20 blocks done):
    python stage2_condition_check.py [--path ~/ligo-data/stage2_smoke.h5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_condition import condition as sim_condition
from stage1_dataset import place_in_buffer
from stage1_injection_check import make_waveform
from stage1_noise_check import BAND, SAMPLE_RATE, generate_noise

import stage2_condition as s2c
from stage2_condition import (
    N_BUFFERS_PER_BLOCK,
    N_NORM_BLOCKS,
    _condition_unnormed,
    condition_real,
    data_rows,
    norm,
    raw_buffer,
    whitening_psd,
)

OUT = Path(__file__).parent / "outputs"
GAUSS_TAILS = {4: 6.334e-5, 5: 5.733e-7}  # 2*(1-Phi(k))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 2 — check the conditioner")
    ap.add_argument("--path", type=Path, default=None,
                    help="override the strain file (e.g. the smoke file)")
    args = ap.parse_args()
    if args.path is not None:
        s2c.STRAIN_H5 = args.path  # noqa — test hook; must be set before first read
        s2c.strain_file(args.path)

    rows = data_rows()
    print(f"{len(rows)} complete data blocks on disk; norm from the first "
          f"{min(N_NORM_BLOCKS, len(rows))}, quality checked on the last {min(8, len(rows))}")
    results: list[tuple[str, bool, str]] = []

    # --- 1. signal-blindness: same signal, two noises, identical response -----------
    row = int(rows[len(rows) // 2])
    psd = whitening_psd(row)
    n1, n2 = raw_buffer(row, 40), raw_buffer(row, 400)
    placed = place_in_buffer(make_waveform(36.0, 29.0), 0.85)
    placed *= float(n1.numpy().std() / placed.numpy().std())  # signal ~ noise loud

    d1 = _condition_unnormed(n1 + placed, psd) - _condition_unnormed(n1, psd)
    d2 = _condition_unnormed(n2 + placed, psd) - _condition_unnormed(n2, psd)
    rel = float(np.max(np.abs(d1 - d2)) / np.max(np.abs(d1)))
    results.append(("conditioner is signal-blind (linearity to machine precision)",
                    rel < 1e-9, f"relative disagreement {rel:.1e}"))
    print(f"  signal response disagreement across noises: {rel:.1e}  (machine precision ~1e-15)")

    # --- 2 + 3. whitening quality + scale transfer, on the LATEST blocks -------------
    far = [int(r) for r in rows[-8:]]
    crops, stds = [], []
    for r in far:
        xs = np.stack([condition_real(raw_buffer(r, off), r)
                       for off in range(0, N_BUFFERS_PER_BLOCK, 8)])
        crops.append(xs)
        stds.append(float(xs.std()))
    real = np.concatenate([c.ravel() for c in crops])
    stds = np.array(stds)

    from scipy.signal import welch as scipy_welch

    freqs, psd_c = scipy_welch(np.concatenate([c.ravel() for c in crops]),
                               fs=SAMPLE_RATE, nperseg=2048, average="median")
    in_band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f_b, p_b = freqs[in_band], psd_c[in_band]
    thirds = np.array_split(np.arange(len(p_b)), 3)
    tilt = float(np.mean(p_b[thirds[2]]) / np.mean(p_b[thirds[0]]))
    results.append(("conditioned real noise is flat in band",
                    0.6 < tilt < 1.6, f"high/low-third tilt {tilt:.2f}"))
    print(f"  in-band tilt (upper third / lower third): {tilt:.2f}  (want ~1)")

    drift = float(stds.mean())
    results.append((f"global scale transfers to the last blocks (std {drift:.3f})",
                    0.75 < drift < 1.30,
                    f"per-block std {stds.min():.2f}-{stds.max():.2f}"))
    print(f"  std of conditioned noise, last blocks: mean {drift:.3f}, "
          f"range {stds.min():.3f}-{stds.max():.3f}  (norm came from the FIRST blocks)")

    # --- 4. non-Gaussianity, against conditioned SIMULATED noise ---------------------
    sim = np.concatenate([sim_condition(generate_noise(4.0, seed=700_000 + i)).ravel()
                          for i in range(256)])

    def stats(x):
        z = (x - x.mean()) / x.std()
        kurt = float(np.mean(z**4) - 3.0)
        return kurt, float(np.mean(np.abs(z) > 4)), float(np.mean(np.abs(z) > 5))

    rk, r4, r5 = stats(real)
    sk, s4, s5 = stats(sim)
    print("\n  non-Gaussianity (the stage's thesis, at the data level):")
    print(f"    {'':>24} {'excess kurtosis':>16} {'P(|x|>4)':>12} {'P(|x|>5)':>12}")
    print(f"    {'Gaussian prediction':>24} {0.0:>16.3f} {GAUSS_TAILS[4]:>12.2e} {GAUSS_TAILS[5]:>12.2e}")
    print(f"    {'conditioned SIMULATED':>24} {sk:>16.3f} {s4:>12.2e} {s5:>12.2e}")
    print(f"    {'conditioned REAL O3':>24} {rk:>16.3f} {r4:>12.2e} {r5:>12.2e}")

    # --- figure ----------------------------------------------------------------------
    fig, (axp, axh) = plt.subplots(1, 2, figsize=(13.5, 5))

    axp.semilogy(f_b, p_b, lw=0.8, color="tab:blue", label="conditioned real O3 noise")
    axp.axhline(np.median(p_b), color="k", lw=1.2, ls="--", label="in-band median")
    axp.set_xlabel("frequency [Hz]")
    axp.set_ylabel("PSD of conditioned noise")
    axp.set_title(f"Whitened with the previous block's PSD — tilt {tilt:.2f}\n"
                  "(residual lines are real detector features, not bugs)")
    axp.legend()
    axp.grid(alpha=0.3, which="both")

    bins = np.linspace(-8, 8, 161)
    axh.hist((real - real.mean()) / real.std(), bins=bins, density=True, log=True,
             histtype="step", lw=1.4, color="tab:red",
             label=f"real O3, kurt {rk:+.2f}")
    axh.hist((sim - sim.mean()) / sim.std(), bins=bins, density=True, log=True,
             histtype="step", lw=1.4, color="tab:blue",
             label=f"simulated, kurt {sk:+.2f}")
    zz = 0.5 * (bins[1:] + bins[:-1])
    axh.plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="N(0,1)")
    axh.set_ylim(1e-8, 1)
    axh.set_xlabel("conditioned amplitude [σ]")
    axh.set_ylabel("density (log)")
    axh.set_title("Where Stage 2's false alarms will come from:\n"
                  "the tails real noise has and Gaussian noise doesn't")
    axh.legend(loc="upper right", fontsize=8)
    axh.grid(alpha=0.3)

    print()
    all_ok = True
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + ("" if ok else f" — {detail}"))
        all_ok &= ok

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "2_condition_check.png"
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

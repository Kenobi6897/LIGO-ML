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
    # Per-CROP stds, and a MEDIAN-based gate. The first run of this check used the
    # pooled std and FAILED at 5.0 — which turned out to be two real glitches (one
    # 263-sigma-std crop at GPS 1238343793, one at 32) sitting in otherwise perfectly
    # calibrated blocks (median crop std 0.993 across all 197). Real O3 data contains
    # monsters; they are Stage 2's subject matter, not a calibration error. So the
    # CALIBRATION gate is robust to them, and the monsters are REPORTED separately
    # below as a measured glitch rate.
    far = [int(r) for r in rows[-8:]]
    crops, crop_stds = [], []
    for r in far:
        xs = np.stack([condition_real(raw_buffer(r, off), r)
                       for off in range(0, N_BUFFERS_PER_BLOCK, 8)])
        crops.append(xs)
        crop_stds.append(xs.std(axis=1))
    real = np.concatenate([c.ravel() for c in crops])
    crop_stds = np.concatenate(crop_stds)

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

    drift = float(np.median(crop_stds))
    results.append((f"global scale transfers to the last blocks (median crop std {drift:.3f})",
                    0.75 < drift < 1.30,
                    f"5-95% {np.percentile(crop_stds,5):.2f}-{np.percentile(crop_stds,95):.2f}"))
    print(f"  crop std, last blocks: median {drift:.3f} "
          f"(5-95% {np.percentile(crop_stds,5):.3f}-{np.percentile(crop_stds,95):.3f}, "
          f"max {crop_stds.max():.1f})  (norm came from the FIRST blocks)")

    # The monsters, measured not hidden: how often does a 1 s crop of conditioned real
    # noise carry a transient loud enough to double its std? This is the glitch rate
    # the Stage 1 model has never met, per hour, from the sampled crops.
    n_loud = int((crop_stds > 2.0).sum())
    hours_sampled = len(crop_stds) / 3600.0
    print(f"  glitchy crops (std > 2): {n_loud}/{len(crop_stds)} sampled "
          f"= {n_loud / hours_sampled:.0f}/h of real noise, loudest std {crop_stds.max():.0f}")

    # --- 4. non-Gaussianity, against conditioned SIMULATED noise ---------------------
    sim = np.concatenate([sim_condition(generate_noise(4.0, seed=700_000 + i)).ravel()
                          for i in range(256)])

    def stats(x):
        z = (x - x.mean()) / x.std()
        kurt = float(np.mean(z**4) - 3.0)
        return kurt, float(np.mean(np.abs(z) > 4)), float(np.mean(np.abs(z) > 5))

    # "Bulk" excludes the glitchy crops (std > 2), so the two rows separate the two
    # distinct non-Gaussian phenomena: fat tails everywhere, and rare monsters.
    bulk = np.concatenate([c[c.std(axis=1) <= 2.0].ravel() for c in crops])
    rk, r4, r5 = stats(real)
    bk, b4, b5 = stats(bulk)
    sk, s4, s5 = stats(sim)
    print("\n  non-Gaussianity (the stage's thesis, at the data level):")
    print(f"    {'':>26} {'excess kurtosis':>16} {'P(|x|>4)':>12} {'P(|x|>5)':>12}")
    print(f"    {'Gaussian prediction':>26} {0.0:>16.3f} {GAUSS_TAILS[4]:>12.2e} {GAUSS_TAILS[5]:>12.2e}")
    print(f"    {'conditioned SIMULATED':>26} {sk:>16.3f} {s4:>12.2e} {s5:>12.2e}")
    print(f"    {'REAL O3, bulk (no glitch)':>26} {bk:>16.3f} {b4:>12.2e} {b5:>12.2e}")
    print(f"    {'REAL O3, everything':>26} {rk:>16.3f} {r4:>12.2e} {r5:>12.2e}")

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
    axh.hist((bulk - bulk.mean()) / bulk.std(), bins=bins, density=True, log=True,
             histtype="step", lw=1.4, color="tab:red",
             label=f"real O3 bulk, kurt {bk:+.2f} (glitches off-scale: worst crop std "
                   f"{crop_stds.max():.0f})")
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

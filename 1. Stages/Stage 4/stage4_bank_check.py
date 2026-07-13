"""Stage 4, Step 1's check — does the bank actually cover the mass plane?

The claims under test, each of which would silently corrupt the benchmark:
  - EFFECTUALNESS: a random injection's best match against the bank (its fitting
    factor) must sit at or above the minimal match the bank promised. If not, arm E's
    efficiency is capped by bank holes, and "MF loses to the CNN" could mean "the bank
    was thin", not "the theorem cracked".
  - REFERENCE-FILTER ROBUSTNESS: the bank was placed under ONE train block's whitening
    filter; scoring re-conditions per block. If coverage collapsed under other blocks'
    filters, the reference choice would be load-bearing — it must not be.
  - DETERMINISM: same seed, same bank, bit-exact masses — reruns of the benchmark must
    be reruns, not re-rolls.
  - The match machinery itself: match(T, T) == 1 to machine precision, and a template
    time/phase-shifted must still match ~1 (that's the freedom the statistic maxes over).

Run (from WSL2, after stage4_bank.py, ~3 min):
    python stage4_bank_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_injection_check import MASS_RANGE
from stage1_noise_check import BAND, SAMPLE_RATE

import stage2_condition as s2c

from stage4_bank import (
    BANK_SEED, DEFAULT_OUT, FFT_N, MIN_MATCH,
    build_bank, corr_series, crop_template, fdom, match_many, self_norm,
)

OUT = Path(__file__).parent / "outputs"
N_HELDOUT = 250
N_ROBUST = 50
FF_FLOOR = 0.965   # stochastic banks guarantee MM statistically, not pointwise


def fitting_factors(fbank: np.ndarray, draws: np.ndarray, row: int) -> np.ndarray:
    ffs = []
    for m1, m2 in draws:
        ft = fdom(crop_template(float(m1), float(m2), row))
        ffs.append(float(match_many(fbank, ft / self_norm(ft)).max()))
    return np.array(ffs)


def main() -> int:
    results = []
    with h5py.File(DEFAULT_OUT, "r") as f:
        m1, m2 = f["m1"][:], f["m2"][:]
        temps = f["templates"][:]
        ref_row = int(f.attrs["ref_block"])
        seed = int(f.attrs["seed"])
    n = len(m1)
    print(f"{DEFAULT_OUT}: {n} templates (seed {seed}, ref block {ref_row})\n")
    fbank = np.stack([fdom(t) for t in temps])

    # --- the machinery: identity, shift/phase freedom, unit norms -----------------------
    norms = np.array([self_norm(fb) for fb in fbank])
    results.append(("templates stored unit-norm (max dev "
                    f"{np.abs(norms - 1).max():.2e})", bool(np.abs(norms - 1).max() < 1e-9),
                    "normalization broken"))
    self_m = float(match_many(fbank[:1], fbank[0]).max())
    results.append((f"match(T, T) == 1 (got {self_m:.12f})", abs(self_m - 1) < 1e-9, "identity broken"))
    shifted = np.roll(temps[0], 137)  # a lag the statistic must absorb
    fs = fdom(shifted)
    m_shift = float(match_many(fbank[:1], fs / self_norm(fs)).max())
    results.append((f"match(T, T shifted 137 samples) ~ 1 (got {m_shift:.6f})",
                    m_shift > 0.999, "lag maximization broken"))
    results.append(("all templates finite", bool(np.isfinite(temps).all()), "NaN/inf"))

    # in-band: spectral peak of every template inside BAND
    freqs = np.fft.rfftfreq(FFT_N, 1.0 / SAMPLE_RATE)
    peaks = freqs[np.abs(fbank).argmax(axis=1)]
    inband = (peaks >= BAND[0]) & (peaks <= BAND[1])
    results.append((f"every template peaks in-band ({peaks.min():.0f}-{peaks.max():.0f} Hz)",
                    bool(inband.all()), f"{int((~inband).sum())} outside"))
    results.append((f"bank size sane ({n})", 10 <= n <= 500, "degenerate or runaway"))

    # --- effectualness on held-out draws -------------------------------------------------
    rng = np.random.default_rng(seed + 1)
    draws = np.sort(rng.uniform(*MASS_RANGE, size=(N_HELDOUT, 2)), axis=1)[:, ::-1]
    ff = fitting_factors(fbank, draws, ref_row)
    results.append((f"effectualness: median FF {np.median(ff):.4f} >= {MIN_MATCH}",
                    float(np.median(ff)) >= MIN_MATCH, "bank has holes"))
    frac = float((ff >= FF_FLOOR).mean())
    results.append((f"effectualness: {frac:.0%} of draws >= {FF_FLOOR} (min {ff.min():.4f})",
                    frac >= 0.95, "coverage tail too thin"))

    # --- robustness: does coverage survive OTHER train blocks' filters? -----------------
    rows = s2c.data_rows()
    n_train = int(0.8 * len(rows))  # stage2's time-ordered split: first 80% is train
    probe_rows = [int(rows[int(p * n_train)]) for p in (0.1, 0.4, 0.7)]
    meds = []
    for r in probe_rows:
        fb_r = []
        for a, b in zip(m1, m2):
            ft = fdom(crop_template(float(a), float(b), r))
            fb_r.append(ft / self_norm(ft))
        ff_r = fitting_factors(np.stack(fb_r), draws[:N_ROBUST], r)
        meds.append(float(np.median(ff_r)))
        print(f"  block {r}: median FF {meds[-1]:.4f} over {N_ROBUST} draws")
    results.append((f"coverage robust to the whitening filter (worst median "
                    f"{min(meds):.4f} across 3 other train blocks)", min(meds) >= FF_FLOOR,
                    "reference block is load-bearing"))

    # --- determinism ---------------------------------------------------------------------
    masses2, _, _ = build_bank(seed, ref_row, verbose=False)
    same = len(masses2) == n and np.array_equal(masses2[:, 0], m1) and np.array_equal(masses2[:, 1], m2)
    results.append(("deterministic rebuild (same seed, bit-exact masses)", same, "re-roll, not rerun"))

    print()
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {name}" + ("" if passed else f" — {detail}"))

    # --- figure: the bank on the mass plane + FF histogram --------------------------------
    fig, (axm, axf) = plt.subplots(1, 2, figsize=(13, 5))
    sc = axm.scatter(m1, m2, c=np.arange(n), cmap="viridis", s=28)
    axm.plot(MASS_RANGE, MASS_RANGE, "k:", lw=1, alpha=0.5)
    axm.set_xlabel("m1 [Msun]"), axm.set_ylabel("m2 [Msun]")
    axm.set_title(f"{n} templates, MM {MIN_MATCH} on crop-truncated waveforms")
    fig.colorbar(sc, ax=axm, label="acceptance order")
    axf.hist(ff, bins=40, color="tab:blue", alpha=0.8)
    axf.axvline(MIN_MATCH, color="k", ls="--", lw=1, label=f"MM {MIN_MATCH}")
    axf.axvline(float(np.median(ff)), color="tab:red", lw=1.5,
                label=f"median {np.median(ff):.4f}")
    axf.set_xlabel("fitting factor of held-out draws"), axf.set_ylabel("count")
    axf.set_title(f"Effectualness ({N_HELDOUT} random mass pairs)")
    axf.legend(fontsize=8), axf.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "1_bank_check.png", dpi=130)
    print(f"\n  plot -> {OUT / '1_bank_check.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

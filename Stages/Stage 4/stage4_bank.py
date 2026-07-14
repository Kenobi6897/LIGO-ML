"""Stage 4, Step 1 — the template bank: stochastic placement at minimal match 0.97.

Templates are built through THE IDENTICAL PIPELINE the injections went through —
`make_waveform` -> `place_in_buffer` -> the block's conditioning — so each one is the
crop-truncated, whitened, bandpassed 1 s the data actually contains. A 10+10 Msun
waveform runs ~3 s from 30 Hz; the crop keeps the last ~1 s, and a full-length template
would claim SNR the data never held (the truncation footgun in [[Stage 4]]).

Matches use the standard phase-maximized overlap, computed in the frequency domain on
zero-padded crops: z(t) = IFFT of the one-sided cross-spectrum (negative frequencies
zeroed), match = max_t |z_ab| / sqrt(z_aa[0] * z_bb[0]). All FFT constants cancel by
construction; the check proves match(T, T) == 1 to machine precision.

The reference whitening filter is the EARLIEST train data block's — the bank never
touches val/test noise. The bank's job is geometry (covering the mass plane), not
per-block calibration: stage4_mf.py re-conditions every template with each scored row's
own block filter. The check measures how little the reference choice matters.

Run (from WSL2, ~2 min):
    python stage4_bank.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_condition import CROP_N
from stage1_dataset import place_in_buffer
from stage1_injection_check import MASS_RANGE, make_waveform

import stage2_condition as s2c

DEFAULT_OUT = Path.home() / "ligo-data" / "stage4_bank.h5"
MIN_MATCH = 0.97
BANK_SEED = 20260714
REF_MERGER_POS = 0.85     # centre of the injection range; lags cover the rest
STOP_AFTER = 500          # consecutive rejections = the plane is covered
FFT_N = 2 * CROP_N        # 4096 >= 2048 + 2048: all linear lags, no wraparound


def ref_block() -> int:
    """The earliest data block — train by construction (time-ordered splits)."""
    return int(s2c.data_rows()[0])


def crop_template(m1: float, m2: float, row: int) -> np.ndarray:
    """One conditioned crop containing ONLY the (truncated) waveform — the exact
    signal content an injection with these masses adds to a crop of block `row`."""
    buf = place_in_buffer(make_waveform(m1, m2), REF_MERGER_POS)
    return s2c.condition_real(buf, row).astype(np.float64)


def fdom(x: np.ndarray) -> np.ndarray:
    """One-sided spectrum of a zero-padded crop."""
    return np.fft.rfft(x, FFT_N)


def corr_series(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Complex correlation over all lags with negative frequencies zeroed — |z(t)| is
    the phase-maximized overlap at lag t. fa/fb may be (..., FFT_N//2+1) stacks."""
    prod = fa * np.conj(fb)
    full = np.zeros((*prod.shape[:-1], FFT_N), dtype=complex)
    full[..., : prod.shape[-1]] = prod
    return np.fft.ifft(full, axis=-1)


def self_norm(fa: np.ndarray) -> float:
    """sqrt(z_aa[0]) — the normalization that makes match(a, a) == 1 exactly."""
    return float(np.sqrt(np.abs(corr_series(fa, fa)[..., 0])))


def match_many(fbank: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """Best phase/time-maximized match of candidate `ft` against every bank row.
    Both sides must already be self-normalized."""
    return np.abs(corr_series(fbank, ft[None, :])).max(axis=-1)


def build_bank(seed: int, row: int, verbose: bool = True):
    rng = np.random.default_rng(seed)
    masses: list[tuple[float, float]] = []
    temps: list[np.ndarray] = []
    fbank: list[np.ndarray] = []
    tried = rejects = 0
    t0 = time.time()
    while rejects < STOP_AFTER:
        m2, m1 = np.sort(rng.uniform(*MASS_RANGE, size=2))  # m1 >= m2
        t = crop_template(float(m1), float(m2), row)
        ft = fdom(t)
        n = self_norm(ft)
        if not (np.isfinite(n) and n > 0):
            raise RuntimeError(f"degenerate template at m1={m1:.2f}, m2={m2:.2f}")
        ft = ft / n
        tried += 1
        if fbank and float(match_many(np.stack(fbank), ft).max()) >= MIN_MATCH:
            rejects += 1
            continue
        masses.append((float(m1), float(m2)))
        temps.append(t / n)  # time-domain, unit in the same metric
        fbank.append(ft)
        rejects = 0
        if verbose:
            print(f"  {len(temps):3d} templates after {tried:5d} draws "
                  f"({time.time()-t0:5.0f}s)  m1={m1:5.1f} m2={m2:5.1f}")
    return np.array(masses), np.stack(temps), tried


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 Step 1 — build the template bank")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=BANK_SEED)
    args = ap.parse_args()

    row = ref_block()
    print(f"stochastic bank: MM {MIN_MATCH}, masses U{MASS_RANGE} (m1>=m2), "
          f"reference block {row}, seed {args.seed}\n")
    masses, temps, tried = build_bank(args.seed, row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("m1", data=masses[:, 0])
        f.create_dataset("m2", data=masses[:, 1])
        f.create_dataset("templates", data=temps)  # (n, CROP_N) float64, unit-norm
        f.attrs.update(min_match=MIN_MATCH, seed=args.seed, ref_block=row,
                       ref_merger_pos=REF_MERGER_POS, n_tried=tried,
                       stop_after=STOP_AFTER, fft_n=FFT_N,
                       created=time.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n{len(masses)} templates ({tried} draws) -> {args.out}")
    print("\nnow run:  python stage4_bank_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

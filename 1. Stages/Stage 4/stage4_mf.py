"""Stage 4, Step 2 — score every crop with the matched filter. Once, cached.

For every row of Stage-2 val, Stage-2 test, and Stage-3 test, over the 62-template bank:

    rho        max over templates x lags x phase of the normalized correlation
    chisqr     8-bin power chi-squared (reduced) at the best template's peak
    rhotilde   newSNR: rho re-weighted by chisqr — the statistic real searches use
    rho_oracle (s2test positives only) rho with the TRUE-parameter template — MF's ceiling

THE KNOWLEDGE CONTRACT: templates are re-conditioned with each row's own block filter —
exactly the per-block PSD knowledge the conditioning pipeline already grants the data.
The waveform buffers are block-independent (`place_in_buffer` output), so each worker
generates them once and only re-whitens per block. Crops are read through the same
saturation rule the CNNs see (+-CLIP_SIGMA); for Stage-2 crops that is a no-op.

NORMALIZATION — the bandlimited-noise trap (caught by this step's check, 2026-07-13):
conditioned noise has unit variance but is NOT white — its power sits in the 30-350 Hz
band, at ~3.2x the white spectral level, exactly where the templates live. A flat-metric
correlation therefore came out sqrt(3.2) ~ 1.7x hot on noise AND on signal (measured:
quadrature std 1.719, oracle recovery 1.696, chisq_r 2.98 ~ 1.7^2 on matched
injections — one number, three hats). The fix is the real matched-filter metric:
RE-WHITEN PER BLOCK, CAUSALLY. Each block's conditioned-noise spectrum is estimated
from crops of its role-0 PSD-SOURCE block (never scored, never trained on), all spectra
are weighted by 1/sqrt(S) restricted to the band, and the residual scalar is calibrated
on those same causal crops so noise quadratures are ~N(0,1) by construction. The check
then verifies the calibration transfers to held-out val negatives.

Run (from WSL2, ~5 min on 16 workers):
    python stage4_mf.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 3"))
from scipy.ndimage import median_filter

from stage1_dataset import place_in_buffer
from stage1_injection_check import make_waveform
from stage1_noise_check import BAND, SAMPLE_RATE

import stage2_condition as s2c
from stage2_dataset import DEFAULT_OUT as S2_DATASET, _worker_init
from stage2_fetch import DEFAULT_OUT as S2_STRAIN

from stage3_data import CLIP_SIGMA
from stage3_dataset import DEFAULT_OUT as S3_DATASET
from stage3_fetch import DEFAULT_OUT as S3_STRAIN

from stage4_bank import DEFAULT_OUT as BANK_H5, FFT_N, fdom, self_norm

DEFAULT_OUT = Path.home() / "ligo-data" / "stage4_scores.h5"
CHISQ_BINS = 8
SQRT2 = float(np.sqrt(2.0))
_F_ONESIDED = FFT_N // 2 + 1


def rho_series(fd: np.ndarray, FH: np.ndarray) -> np.ndarray:
    """(n_templates, FFT_N) complex z(t) for one crop against unit-metric templates.
    rho at any lag/template is sqrt(2)|z|."""
    prod = fd[None, :] * np.conj(FH)
    full = np.zeros((FH.shape[0], FFT_N), dtype=complex)
    full[:, :_F_ONESIDED] = prod
    return np.fft.ifft(full, axis=-1)


def chisq_reduced(prod_row: np.ndarray, starts: np.ndarray, lag: int) -> float:
    """Power chi-squared at one template's peak lag, from its one-sided cross-spectrum.

    z_l(lag) = (1/N) sum_{f in bin l} prod_f e^{2pi i f lag / N}; bins hold equal
    template power, so on noise (or a matched signal) chisq_r ~ 1, and a transient
    whose power sits in the wrong bins sends it >> 1.
    """
    phases = np.exp(2j * np.pi * np.arange(_F_ONESIDED) * lag / FFT_N)
    z_l = np.add.reduceat(prod_row * phases, starts) / FFT_N
    rho_l = SQRT2 * z_l
    rho_c = rho_l.sum()
    chisq = CHISQ_BINS * float(np.sum(np.abs(rho_l - rho_c / CHISQ_BINS) ** 2))
    return chisq / (2 * CHISQ_BINS - 2)


def new_snr(rho: float, chi_r: float) -> float:
    """The standard re-weighting: untouched below chisq_r = 1, crushed above."""
    return rho if chi_r <= 1.0 else rho / ((1.0 + chi_r**3) / 2.0) ** (1.0 / 6.0)


# ----------------------------------------------------------------- worker side
_ctx: dict = {}


def _init(strain_path, dataset_path, bank_path) -> None:
    """Fork hygiene per stage2_dataset, then the block-independent half of the bank:
    waveform buffers are generated once per worker, whitening happens per block."""
    _worker_init(strain_path)
    with h5py.File(bank_path, "r") as f:
        masses = np.stack([f["m1"][:], f["m2"][:]], axis=1)
        merger_pos = float(f.attrs["ref_merger_pos"])
    _ctx["masses"] = masses
    _ctx["buffers"] = [place_in_buffer(make_waveform(float(a), float(b)), merger_pos)
                       for a, b in masses]
    _ctx["merger_pos"] = merger_pos
    _ctx["dataset_path"] = dataset_path
    _ctx["file"] = None


def _bin_starts(fh: np.ndarray) -> np.ndarray:
    """8 contiguous equal-power frequency bins of a unit template (reduceat starts)."""
    cp = np.cumsum(np.abs(fh) ** 2)
    starts = np.searchsorted(cp, np.arange(CHISQ_BINS) / CHISQ_BINS * cp[-1])
    return np.maximum.accumulate(np.maximum(starts, np.arange(CHISQ_BINS)))


_CAL_LAGS = slice(3400, FFT_N, 89)  # modest shifts, decorrelated — see stage4_mf_check


@lru_cache(maxsize=2)
def _recolor(block: int):
    """The re-whitening weight for THIS block, estimated CAUSALLY: the conditioned-noise
    spectrum of 64 crops of the block's role-0 PSD-source predecessor — crops that are
    never scored and never trained on. Median over crops (a glitch in the PSD block
    cannot drag it), median-smoothed over frequency, zero outside the band (the noise
    there is filter roll-off, not information)."""
    src = s2c._psd_source_row(block)
    fn = np.stack([fdom(s2c.condition_real(s2c.raw_buffer(src, off), block)
                        .astype(np.float64))
                   for off in range(0, 505, 8)])
    S = median_filter(np.median(np.abs(fn) ** 2, axis=0), size=9)
    freqs = np.arange(_F_ONESIDED) * SAMPLE_RATE / FFT_N
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    w = np.zeros(_F_ONESIDED)
    w[band] = 1.0 / np.sqrt(S[band])
    return w, fn


@lru_cache(maxsize=2)
def _bank_for_block(block: int):
    """The bank under THIS block's whitening filter, in the re-whitened metric:
    unit-norm weighted spectra, chisq bins, and the per-block noise calibration s0
    (measured on the same causal role-0 crops, folded in so that on noise each
    quadrature of sqrt(2)*z is ~N(0,1) by construction)."""
    w, fn = _recolor(block)
    FH = np.empty((len(_ctx["buffers"]), _F_ONESIDED), dtype=complex)
    for i, buf in enumerate(_ctx["buffers"]):
        ft = fdom(s2c.condition_real(buf, block).astype(np.float64)) * w
        FH[i] = ft / self_norm(ft)
    # MEDIAN of per-crop stds, not a pooled variance: one loud glitch among the 64
    # calibration crops would inflate a pooled estimate ~30x and silently deflate every
    # rho in the block (caught 2026-07-13 as a uniform ~5% efficiency plateau).
    stds = []
    for f in fn:
        z = SQRT2 * rho_series(f * w, FH[:8])[:, _CAL_LAGS].ravel()
        stds.append(np.sqrt(0.5 * (np.var(z.real) + np.var(z.imag))))
    s0 = float(np.median(stds))
    FH /= s0
    starts = np.stack([_bin_starts(fh) for fh in FH])
    return FH, starts, w, s0


def score_crop(x: np.ndarray, FH: np.ndarray, starts: np.ndarray, w: np.ndarray):
    """One crop against one conditioned bank -> per-template peak stats."""
    fd = fdom(np.clip(x.astype(np.float64), -CLIP_SIGMA, CLIP_SIGMA)) * w
    Z = rho_series(fd, FH)
    absZ = np.abs(Z)
    lags = absZ.argmax(axis=1)
    rhos = SQRT2 * absZ[np.arange(len(FH)), lags]
    prod = fd[None, :] * np.conj(FH)
    chis = np.array([chisq_reduced(prod[i], starts[i], int(lags[i]))
                     for i in range(len(FH))])
    tildes = np.array([new_snr(float(r), float(c)) for r, c in zip(rhos, chis)])
    return rhos, chis, tildes, lags


def score_block(task):
    """All of one block's rows for one dataset. Returns arrays in `idx` order."""
    tag, block, idx, want_oracle = task
    if _ctx["file"] is None:
        _ctx["file"] = h5py.File(_ctx["dataset_path"], "r")
    f = _ctx["file"]
    X = f["X"][idx, 0, :]
    FH, starts, w, s0 = _bank_for_block(int(block))

    n = len(idx)
    out = {k: np.zeros(n) for k in ("rho", "chisqr", "rhotilde", "best_m1", "best_m2")}
    out["lag"] = np.zeros(n, dtype=np.int32)
    out["rho_oracle"] = np.full(n, np.nan)

    if want_oracle:
        y = f["y"][idx]
        m1s, m2s = f["m1"][idx], f["m2"][idx]

    for j in range(n):
        rhos, chis, tildes, lags = score_crop(X[j], FH, starts, w)
        bE, bF = int(rhos.argmax()), int(tildes.argmax())
        out["rho"][j] = rhos[bE]
        out["lag"][j] = lags[bE]
        out["rhotilde"][j] = tildes[bF]
        out["chisqr"][j] = chis[bE]
        out["best_m1"][j], out["best_m2"][j] = _ctx["masses"][bE]
        if want_oracle and y[j] == 1:
            buf = place_in_buffer(make_waveform(float(m1s[j]), float(m2s[j])),
                                  _ctx["merger_pos"])
            ft = fdom(s2c.condition_real(buf, int(block)).astype(np.float64)) * w
            ft = ft / self_norm(ft) / s0
            fd = fdom(np.clip(X[j].astype(np.float64), -CLIP_SIGMA, CLIP_SIGMA)) * w
            out["rho_oracle"][j] = SQRT2 * float(np.abs(rho_series(fd, ft[None, :])).max())
    return tag, idx, out


# ----------------------------------------------------------------- parent side
def tasks_for(dataset_path: Path, split_id: int, tag: str, want_oracle: bool):
    """One task per (block, split) — rows grouped so the bank conditions once."""
    with h5py.File(dataset_path, "r") as f:
        split = f["split"][:]
        block = f["block"][:]
    rows = np.flatnonzero(split == split_id)
    return [(tag, int(b), rows[block[rows] == b], want_oracle)
            for b in np.unique(block[rows])], len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 Step 2 — matched-filter scores")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    phases = [
        (S2_STRAIN, S2_DATASET, [("s2val", 1, False), ("s2test", 2, True)]),
        (S3_STRAIN, S3_DATASET, [("s3test", 2, False)]),
    ]
    store: dict[str, dict] = {}
    t0 = time.time()
    for strain, dataset, groups in phases:
        tasks, counts = [], {}
        for tag, sid, oracle in groups:
            tk, n = tasks_for(dataset, sid, tag, oracle)
            tasks.extend(tk)
            counts[tag] = n
            pos = {}
            with h5py.File(dataset, "r") as f:
                rows = np.flatnonzero(f["split"][:] == sid)
            pos.update({int(r): i for i, r in enumerate(rows)})
            store[tag] = {"n": n, "pos": pos, "rows": rows,
                          "rho": np.zeros(n), "chisqr": np.zeros(n),
                          "rhotilde": np.zeros(n), "best_m1": np.zeros(n),
                          "best_m2": np.zeros(n), "lag": np.zeros(n, dtype=np.int32),
                          "rho_oracle": np.full(n, np.nan)}
        print(f"{dataset.name}: {sum(counts.values()):,} rows in {len(tasks)} block-tasks "
              f"({', '.join(f'{k}={v:,}' for k, v in counts.items())})")

        from tqdm import tqdm

        with Pool(args.workers, initializer=_init,
                  initargs=(strain, dataset, BANK_H5)) as pool:
            for tag, idx, out in tqdm(pool.imap_unordered(score_block, tasks),
                                      total=len(tasks), unit="block"):
                s = store[tag]
                at = np.array([s["pos"][int(r)] for r in idx])
                for k, v in out.items():
                    s[k][at] = v
    wall = time.time() - t0

    # --- latency, measured honestly: warm bank, one crop at a time, single process ------
    _init(S2_STRAIN, S2_DATASET, BANK_H5)
    with h5py.File(S2_DATASET, "r") as f:
        rows = np.flatnonzero(f["split"][:] == 1)[:100]
        Xs = f["X"][rows, 0, :]
        blk = int(f["block"][rows[0]])
    t_cond = time.time()
    FH, starts, w, s0 = _bank_for_block(blk)
    t_cond = time.time() - t_cond
    t_lat = time.time()
    for x in Xs:
        score_crop(x, FH, starts, w)
    lat_ms = (time.time() - t_lat) / len(Xs) * 1e3

    n_total = sum(store[t]["n"] for t in store)
    print(f"\nlatency {lat_ms:.1f} ms/crop (warm bank, 1 process); bank conditioning "
          f"{t_cond:.1f} s/block; throughput {n_total/wall:,.0f} crops/s "
          f"({args.workers} workers, oracle included)")

    with h5py.File(args.out, "w") as f:
        for tag, s in store.items():
            g = f.create_group(tag)
            g.create_dataset("row", data=s["rows"])
            for k in ("rho", "chisqr", "rhotilde", "best_m1", "best_m2", "lag", "rho_oracle"):
                g.create_dataset(k, data=s[k])
        f.attrs.update(bank=str(BANK_H5), chisq_bins=CHISQ_BINS, clip_sigma=CLIP_SIGMA,
                       latency_ms_per_crop=lat_ms, bank_condition_s_per_block=t_cond,
                       throughput_crops_per_s=n_total / wall, wall_s=wall,
                       workers=args.workers, n_total=n_total,
                       created=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"-> {args.out}")
    print("\nnow run:  python stage4_mf_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

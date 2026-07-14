"""Stage 4, Step 2's check — is rho actually the SNR it claims to be?

The claims, each of which would mislabel the benchmark's axes if false:
  - NOISE CALIBRATION: on plain real negatives, each quadrature of the per-template
    correlation is ~N(0,1). If the std is off, every threshold is mislabelled.
  - ORACLE RECOVERY: an injection at SNR x, filtered with its own template, must return
    rho ~ x. This is the same identity stage1_injection_check proved at injection time,
    now round-tripped through the whole MF machine.
  - BANK vs ORACLE: the bank's fitting factor must survive contact with real injections
    (rho_bank / rho_oracle >= ~MM on average) — and the oracle must WIN on average,
    or the "ceiling" isn't one.
  - CHI-SQUARED DISCRIMINATION: chisq_r ~ 1 for well-matched injections, >> 1 for loud
    glitches. If it can't tell those apart, arm F is arm E with extra steps.
  - DETERMINISM: rescore a block, get the stored numbers bit-exactly.

Run (from WSL2, after stage4_mf.py, ~2 min):
    python stage4_mf_check.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 3"))
from stage2_dataset import DEFAULT_OUT as S2_DATASET
from stage2_fetch import DEFAULT_OUT as S2_STRAIN

from stage3_dataset import DEFAULT_OUT as S3_DATASET

from stage4_bank import DEFAULT_OUT as BANK_H5
from stage4_mf import DEFAULT_OUT, SQRT2, _bank_for_block, _init, rho_series, score_block

OUT = Path(__file__).parent / "outputs"
LOUD_SNR = 20.0  # chisq discrimination scales with rho^2 — judge it where it can act


def main() -> int:
    results = []
    f = h5py.File(DEFAULT_OUT, "r")

    # --- structure -----------------------------------------------------------------------
    with h5py.File(S2_DATASET, "r") as d:
        split2, y2 = d["split"][:], d["y"][:]
        snr2 = d["snr"][:]
    with h5py.File(S3_DATASET, "r") as d:
        split3 = d["split"][:]
        isg3, glab3 = d["is_glitch"][:], d["glitch_label"][:]
        gsnr3 = d["glitch_snr"][:]
        names3 = list(d.attrs["glitch_labels"])
    sizes_ok = (f["s2val"]["rho"].shape[0] == int((split2 == 1).sum())
                and f["s2test"]["rho"].shape[0] == int((split2 == 2).sum())
                and f["s3test"]["rho"].shape[0] == int((split3 == 2).sum()))
    results.append(("score arrays match the split sizes", sizes_ok, "row mismatch"))
    finite = all(np.isfinite(f[g]["rho"][:]).all() and (f[g]["rho"][:] > 0).all()
                 and np.isfinite(f[g]["chisqr"][:]).all()
                 for g in ("s2val", "s2test", "s3test"))
    results.append(("all rho finite and positive, all chisq finite", finite, "bad scores"))

    # rhotilde <= rho, equal where chisq_r <= 1
    ok_tilde = True
    for g in ("s2val", "s2test", "s3test"):
        rho, til, chi = f[g]["rho"][:], f[g]["rhotilde"][:], f[g]["chisqr"][:]
        ok_tilde &= bool((til <= rho * (1 + 1e-12)).all())
    results.append(("rhotilde never exceeds rho", ok_tilde, "re-weighting broken"))

    # --- noise calibration: fresh correlations on plain val negatives ---------------------
    _init(S2_STRAIN, S2_DATASET, BANK_H5)
    val_rows = np.flatnonzero(split2 == 1)
    with h5py.File(S2_DATASET, "r") as d:
        vblocks = d["block"][:][val_rows]
        neg_rows = val_rows[y2[val_rows] == 0]
        take = neg_rows[:: max(1, len(neg_rows) // 300)][:300]
        Xn = d["X"][np.sort(take), 0, :]
        nblk = d["block"][:][np.sort(take)]
    from stage4_bank import fdom

    zs = []
    for blk in np.unique(nblk):
        FH, _, w, _ = _bank_for_block(int(blk))
        for x in Xn[nblk == blk][:40]:
            Z = rho_series(fdom(x.astype(np.float64)) * w, FH[:8])
            # modest left-shifts only: large lags slide the template out of the crop and
            # deflate the variance by construction; stride ~89 decorrelates the draws.
            # These are HELD-OUT val negatives — the per-block calibration used only
            # role-0 PSD-source crops, so this measures that the calibration transfers.
            zs.append(SQRT2 * Z[:, 3400:4096:89].ravel())
    zs = np.concatenate(zs)
    s_re, s_im = float(np.std(zs.real)), float(np.std(zs.imag))
    results.append((f"noise calibration: quadrature std {s_re:.3f} / {s_im:.3f} in [0.9, 1.1]",
                    0.9 < s_re < 1.1 and 0.9 < s_im < 1.1, "thresholds mislabelled"))

    # --- oracle recovery + bank vs oracle -------------------------------------------------
    test_rows = np.flatnonzero(split2 == 2)
    rho_o = f["s2test"]["rho_oracle"][:]
    rho_b = f["s2test"]["rho"][:]
    pos = y2[test_rows] == 1
    results.append(("oracle scored for every positive, only positives",
                    bool(np.isfinite(rho_o[pos]).all() and np.isnan(rho_o[~pos]).all()),
                    "oracle rows wrong"))
    strong = pos & (snr2[test_rows] >= 8)
    ratio = np.median(rho_o[strong] / snr2[test_rows][strong])
    results.append((f"oracle recovery: median rho_oracle/SNR = {ratio:.3f} in [0.9, 1.1] "
                    f"(n={int(strong.sum())})", 0.9 < ratio < 1.1, "x-axis mislabelled"))
    bank_ff = np.median(rho_b[strong] / rho_o[strong])
    results.append((f"bank holds its match on real injections: median rho/rho_oracle = "
                    f"{bank_ff:.3f} >= 0.95", bank_ff >= 0.95, "bank holes on real data"))
    frac_win = float((rho_o[strong] >= rho_b[strong] - 0.5).mean())
    results.append((f"oracle is a ceiling ({frac_win:.0%} of strong positives within "
                    "0.5 of or above bank)", frac_win >= 0.95, "oracle not oracular"))

    # --- chi-squared: matched signals vs loud glitches -------------------------------------
    chi_t = f["s2test"]["chisqr"][:]
    well = pos & (snr2[test_rows] >= 12)
    chi_sig = float(np.median(chi_t[well]))
    results.append((f"chisq_r ~ 1 on well-matched injections (median {chi_sig:.2f} < 2, "
                    f"n={int(well.sum())})", chi_sig < 2.0, "chisq can't pass a signal"))
    t3 = np.flatnonzero(split3 == 2)
    chi3 = f["s3test"]["chisqr"][:]
    print("\n  chisq_r by glitch class (test split, median):")
    for li, nm in enumerate(names3):
        m = (isg3[t3] == 1) & (glab3[t3] == li)
        if m.sum():
            med = float(np.median(chi3[m]))
            loud = float(np.median(gsnr3[t3][m]))
            print(f"    {nm:22s} n={int(m.sum()):4d}  chisq_r {med:8.1f}  (omicron snr ~{loud:.0f})")
    # chisq_r - 1 grows with rho^2 x mismatch: a quiet glitch is chisq-invisible by
    # physics, so the claim is tested where the statistic CAN act — loud specimens.
    loud_g = (isg3[t3] == 1) & (gsnr3[t3] >= LOUD_SNR)
    chi_loud = float(np.median(chi3[loud_g]))
    results.append((f"chisq_r >> 1 on loud glitches (median {chi_loud:.1f} > 5 over "
                    f"{int(loud_g.sum())} specimens with omicron snr >= {LOUD_SNR:.0f})",
                    chi_loud > 5.0, "arm F is arm E with extra steps"))

    # --- determinism: rescore one test block ----------------------------------------------
    blk = int(vblocks[0])
    idx = val_rows[vblocks == blk]
    _, _, out = score_block(("s2val", blk, idx, False))
    at = np.array([int(np.flatnonzero(val_rows == r)) for r in idx])
    same = (np.array_equal(out["rho"], f["s2val"]["rho"][:][at])
            and np.array_equal(out["chisqr"], f["s2val"]["chisqr"][:][at]))
    results.append(("deterministic rescore of a val block (bit-exact)", same, "not reproducible"))

    print()
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}: {name}" + ("" if passed else f" — {detail}"))

    # --- figure -------------------------------------------------------------------------
    fig, (axr, axc) = plt.subplots(1, 2, figsize=(13.5, 5))
    s = snr2[test_rows][pos]
    axr.plot(s, rho_o[pos], ".", ms=2, alpha=0.4, color="tab:blue", label="oracle")
    axr.plot(s, rho_b[pos], ".", ms=2, alpha=0.25, color="tab:red", label="bank")
    axr.plot([4, 20], [4, 20], "k--", lw=1, label="rho = injected SNR")
    axr.set_xlabel("injected SNR"), axr.set_ylabel("recovered rho")
    axr.set_title(f"Recovery (median oracle ratio {ratio:.3f})")
    axr.legend(fontsize=8), axr.grid(alpha=0.3)
    gl = (isg3[t3] == 1)
    axc.semilogy(f["s3test"]["rho"][:][gl], chi3[gl], ".", ms=3, alpha=0.5,
                 color="tab:red", label="glitches (s3 test)")
    axc.semilogy(rho_b[well], chi_t[well], ".", ms=3, alpha=0.5,
                 color="tab:blue", label="injections SNR>=12 (s2 test)")
    axc.axhline(1, color="k", lw=1, ls=":")
    axc.set_xlabel("rho (bank)"), axc.set_ylabel("chisq_r at peak")
    axc.set_title("What the chi-squared sees")
    axc.legend(fontsize=8), axc.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "2_mf_check.png", dpi=130)
    print(f"\n  plot -> {OUT / '2_mf_check.png'}")
    f.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

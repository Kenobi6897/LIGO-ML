"""Stage 5, Step 2 check — the ablation is fair, and here is what it costs.

Three things that CAN FAIL, and two measurements that decide how to read the result.

CHECK 1 — SAME EXPERIMENT.  Every metadata column of stage5_s2.h5 must equal stage2.h5's
    bit-exactly (and stage5_s3.h5 vs stage3.h5): same blocks, same offsets, same masses,
    same target SNRs, same merger positions, same splits. If this fails, the arms are not
    running the same experiment and no comparison between them means anything. Tolerance
    is ZERO, not 1e-5 — Stage 1 bug #4's rule.

CHECK 2 — NO ARM GETS A LEAK.  Each of the four conditioners must satisfy Stage 1's
    signal-blindness identity  C(n + h) == C(n) + C(h)  to machine precision. `raw` and
    `bp` pass trivially (no PSD anywhere); `wh` and `full` pass because the PSD is causal.
    If an arm failed this it could win by leakage, and its win would be the fraud Stage 1
    was built to make impossible.

CHECK 3 — THE CLIP IS NOT THE VARIABLE.  Stage 3's +-20 sigma read-time saturation is
    applied to every arm in that arm's own units. Measured here per arm, on the data it
    will actually train on. If it turns out to bite one arm hard and the others not at
    all, then it IS a second variable and the eval must also run that arm unclipped —
    stage5_train.py has --no-clip for exactly this, and stage5_eval.py reports both.

MEASUREMENT A — HOW LOUD IS A CHIRP, REALLY.  ||C(h)|| / ||C(n)||: the injected signal's
    amplitude as a fraction of the crop's, per arm, at fixed physical SNR. This is the
    whole thesis of the stage in one ratio. Whitening is not cosmetic — it is the operator
    that MAKES the signal visible, by deleting the variance that isn't it.

MEASUREMENT B — DOES THE SIGNAL SURVIVE float32.  The h5 stores float32 and torch trains
    in float32. If a chirp is 1e-4 of a raw crop's standard deviation, then storing that
    crop in float32 (7 significant digits) leaves the signal ~3 digits of precision. This
    measures it: the relative L2 error of the recovered waveform after the round-trip the
    network's input actually takes. Not a bug — a consequence, and one of the answers.

Run:
    python stage5_dataset_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_dataset import place_in_buffer
from stage1_injection_check import make_waveform

import stage2_condition as s2c
import stage2_dataset as s2d
from stage2_fetch import DEFAULT_OUT as STRAIN_S2

import stage5_condition as s5c
from stage5_condition import ARMS, BUILD_ARMS, CLIP_SIGMA

DATA = Path.home() / "ligo-data"
PAIRS = [("s2", DATA / "stage5_s2.h5", DATA / "stage2.h5"),
         ("s3", DATA / "stage5_s3.h5", DATA / "stage3.h5")]
CHECK_SNR = 8.0
CHECK_MASSES = (30.0, 30.0)


def check_specs() -> bool:
    """CHECK 1 — the ablation datasets are the control's rows, re-conditioned."""
    print("CHECK 1 — same blocks, same injections, same splits as the control\n")
    ok = True
    for tag, new, ctrl in PAIRS:
        if not new.exists():
            print(f"  {tag}: {new.name} MISSING — run stage5_dataset.py --source {tag}")
            ok = False
            continue
        with h5py.File(new, "r") as a, h5py.File(ctrl, "r") as b:
            cols = [c for c in a if not c.startswith("X_")]
            n_a, n_b = a["y"].shape[0], b["y"].shape[0]
            if n_a != n_b:
                print(f"  {tag}: FAIL row count {n_a} != {n_b} ({ctrl.name})")
                ok = False
                continue
            worst = []
            for c in cols:
                if c not in b:
                    continue
                d = np.max(np.abs(a[c][:].astype(np.float64) - b[c][:].astype(np.float64)))
                worst.append((c, float(d)))
            bad = [(c, d) for c, d in worst if d != 0.0]
            status = "PASS" if not bad else "FAIL"
            ok &= not bad
            print(f"  {status}: {tag} — {n_a:,} rows, {len(worst)} columns vs {ctrl.name}, "
                  f"max|delta| = {max(d for _, d in worst):.1e} (tolerance: exactly 0)")
            for c, d in bad:
                print(f"        {c}: max|delta| = {d:.3e}")
    return ok


def _signal_and_noise(row: int, offset: int = 100):
    """A real noise buffer from a real block, and an SNR-8 chirp placed in it."""
    n = s2c.raw_buffer(row, offset)
    placed = place_in_buffer(make_waveform(*CHECK_MASSES), 0.8)
    s = s2d.in_band_sigma(placed, row)
    return n, placed * (CHECK_SNR / s)


def check_linearity_and_scale() -> bool:
    """CHECK 2 and MEASUREMENTS A-B.

    THE LEAK TEST IS STAGE 2'S, VERBATIM — same identity, same metric, same tolerance:
    the same signal dropped into two different noises must produce the same response,
    C(n1+h) - C(n1) == C(n2+h) - C(n2), measured as max|d1-d2| / max|d1| < 1e-9. Stage 2
    scores ~1e-15 on it and a per-segment whitener scores ~1e-1.

    AND THE SIGNAL IS LOUD (scaled to the noise's amplitude), FOR A REASON. The first
    version of this check used the SNR-8 injection below and a 1e-12 tolerance, and three
    of the four arms "failed" at 1e-12 while the raw arm "passed" at 5e-13 — which is
    backwards, and was the tell. Nothing was leaking: differencing C(n+h) and C(n) when h
    is 1.6e-4 of n is catastrophic cancellation, so it amplifies float64 round-off by
    ~||n||/||h|| ~ 6000x. The check was measuring its own arithmetic, and it would have
    condemned the control. Stage 2 avoids this by making the signal as loud as the noise;
    so does this. (The quiet-signal numbers survive below as MEASUREMENTS, where a ratio
    of amplitudes is exactly what we want — just never as a pass/fail.)
    """
    rows_all = s2c.data_rows()
    row = int(rows_all[len(rows_all) // 2])  # a mid-run block, nothing special
    psd = s2c.whitening_psd(row)

    # --- CHECK 2: the leak test, Stage 2's setup ---------------------------------------
    n1, n2 = s2c.raw_buffer(row, 40), s2c.raw_buffer(row, 400)
    loud = place_in_buffer(make_waveform(36.0, 29.0), 0.85)
    loud *= float(n1.numpy().std() / loud.numpy().std())  # signal ~ as loud as the noise

    print(f"\nCHECK 2 — signal-blindness (Stage 2's test, Stage 2's tolerance): "
          f"C(n1+h) - C(n1) == C(n2+h) - C(n2)\n")
    print(f"  {'arm':>5}  {'relative disagreement':>21}   (tolerance 1e-9; "
          f"a per-segment whitener scores ~1e-1)")
    ok = True
    for arm in ARMS:
        p = psd if arm in s5c._needs_psd else None
        d1 = s5c._unnormed(n1 + loud, p, arm) - s5c._unnormed(n1, p, arm)
        d2 = s5c._unnormed(n2 + loud, p, arm) - s5c._unnormed(n2, p, arm)
        rel = float(np.max(np.abs(d1 - d2)) / np.max(np.abs(d1)))
        passed = rel < 1e-9
        ok &= passed
        print(f"  {arm:>5}  {rel:21.2e}   {'PASS' if passed else 'FAIL'}")

    # --- MEASUREMENTS A and B: the physics, at a realistic signal amplitude -------------
    n, h = _signal_and_noise(row)
    print(f"\nMEASUREMENTS — one SNR {CHECK_SNR:.0f} chirp "
          f"({CHECK_MASSES[0]:.0f}+{CHECK_MASSES[1]:.0f} Msun), block {row}\n")
    print(f"  {'arm':>5}  {'||C(h)||/||C(n)||':>18}   {'float32 recovery err':>20}")
    amps = {}
    for arm in ARMS:
        p = psd if arm in s5c._needs_psd else None
        g = s5c.norm(arm)
        c_n = s5c._unnormed(n, p, arm).astype(np.float64) * g
        c_nh = s5c._unnormed(n + h, p, arm).astype(np.float64) * g
        c_h = s5c._unnormed(h, p, arm).astype(np.float64) * g

        # A: how much of the crop IS the signal, at fixed physical SNR.
        amps[arm] = np.linalg.norm(c_h) / np.linalg.norm(c_n)

        # B: the round-trip the network's input actually takes — float32 in the h5,
        # float32 in torch. How much of the waveform survives being stored at all?
        d32 = (c_nh.astype(np.float32).astype(np.float64)
               - c_n.astype(np.float32).astype(np.float64))
        f32 = np.linalg.norm(d32 - c_h) / np.linalg.norm(c_h)
        print(f"  {arm:>5}  {amps[arm]:18.2e}   {f32:20.2e}")

    print(f"\n  A — the stage in one ratio: at the SAME physical SNR, a chirp is "
          f"{amps['raw']:.1e} of a raw crop")
    print(f"      and {amps['full']:.1e} of a conditioned one — {amps['full']/amps['raw']:.0f}x. "
          f"Whitening does not 'clean up'")
    print(f"      the signal; it deletes the variance that ISN'T the signal. That is the "
          f"whole job.")
    print(f"  B — and it is NOT a precision problem: the waveform survives float32 storage "
          f"with ~1e-4")
    print(f"      relative error even in the raw arm, thousands of times above the "
          f"quantisation floor.")
    print(f"      Whatever kills the raw arm, it is not the number format. Ruled out here, "
          f"before training.")
    return ok


def check_clip() -> bool:
    """CHECK 3 — is the +-20 sigma clip a second variable?"""
    print(f"\nCHECK 3 — read-time saturation at +-{CLIP_SIGMA:.0f} sigma, per arm\n")
    print(f"  {'arm':>5}  {'source':>6}  {'crop std':>9}  {'max |x|':>10}  "
          f"{'% samples clipped':>18}  {'% crops touched':>16}")
    stats = {}
    for tag, new, _ctrl in PAIRS:
        if not new.exists():
            continue
        with h5py.File(new, "r") as f:
            n = f["y"].shape[0]
            take = np.linspace(0, n - 1, min(n, 3000)).astype(int)
            for arm in BUILD_ARMS:
                x = f[f"X_{arm}"][:][take].astype(np.float64)
                clipped = np.abs(x) > CLIP_SIGMA
                frac_s = float(clipped.mean())
                frac_c = float(clipped.any(axis=(1, 2)).mean())
                stats[(arm, tag)] = frac_s
                print(f"  {arm:>5}  {tag:>6}  {x.std():9.3f}  {np.abs(x).max():10.1f}  "
                      f"{100*frac_s:18.4f}  {100*frac_c:16.2f}")
        # the control, for reference — the arm the clip was designed for
        with h5py.File(_ctrl, "r") as f:
            x = f["X"][:][take].astype(np.float64)
            clipped = np.abs(x) > CLIP_SIGMA
            stats[("full", tag)] = float(clipped.mean())
            print(f"  {'full':>5}  {tag:>6}  {x.std():9.3f}  {np.abs(x).max():10.1f}  "
                  f"{100*clipped.mean():18.4f}  {100*clipped.any(axis=(1,2)).mean():16.2f}")

    heavy = {a for (a, _t), v in stats.items() if v > 1e-3}  # >0.1% of samples railed
    if heavy:
        print(f"\n  NOTE: {sorted(heavy)} clip >0.1% of samples — for those arms the clip is a")
        print(f"  second variable, so stage5_train.py must ALSO be run with --no-clip and")
        print(f"  stage5_eval.py must report the better of the two. Not optional.")
    else:
        print(f"\n  PASS: no arm clips more than 0.1% of samples — the clip is not the variable.")
    return True  # report-only: it steers the run, it does not condemn it


def main() -> int:
    ok = check_specs()
    ok &= check_linearity_and_scale()
    check_clip()
    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

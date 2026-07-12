"""Stage 1, Step 4's check — is the dataset on disk the dataset we think we wrote?

Four properties, each of which can fail. Two are bookkeeping; two are the ones that matter.

  1. STRUCTURE. Shapes, dtypes, 50/50 balance, no NaN/Inf, negatives carry no waveform
     metadata, and — the one with teeth — **noise seeds are unique, and no seed appears in
     two splits.** A repeated noise realisation straddling train/test means the model meets
     the same noise twice and can memorise it. That is a leak, and it is invisible in every
     metric except the one you care about.

  2. REPRODUCIBILITY, BIT-EXACT. Take stored rows, rebuild them from their stored metadata
     alone (seed, masses, SNR, merger position) through the same `build_segment`, and
     require the result to be *identical* to what is in the file. This is the check that
     the file is what the code says it is. It would catch a writer that shuffled rows out
     from under its labels, a parameter that never made it into the segment, or any hidden
     per-segment randomness — all of which produce a file that trains happily and means
     something other than what you think.

  3. SNR CALIBRATION — THE MONEY PLOT'S X-AXIS. `condition()` is linear and signal-blind,
     so we may legally condition the *pure waveform*: the signal present in the CNN's input
     is exactly C(h), with no noise term. We then measure its SNR the way a detection
     statistic actually would — against the **empirical background of this very file**:

         rho = ||C(h)||^2 / sigma_bg,   sigma_bg = std over negatives of  C(h) . C(n)

     and require it to come back as the SNR we asked for. If it doesn't, every point on
     the efficiency-vs-SNR curve is mislabelled and the curve is fiction.

     Why not the obvious ||C(h)||_2? Because that identity needs the conditioned noise to
     be WHITE and UNIT-VARIANCE in the same basis, and ours is neither: condition() band-
     passes (so the samples are correlated, not white) and then applies one global scale
     constant (`_norm()`), which ||C(h)|| inherits and the true SNR does not. The version
     above cancels both — numerator and denominator carry the same arbitrary scale, and
     both see the same band. It assumes nothing about the noise we did not measure.

  3b. AND THE STATISTIC HAS UNIT VARIANCE. Same background, but correlated against the
     STORED row rather than the pure template: (C(h) . x) / sigma_bg must equal the SNR
     from (3) plus N(0,1). Getting the unit variance back is what says sigma_bg is the
     right normalisation — and, because a misaligned or missing signal would drive the
     recovered value to ~0, it is also what says the row on disk really contains the
     waveform its metadata claims, at the sample offset we think.

  4. NO VARIANCE LEAK BEYOND THE SIGNAL'S OWN ENERGY. A positive genuinely *does* have more
     energy than a negative — that is the signal, not a leak. But how much more is
     predictable: adding a signal of SNR rho to unit-variance noise raises the segment's
     variance to 1 + rho^2/N. We check the measured std against that prediction. Coming in
     *above* it means something is scaling segments per-segment, which is the unit-variance
     leak wearing a disguise.

Run (from WSL2):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_dataset_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage1_condition import CROP_N, condition
from stage1_dataset import DEFAULT_OUT, build_segment, in_band_sigma, make_specs, place_in_buffer
from stage1_injection_check import make_waveform
from stage1_noise_check import SAMPLE_RATE

import h5py

OUT = Path(__file__).parent / "outputs"
N_REPRO = 12  # rows rebuilt bit-for-bit — each one costs a full segment build
N_SNR = 150  # rows whose SNR calibration we verify
N_BG = 512  # negatives correlated against each template to measure sigma_bg
            # std of a std from N samples is ~1/sqrt(2N) — 512 buys ~3%, well inside the 10% band


def signal_only(m1: float, m2: float, merger_pos: float, target_snr: float) -> np.ndarray:
    """C(h): the conditioned waveform, with no noise — the exact signal term of a stored row.

    Legal precisely because `condition()` is signal-blind (Step 3). Feed the leaky
    whitener a noiseless signal and it is not even well-defined — which is the point.

    ⚠️ THE SCALING IS NOT OPTIONAL. `build_segment` stores `noise + placed*(snr/sigma)`, and
    condition() is linear, so the signal term of the row is C(placed * snr/sigma) — NOT
    C(placed). Dropping the factor conditions the raw IMRPhenomD waveform at pycbc's default
    distance, whose optimal SNR is in the *thousands*, and every SNR in this check comes back
    ~1000x high with a mass-dependent spread (sigma depends on the masses). It did, and the
    spread is what gave it away. Mirror `build_segment` here or the check measures a
    different signal from the one on disk.
    """
    placed = place_in_buffer(make_waveform(float(m1), float(m2)), float(merger_pos))
    scaled = placed * (float(target_snr) / in_band_sigma(placed))
    return condition(scaled)


def empirical_snr(t: np.ndarray, bg: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Optimal and recovered SNR of template `t`, normalised by the file's own background.

    `bg` is (N_BG, CROP_N) of conditioned NEGATIVES straight off the disk. Correlating the
    template against them at zero lag samples the detection statistic's null distribution
    directly, so sigma_bg is measured, not assumed — which is the whole point: it needs no
    claim about the conditioned noise being white, unit-variance, or anything else.

    Zero lag, not a max over lags: we know exactly where the signal was placed, and the
    max would import the low-SNR upward bias of [[Stage 1#Step 2]] into a check that is
    supposed to be measuring the *scaling*, not the difficulty of finding it.

    Returns (sigma_bg, rho_opt, rho_recovered):
      rho_opt = ||t||^2 / sigma_bg  — the SNR of the signal in the CNN's input, noise-free
      rho_rec = (t . x) / sigma_bg  — the same signal measured off the STORED row, so it is
                                      rho_opt + N(0,1)
    """
    sigma_bg = float((bg @ t).std())
    return sigma_bg, float(t @ t) / sigma_bg, float(x @ t) / sigma_bg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if not args.path.exists():
        raise SystemExit(f"{args.path} not found — run stage1_dataset.py first")

    checks: list[tuple[str, bool, str]] = []
    with h5py.File(args.path, "r") as f:
        X, y = f["X"], f["y"][:]
        snr, m1, m2 = f["snr"][:], f["m1"][:], f["m2"][:]
        merger, seed, split = f["merger_pos"][:], f["seed"][:], f["split"][:]
        n = len(y)
        attrs = dict(f.attrs)

        print(f"{args.path}  —  {n:,} segments, {args.path.stat().st_size/1e6:.0f} MB")
        print(f"  written {attrs.get('created')}, seed {attrs.get('dataset_seed')}")
        print(f"  SNR definition: {attrs.get('snr_definition')}\n")

        # --- 1. structure ------------------------------------------------------
        print("[1] structure")
        pos, neg = y == 1, y == 0
        checks.append(("X shape (N,1,2048)", X.shape == (n, 1, CROP_N), str(X.shape)))
        checks.append(("X dtype float32", X.dtype == np.float32, str(X.dtype)))

        # The parameters must be stored at the precision they were USED, or check [2] below
        # cannot be bit-exact: a float32 mass is a different waveform. This check exists so
        # that a regression to f4 says so, instead of resurfacing as an unexplained 1e-5.
        param_dtypes = {k: f[k].dtype for k in ("m1", "m2", "snr", "merger_pos")}
        checks.append(("waveform params stored float64",
                       all(d == np.float64 for d in param_dtypes.values()),
                       ", ".join(f"{k} {d}" for k, d in param_dtypes.items())))
        checks.append(("50/50 balance", pos.sum() == neg.sum() == n // 2,
                       f"{pos.sum():,}+ / {neg.sum():,}-"))

        sample = X[:: max(1, n // 2000)][:]
        finite = bool(np.isfinite(sample).all())
        checks.append(("X finite (no NaN/Inf)", finite, f"{sample.size:,} sampled"))

        checks.append(("negatives carry no waveform",
                       bool(np.all(snr[neg] == 0) and np.all(m1[neg] == 0) and np.all(merger[neg] == 0)),
                       "snr/m1/m2/merger all 0"))
        checks.append(("positives all have SNR in range",
                       bool(np.all((snr[pos] >= attrs["snr_low"]) & (snr[pos] <= attrs["snr_high"]))),
                       f"{snr[pos].min():.1f}-{snr[pos].max():.1f}"))
        checks.append(("merger inside the crop, randomised",
                       bool(np.all((merger[pos] >= attrs["merger_low"]) & (merger[pos] <= attrs["merger_high"]))
                            and merger[pos].std() > 0.05),
                       f"{merger[pos].min():.2f}-{merger[pos].max():.2f}, sd {merger[pos].std():.3f}"))

        # The leak with teeth: a noise realisation shared across splits.
        uniq = len(np.unique(seed)) == n
        cross = 0
        if not uniq:
            for a in range(3):
                for b in range(a + 1, 3):
                    cross += len(np.intersect1d(seed[split == a], seed[split == b]))
        checks.append(("noise seeds unique across the dataset", uniq,
                       f"{len(np.unique(seed)):,} distinct of {n:,}"))
        checks.append(("no noise realisation shared between splits", cross == 0,
                       f"{cross} shared"))

        counts = [int((split == s).sum()) for s in range(3)]
        checks.append(("splits train/val/test = 80/10/10", counts[0] == int(0.8 * n),
                       f"{counts[0]:,} / {counts[1]:,} / {counts[2]:,}"))

        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {detail}")

        # --- 2. bit-exact reproduction from metadata --------------------------
        print(f"\n[2] rebuild {N_REPRO} stored rows from their metadata alone")
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(n, size=N_REPRO, replace=False))
        worst = 0.0
        for r in rows:
            spec = (int(r), int(y[r]), int(seed[r]), float(m1[r]), float(m2[r]),
                    float(snr[r]), float(merger[r]))
            _, rebuilt = build_segment(spec)
            worst = max(worst, float(np.abs(rebuilt - X[r, 0]).max()))
        exact = worst == 0.0
        print(f"  {'PASS' if exact else 'FAIL'}  max |rebuilt - stored| = {worst:.3e}  (want exactly 0)")
        checks.append(("rebuilds bit-exactly from metadata", exact, f"max diff {worst:.1e}"))

        # Independently: the specs the writer *claims* to have drawn must be the specs on
        # disk. Catches a metadata array written from the wrong source or out of order.
        specs = make_specs(n, int(attrs["dataset_seed"]))
        specs_match = bool(
            np.array_equal(specs.label.astype(np.int8), y)
            and np.array_equal(specs.seed.astype(np.int64), seed)
            and np.array_equal(specs.snr, snr)  # exact: both are the same float64 draw
            and np.array_equal(specs.split, split)
        )
        print(f"  {'PASS' if specs_match else 'FAIL'}  metadata matches a fresh draw from seed "
              f"{attrs['dataset_seed']}")
        checks.append(("metadata reproducible from the dataset seed", specs_match, ""))

        # --- 3. SNR calibration, against the file's own background ---------------
        neg_all = np.flatnonzero(neg)
        n_bg = min(N_BG, len(neg_all))
        bg_rows = np.sort(rng.choice(neg_all, size=n_bg, replace=False))
        print(f"\n[3] SNR calibration on {N_SNR} positives — background from {n_bg} negatives in the file")

        # The background segments themselves. Read once: every template is correlated
        # against the same ones, so sigma_bg differences are the template's, not the draw's.
        bg = np.asarray(X[bg_rows, 0, :], dtype=np.float64)

        pos_rows = rng.choice(np.flatnonzero(pos), size=min(N_SNR, int(pos.sum())), replace=False)
        want = snr[pos_rows].astype(np.float64)
        got = np.empty(len(pos_rows))  # rho_opt — the signal in the CNN's input
        rec = np.empty(len(pos_rows))  # rho_rec — the same signal, off the stored row
        sig_e = np.empty(len(pos_rows))  # ||C(h)||^2 — the signal's energy, for check 4
        for i, r in enumerate(pos_rows):
            t = signal_only(m1[r], m2[r], merger[r], snr[r]).astype(np.float64)
            s_bg, got[i], rec[i] = empirical_snr(t, bg, np.asarray(X[int(r), 0], dtype=np.float64))
            sig_e[i] = got[i] * s_bg  # = ||t||^2, by definition of rho_opt

        ratio = got / want
        med = float(np.median(ratio))
        spread = float(np.std(ratio))
        snr_ok = 0.9 <= med <= 1.1 and spread < 0.1
        print(f"  ||C(h)||^2 / sigma_bg,  / requested SNR:  median {med:.3f}  "
              f"(5th-95th {np.percentile(ratio,5):.3f}-{np.percentile(ratio,95):.3f}, sd {spread:.3f})")
        print(f"  {'PASS' if snr_ok else 'FAIL'}  the x-axis of the money plot means what it says")
        checks.append(("injected SNR is calibrated in-band", snr_ok, f"median {med:.3f}x"))

        # [3b] The same statistic on the stored row: rho_opt + N(0,1). Unit variance says
        # sigma_bg is the right normalisation; the near-zero mean says the signal is really
        # in the row, at the alignment the metadata claims. A misplaced or absent waveform
        # would push rec -> 0 and the residual mean to -rho.
        resid = rec - got
        r_mean, r_std = float(resid.mean()), float(resid.std())
        # 150 samples: the sd of the mean is ~1/sqrt(150) = 0.08, of the sd ~0.06.
        stat_ok = abs(r_mean) < 0.25 and 0.75 < r_std < 1.3
        print(f"  recovered - optimal:  mean {r_mean:+.3f}  sd {r_std:.3f}  "
              f"(want 0 +- 1: the statistic has unit variance)")
        print(f"  {'PASS' if stat_ok else 'FAIL'}  the stored row contains the signal its metadata claims")
        checks.append(("detection statistic is unit-variance about the injected SNR", stat_ok,
                       f"{r_mean:+.2f} +- {r_std:.2f}"))

        # --- 4. variance: signal energy, and nothing else -----------------------
        print("\n[4] segment variance — signal energy only, no per-segment rescale")
        neg_rows = rng.choice(neg_all, size=min(400, len(neg_all)), replace=False)
        neg_std = np.array([X[int(r), 0].std() for r in neg_rows])
        pos_std = np.array([X[int(r), 0].std() for r in pos_rows])
        # Predicted: var = 1 + ||C(h)||^2/N — unit-variance noise plus the signal's own energy,
        # and NOT 1 + rho^2/N. Those coincide only if the conditioned noise is white, which it
        # is not (it is band-limited, and globally rescaled); ||C(h)||^2 is measured above, so
        # use it rather than assume it. Excess above this is a per-segment rescale — the leak.
        pred = np.sqrt(1.0 + sig_e / CROP_N)
        excess = float(np.median(pos_std / pred))
        noise_ok = 0.9 <= float(np.median(neg_std)) <= 1.1
        var_ok = 0.9 <= excess <= 1.1
        print(f"  negatives std        : {neg_std.mean():.3f} ± {neg_std.std():.3f}  (want ~1.0 — the global norm)")
        print(f"  positives std / pred : {excess:.3f}  (want ~1.0 — energy is the signal's, nothing added)")
        print(f"  {'PASS' if noise_ok and var_ok else 'FAIL'}  no unit-variance leak")
        checks.append(("conditioned noise is unit-variance", noise_ok, f"{neg_std.mean():.3f}"))
        checks.append(("positive variance = noise + signal energy", var_ok, f"{excess:.3f}x"))

        # --- the plot ----------------------------------------------------------
        loud = pos_rows[np.argmax(want)]
        quiet = pos_rows[np.argmin(want)]
        ex_neg = int(neg_rows[0])
        t_ax = np.arange(CROP_N) / SAMPLE_RATE

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        (a, b), (c, d) = axes

        for r, colour, lab in [
            (int(loud), "tab:red", f"positive, SNR {snr[loud]:.1f}"),
            (int(quiet), "tab:blue", f"positive, SNR {snr[quiet]:.1f}"),
            (ex_neg, "0.55", "negative (noise only)"),
        ]:
            a.plot(t_ax, X[r, 0], lw=0.7, alpha=0.85, color=colour, label=lab)
        for r, colour in [(int(loud), "k"), (int(quiet), "tab:cyan")]:
            a.plot(t_ax, signal_only(m1[r], m2[r], merger[r], snr[r]), lw=1.4, color=colour,
                   alpha=0.9, ls="--", label=f"C(h) alone, SNR {snr[r]:.1f}")
        a.set_xlabel("time [s]")
        a.set_ylabel("conditioned strain [σ]")
        a.set_title("What the CNN eats\n(the SNR-4 chirp is invisible by eye — that is the point)")
        a.legend(fontsize=7, loc="upper left")
        a.grid(alpha=0.3)

        b.scatter(want, rec, s=12, alpha=0.35, color="tab:orange",
                  label=f"measured off the stored row (spread {r_std:.2f}, want 1.0)")
        b.scatter(want, got, s=14, alpha=0.6, color="tab:blue",
                  label="‖C(h)‖²/σ_bg — signal in the CNN's input")
        lim = [0, max(want.max(), rec.max()) * 1.05]
        b.plot(lim, lim, "k--", lw=1, label="ideal")
        b.set_xlabel("requested injected SNR (stored in `snr`)")
        b.set_ylabel("SNR against the file's own background")
        b.set_title(f"The money plot's x-axis is calibrated\noptimal / requested = {med:.3f}× median "
                    f"— and the scatter about it is unit-variance noise")
        b.legend(fontsize=8)
        b.grid(alpha=0.3)

        c.hist(neg_std, bins=40, alpha=0.65, color="0.5", label="negatives", density=True)
        c.hist(pos_std, bins=40, alpha=0.65, color="tab:red", label="positives", density=True)
        c.axvline(1.0, color="k", lw=1.5, ls="--", label="unit variance")
        c.set_xlabel("per-segment std [σ]")
        c.set_ylabel("density")
        c.set_title("Positives are louder — by exactly the signal's own energy\n"
                    f"(measured / predicted = {excess:.3f}×; a per-segment rescale would flatten both to 1.0)")
        c.legend(fontsize=8)
        c.grid(alpha=0.3)

        d.hist(snr[pos], bins=40, color="tab:blue", alpha=0.75)
        d.set_xlabel("injected SNR")
        d.set_ylabel("segments")
        d.set_title(f"Training SNR distribution — U({attrs['snr_low']:.0f}, {attrs['snr_high']:.0f})\n"
                    f"{int(pos.sum()):,} positives / {int(neg.sum()):,} negatives, "
                    f"splits {counts[0]:,}/{counts[1]:,}/{counts[2]:,}")
        d.grid(alpha=0.3)

        fig.tight_layout()
        path = OUT / "4_dataset_check.png"
        fig.savefig(path, dpi=130)
        print(f"\n  plot -> {path}")

    failed = [name for name, ok, _ in checks if not ok]
    print("\n" + "=" * 66)
    if failed:
        print(f"  FAIL — {len(failed)} check(s): " + "; ".join(failed))
        return 1
    print(f"  PASS — all {len(checks)} checks. The dataset is what it claims to be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

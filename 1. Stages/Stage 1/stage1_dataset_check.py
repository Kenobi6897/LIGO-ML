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
     is exactly C(h), with no noise term. Conditioned noise has unit variance and is white
     in band, so the optimal SNR of that signal is just ||C(h)||_2. It must come back as
     the SNR we asked for. If it doesn't, every point on the efficiency-vs-SNR curve is
     mislabelled and the curve is fiction.

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
from stage1_dataset import DEFAULT_OUT, build_segment, make_specs, place_in_buffer
from stage1_injection_check import make_waveform
from stage1_noise_check import SAMPLE_RATE

import h5py

OUT = Path(__file__).parent / "outputs"
N_REPRO = 12  # rows rebuilt bit-for-bit — each one costs a full segment build
N_SNR = 150  # rows whose SNR calibration we verify


def signal_only(m1: float, m2: float, merger_pos: float) -> np.ndarray:
    """C(h): the conditioned waveform, with no noise.

    Legal precisely because `condition()` is signal-blind (Step 3). Feed the leaky
    whitener a noiseless signal and it is not even well-defined — which is the point.
    """
    placed = place_in_buffer(make_waveform(float(m1), float(m2)), float(merger_pos))
    return condition(placed)


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
            and np.allclose(specs.snr, snr, atol=1e-6)
            and np.array_equal(specs.split, split)
        )
        print(f"  {'PASS' if specs_match else 'FAIL'}  metadata matches a fresh draw from seed "
              f"{attrs['dataset_seed']}")
        checks.append(("metadata reproducible from the dataset seed", specs_match, ""))

        # --- 3. SNR calibration ------------------------------------------------
        print(f"\n[3] SNR calibration on {N_SNR} positives — ||C(h)|| vs the SNR we asked for")
        pos_rows = rng.choice(np.flatnonzero(pos), size=min(N_SNR, int(pos.sum())), replace=False)
        want = snr[pos_rows]
        got = np.empty(len(pos_rows))
        mf = np.empty(len(pos_rows))
        for i, r in enumerate(pos_rows):
            t = signal_only(m1[r], m2[r], merger[r])
            got[i] = np.linalg.norm(t)
            # Matched filter of the *stored* segment against that template: peak over lag.
            # Unit-variance white noise in band, so this is an SNR directly.
            x = X[r, 0]
            mf[i] = np.abs(np.correlate(x, t / np.linalg.norm(t), mode="same")).max()

        ratio = got / want
        med = float(np.median(ratio))
        snr_ok = 0.85 <= med <= 1.15 and float(np.std(ratio)) < 0.15
        print(f"  ||C(h)|| / requested SNR:  median {med:.3f}  "
              f"(5th-95th {np.percentile(ratio,5):.3f}-{np.percentile(ratio,95):.3f})")
        print(f"  {'PASS' if snr_ok else 'FAIL'}  the x-axis of the money plot means what it says")
        checks.append(("injected SNR is calibrated in-band", snr_ok, f"median {med:.3f}x"))

        # --- 4. variance: signal energy, and nothing else -----------------------
        print("\n[4] segment variance — signal energy only, no per-segment rescale")
        neg_rows = rng.choice(np.flatnonzero(neg), size=400, replace=False)
        neg_std = np.array([X[int(r), 0].std() for r in neg_rows])
        pos_std = np.array([X[int(r), 0].std() for r in pos_rows])
        # Predicted: var = 1 + rho^2/N for a signal of SNR rho added to unit-variance noise.
        pred = np.sqrt(1.0 + want**2 / CROP_N)
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
            a.plot(t_ax, signal_only(m1[r], m2[r], merger[r]), lw=1.4, color=colour, alpha=0.9,
                   ls="--", label=f"C(h) alone, SNR {snr[r]:.1f}")
        a.set_xlabel("time [s]")
        a.set_ylabel("conditioned strain [σ]")
        a.set_title("What the CNN eats\n(the SNR-4 chirp is invisible by eye — that is the point)")
        a.legend(fontsize=7, loc="upper left")
        a.grid(alpha=0.3)

        b.scatter(want, got, s=14, alpha=0.5, color="tab:blue", label="‖C(h)‖ — signal in the CNN's input")
        b.scatter(want, mf, s=10, alpha=0.35, color="tab:orange", label="matched filter on the stored segment")
        lim = [0, max(want.max(), mf.max()) * 1.05]
        b.plot(lim, lim, "k--", lw=1, label="ideal")
        b.set_xlabel("requested injected SNR (stored in `snr`)")
        b.set_ylabel("recovered SNR")
        b.set_title(f"The money plot's x-axis is calibrated\n‖C(h)‖ / requested = {med:.3f}× median")
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
        d.set_title(f"Training SNR distribution — U{tuple(np.round([attrs['snr_low'], attrs['snr_high']],0))}\n"
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
